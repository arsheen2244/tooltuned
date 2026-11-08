"""Merge the trained LoRA adapter into the base model, so tools like BFCL's vLLM can load it."""

import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel

from tooltuned.train import create_private_repo

MERGED_MODEL_NAME = "tooltuned-qwen2.5-1.5b"

# Copied unchanged from the base model, because fine-tuning doesn't change them and newer
# transformers versions save them in a format BFCL's older transformers and vLLM misread.
BASE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def account_repo_id(repo_name: str, token: str) -> str:
    """Return a repo name under the token's own account, like "someone/repo_name"."""
    return f"{HfApi(token=token).whoami()['name']}/{repo_name}"


def merge_adapter(
    base_model: str,
    adapter: str,
    token: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> PreTrainedModel:
    """Load the full-precision base model, add the adapter's weights into it, and drop the adapter.

    The base model is loaded in fp16 rather than 4 bits, because merging into rounded 4-bit
    weights would lose accuracy. The result is an ordinary model with no LoRA layers left.
    """
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, token=token)
    with_adapter = PeftModel.from_pretrained(model, adapter, token=token)
    return with_adapter.merge_and_unload()


def base_file(base_model: str, name: str, token: str | None) -> Path | None:
    """Find one of the base model's files. A local folder may lack some; on the Hub all must exist.

    Errors from the Hub are not caught, so a failed download stops the merge rather than
    silently leaving a newer-format file in place.
    """
    if Path(base_model).is_dir():
        path = Path(base_model) / name
        return path if path.exists() else None
    return Path(hf_hub_download(base_model, name, token=token))


def save_merged(
    model: PreTrainedModel, base_model: str, save_dir: Path, token: str | None = None
) -> None:
    """Save the merged weights next to the base model's original config and tokenizer files.

    Anything already in save_dir is removed first, so no file from an older run is left.
    """
    shutil.rmtree(save_dir, ignore_errors=True)
    model.save_pretrained(save_dir)
    for name in BASE_FILES:
        source = base_file(base_model, name, token)
        if source is not None:
            shutil.copyfile(source, save_dir / name)
    (save_dir / "chat_template.jinja").unlink(missing_ok=True)
    config_path = save_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["torch_dtype"] = str(model.dtype).removeprefix("torch.")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def push_folder_private(folder: Path, repo_name: str, token: str) -> str:
    """Upload a folder to a private repo, deleting any older files the folder doesn't have."""
    api = HfApi(token=token)
    repo_id = create_private_repo(api, repo_name)
    api.upload_folder(repo_id=repo_id, folder_path=folder, delete_patterns="*")
    return repo_id

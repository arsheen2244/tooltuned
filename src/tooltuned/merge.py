"""Merge the trained LoRA adapter into the base model, so tools like BFCL's vLLM can load it."""

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel

MERGED_MODEL_NAME = "tooltuned-qwen2.5-1.5b"


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

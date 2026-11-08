"""Tests for merging a LoRA adapter into its base model, using a tiny random model on the CPU."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import Qwen2Config

from tooltuned import merge
from tooltuned.merge import account_repo_id, merge_adapter, push_folder_private, save_merged

INPUT_IDS = torch.tensor([[1, 5, 9, 3, 7]])


@pytest.fixture(scope="module")
def tiny_adapter(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Save a tiny base model and an adapter whose weights are not zero, so merging changes it."""
    root = tmp_path_factory.mktemp("merge")
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(root / "base")
    base = AutoModelForCausalLM.from_pretrained(root / "base")
    lora = LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", init_lora_weights=False)
    get_peft_model(base, lora).save_pretrained(root / "adapter")
    return root / "base", root / "adapter"


def _logits(model: Any) -> torch.Tensor:
    with torch.no_grad():
        return model(INPUT_IDS).logits


def test_merged_model_gives_the_same_outputs_as_base_plus_adapter(
    tiny_adapter: tuple[Path, Path],
) -> None:
    base_dir, adapter_dir = tiny_adapter
    merged = merge_adapter(str(base_dir), str(adapter_dir), dtype=torch.float32)
    with_adapter = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_dir), adapter_dir
    )
    plain_base = AutoModelForCausalLM.from_pretrained(base_dir)
    assert torch.allclose(_logits(merged), _logits(with_adapter), atol=1e-5)
    assert not torch.allclose(_logits(merged), _logits(plain_base), atol=1e-3)


def test_merged_model_has_no_lora_layers_left(tiny_adapter: tuple[Path, Path]) -> None:
    merged = merge_adapter(*map(str, tiny_adapter), dtype=torch.float32)
    assert not isinstance(merged, PeftModel)
    assert not any("lora_" in name for name, _ in merged.named_parameters())


def test_merged_model_is_fp16_by_default(tiny_adapter: tuple[Path, Path]) -> None:
    merged = merge_adapter(*map(str, tiny_adapter))
    assert {parameter.dtype for parameter in merged.parameters()} == {torch.float16}


def test_save_merged_writes_a_normal_model_that_loads_back(
    tiny_adapter: tuple[Path, Path], tmp_path: Path
) -> None:
    base_dir, adapter_dir = tiny_adapter
    merged = merge_adapter(str(base_dir), str(adapter_dir), dtype=torch.float32)
    save_merged(merged, str(base_dir), tmp_path / "out")
    assert (tmp_path / "out" / "model.safetensors").exists()
    assert not (tmp_path / "out" / "adapter_config.json").exists()
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "out")
    assert torch.allclose(_logits(reloaded), _logits(merged), atol=1e-5)


def test_save_merged_uses_the_base_models_own_config(
    tiny_adapter: tuple[Path, Path], tmp_path: Path
) -> None:
    base_dir, adapter_dir = tiny_adapter
    original = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    original["marker"] = "from the base model"
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "config.json").write_text(json.dumps(original), encoding="utf-8")

    save_merged(merge_adapter(str(base_dir), str(adapter_dir)), str(tmp_path / "base"), tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    assert saved == {**original, "torch_dtype": "float16"}


def test_save_merged_removes_files_from_an_older_run(
    tiny_adapter: tuple[Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "chat_template.jinja").write_text("old", encoding="utf-8")
    save_merged(merge_adapter(*map(str, tiny_adapter)), str(tiny_adapter[0]), out)
    assert not (out / "chat_template.jinja").exists()


def test_push_folder_private_uploads_to_a_private_repo_and_clears_old_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    class FakeApi:
        def __init__(self, token: str) -> None:
            pass

        def whoami(self) -> dict[str, str]:
            return {"name": "someone"}

        def create_repo(self, repo_id: str, **kwargs: Any) -> None:
            calls["create"] = kwargs

        def repo_info(self, repo_id: str) -> Any:
            return type("Info", (), {"private": True})

        def upload_folder(self, **kwargs: Any) -> None:
            calls["upload"] = kwargs

    monkeypatch.setattr(merge, "HfApi", FakeApi)
    assert push_folder_private(tmp_path, "model", "token") == "someone/model"
    assert calls["create"]["private"] is True
    assert calls["upload"] == {
        "repo_id": "someone/model", "folder_path": tmp_path, "delete_patterns": "*"
    }


def test_account_repo_id_uses_the_tokens_account(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApi:
        def __init__(self, token: str) -> None:
            assert token == "token"

        def whoami(self) -> dict[str, str]:
            return {"name": "someone"}

    monkeypatch.setattr(merge, "HfApi", FakeApi)
    assert account_repo_id("model", "token") == "someone/model"

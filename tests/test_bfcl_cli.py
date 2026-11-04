"""Tests for the BFCL wrapper, using stand-ins so BFCL needn't be installed."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bfcl_cli.py"


class FakeHandler:
    """Mirrors QwenFCHandler, which leaves dtype at bf16 whatever it's passed."""

    def __init__(self, **kwargs: Any) -> None:
        self.dtype = "bfloat16"


class FakeTokenizer:
    def __init__(self) -> None:
        self.call: tuple[Any, dict[str, Any]] | None = None

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.call = (messages, kwargs)
        return "PROMPT"


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load scripts/bfcl_cli.py with BFCL's modules replaced by stand-ins."""
    fakes = {
        "bfcl_eval": {},
        "bfcl_eval.__main__": {"cli": lambda: None},
        "bfcl_eval.constants": {},
        "bfcl_eval.constants.model_config": {
            "MODEL_CONFIG_MAPPING": {},
            "ModelConfig": SimpleNamespace,
        },
        "bfcl_eval.model_handler": {},
        "bfcl_eval.model_handler.local_inference": {},
        "bfcl_eval.model_handler.local_inference.qwen_fc": {
            "QwenFCHandler": FakeHandler
        },
        "overrides": {"override": lambda function: function},
    }
    for name, attributes in fakes.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("bfcl_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handler_uses_fp16(wrapper: ModuleType) -> None:
    assert wrapper.Qwen25FCHandler().dtype == "float16"


def test_register_model_points_at_qwen25_handler(wrapper: ModuleType) -> None:
    wrapper.register_model()
    config = wrapper.MODEL_CONFIG_MAPPING[wrapper.BFCL_MODEL_NAME]
    assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.model_handler is wrapper.Qwen25FCHandler
    assert config.is_fc_model is True


@pytest.mark.parametrize(
    ("functions", "expected_tools"),
    [([{"name": "f"}], [{"name": "f"}]), ([], None)],
)
def test_prompt_uses_tokenizer_chat_template(
    wrapper: ModuleType, functions: list[dict[str, str]], expected_tools: Any
) -> None:
    handler = wrapper.Qwen25FCHandler()
    handler.tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]

    assert handler._format_prompt(messages, functions) == "PROMPT"
    assert handler.tokenizer.call == (
        messages,
        {"tools": expected_tools, "add_generation_prompt": True, "tokenize": False},
    )

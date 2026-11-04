"""Run the BFCL command line with Qwen2.5-1.5B-Instruct registered as a model."""

from typing import Any

from bfcl_eval.__main__ import cli
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override

BFCL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct-FC"
HF_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class Qwen25FCHandler(QwenFCHandler):
    """Qwen FC handler using fp16 and Qwen2.5's own chat template."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dtype = "float16"  # The T4 doesn't support bf16, BFCL's default.

    @override
    def _format_prompt(self, messages, function):
        return self.tokenizer.apply_chat_template(
            messages,
            tools=function or None,
            add_generation_prompt=True,
            tokenize=False,
        )


def register_model() -> None:
    """Add the model to BFCL's registry, which generate and evaluate both read."""
    MODEL_CONFIG_MAPPING[BFCL_MODEL_NAME] = ModelConfig(
        model_name=HF_MODEL_NAME,
        display_name="Qwen2.5-1.5B-Instruct (FC)",
        url=f"https://huggingface.co/{HF_MODEL_NAME}",
        org="Qwen",
        license="apache-2.0",
        model_handler=Qwen25FCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )


if __name__ == "__main__":
    register_model()
    cli()

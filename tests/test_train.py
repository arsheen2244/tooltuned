"""Tests for the training setup, including a real run of a tiny model on the CPU."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen2 import Qwen2Config

from tooltuned import train
from tooltuned.data import BASE_MODEL, render_example, to_hub_row
from tooltuned.formatting import token_length
from tooltuned.train import (
    TrainConfig,
    build_lora_config,
    build_quantization_config,
    build_trainer_dataset,
    build_training_args,
    expected_answer,
    force_adapter_float32,
    is_quick_run,
    load_config,
    push_adapter,
    train_on_rows,
    wandb_report_target,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "qlora_r16.yaml"
TOOL = {
    "name": "get_weather",
    "description": "Get the weather.",
    "parameters": {"city": {"description": "City name.", "type": "str"}},
}
CALL = {"name": "get_weather", "arguments": {"city": "Paris"}}


def tool_row(row_id: int = 0, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    answer = {**CALL, "arguments": arguments or CALL["arguments"]}
    return to_hub_row(
        {
            "id": row_id,
            "query": f"What's the weather in Paris? {row_id}",
            "answers": json.dumps([answer]),
            "tools": json.dumps([TOOL]),
            "reply": "",
        }
    )


def no_tool_row(row_id: int = 100) -> dict[str, Any]:
    return to_hub_row(
        {
            "id": row_id,
            "query": f"Tell me a joke. {row_id}",
            "answers": "[]",
            "tools": json.dumps([TOOL]),
            "reply": "None of the available tools fit this request.",
        }
    )


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL)
    except OSError as error:
        pytest.skip(f"Qwen2.5 tokenizer unavailable: {error}")


@pytest.fixture
def config() -> TrainConfig:
    return load_config(CONFIG_PATH)


def test_config_file_has_the_requested_values(config: TrainConfig) -> None:
    expected = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "all-linear",
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "fp16": True,
        "gradient_checkpointing": True,
        "max_seq_length": 1072,
        "eval_steps": 200,
        "seed": 42,
        "hub_model_name": "tooltuned-qwen2.5-1.5b-lora",
    }
    assert {name: getattr(config, name) for name in expected} == expected


def test_config_values_have_their_declared_number_types(config: TrainConfig) -> None:
    # YAML reads a value like 2e-4 as text, which would only fail later in training.
    assert isinstance(config.learning_rate, float)
    assert isinstance(config.warmup_ratio, float)
    assert isinstance(config.lora_dropout, float)


def test_unknown_config_key_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("learning_rat: 0.1\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(path)


@pytest.mark.parametrize(
    ("set_key", "expected"), [(True, "wandb"), (False, "none")]
)
def test_wandb_logging_follows_the_api_key(
    monkeypatch: pytest.MonkeyPatch, set_key: bool, expected: str
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    if set_key:
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
    assert wandb_report_target() == expected


def test_wandb_runs_go_to_this_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    wandb_report_target()
    assert train.os.environ["WANDB_PROJECT"] == "tooltuned"


@pytest.mark.parametrize(
    ("max_steps", "train_limit", "expected"),
    [(None, None, False), (20, None, True), (None, 200, True), (20, 200, True)],
)
def test_is_quick_run(max_steps: int | None, train_limit: int | None, expected: bool) -> None:
    assert is_quick_run(max_steps, train_limit) is expected


def test_lora_config_matches_the_settings(config: TrainConfig) -> None:
    lora = build_lora_config(config)
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (16, 32, 0.05)
    assert lora.target_modules == "all-linear"


def test_quantization_config_is_4bit_nf4_with_fp16_maths(config: TrainConfig) -> None:
    quantization = build_quantization_config(config)
    assert quantization.load_in_4bit
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert quantization.bnb_4bit_compute_dtype == torch.float16


def _cpu_config(config: TrainConfig, tmp_path: Path, **changes: Any) -> TrainConfig:
    return dataclasses.replace(
        config, fp16=False, output_dir=str(tmp_path / "out"), **changes
    )


def test_training_args_match_the_settings(config: TrainConfig, tmp_path: Path) -> None:
    args = build_training_args(_cpu_config(config, tmp_path), None)
    assert args.learning_rate == 2e-4
    assert args.lr_scheduler_type.value == "cosine"
    assert args.get_warmup_steps(1000) == 30
    assert args.num_train_epochs == 2
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 4
    assert args.max_length == 1072
    assert args.eval_strategy.value == "steps"
    assert args.eval_steps == 200
    assert args.seed == 42
    assert args.eos_token == "<|im_end|>"
    assert args.max_steps == -1
    assert args.bf16 is False


def test_quick_run_evaluates_before_it_ends(config: TrainConfig, tmp_path: Path) -> None:
    args = build_training_args(_cpu_config(config, tmp_path), 20)
    assert args.max_steps == 20
    assert args.eval_steps == 20


def test_force_adapter_float32_only_changes_trainable_weights() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model[0].requires_grad_(False)
    model.to(torch.bfloat16)
    force_adapter_float32(model)
    assert model[0].weight.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float32


def test_expected_answer_uses_reply_or_tool_calls() -> None:
    assert expected_answer(no_tool_row()).startswith("None of")
    assert "get_weather" in expected_answer(tool_row())


def test_dataset_keeps_rows_at_the_limit_and_drops_longer_ones(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    length = token_length(tool_row(), tokenizer)
    _, dropped = build_trainer_dataset([tool_row()], tokenizer, length)
    assert dropped == 0
    dataset, dropped = build_trainer_dataset([tool_row()], tokenizer, length - 1)
    assert dropped == 1
    assert len(dataset) == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"city": "Paris"},
        {"days": 3, "places": ["a", {"b": 1}]},
        # These became 0.30000000000000004 and similar through TRL's documented Json column.
        {"od": 0.3, "weights": [0.4, 0.3, 0.6], "area": 0.0003},
    ],
)
def test_dataset_keeps_each_rows_arguments_exactly(
    tokenizer: PreTrainedTokenizerBase, arguments: dict[str, Any]
) -> None:
    row = tool_row(0, arguments)
    dataset, _ = build_trainer_dataset([row], tokenizer, 2000)
    assert tokenizer.decode(dataset[0]["input_ids"]) == render_example(row, tokenizer)
    assert json.dumps(arguments) in tokenizer.decode(dataset[0]["input_ids"])


def test_dataset_marks_assistant_tokens_for_both_kinds_of_row(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    dataset, _ = build_trainer_dataset([tool_row(), no_tool_row()], tokenizer, 2000)
    for item, answer in zip(dataset, ("<tool_call>", "None of the available")):
        marked = [t for t, mask in zip(item["input_ids"], item["assistant_masks"]) if mask]
        assert tokenizer.decode(marked).strip().startswith(answer)
        assert tokenizer.decode(marked).strip().endswith("<|im_end|>")


class FakeUploader:
    def __init__(self) -> None:
        self.pushes: list[tuple[str, dict[str, Any]]] = []

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> None:
        self.pushes.append((repo_id, kwargs))


class FakeApi:
    is_private = True
    created: list[dict[str, Any]] = []

    def __init__(self, token: str) -> None:
        pass

    def whoami(self) -> dict[str, str]:
        return {"name": "someone"}

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def repo_info(self, repo_id: str) -> Any:
        return type("Info", (), {"private": self.is_private})


def test_push_adapter_uses_a_private_repo_under_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeApi.is_private = True
    FakeApi.created = []
    monkeypatch.setattr(train, "HfApi", FakeApi)
    model, tokenizer = FakeUploader(), FakeUploader()
    assert push_adapter(model, tokenizer, "name", "token") == "someone/name"
    assert FakeApi.created[0]["private"] is True
    assert model.pushes == [("someone/name", {"private": True, "token": "token"})]
    assert len(tokenizer.pushes) == 1


def test_push_adapter_refuses_an_existing_public_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeApi.is_private = False
    monkeypatch.setattr(train, "HfApi", FakeApi)
    model = FakeUploader()
    with pytest.raises(RuntimeError, match="public"):
        push_adapter(model, FakeUploader(), "name", "token")
    assert model.pushes == []


@pytest.fixture(scope="module")
def tiny_run(tokenizer: PreTrainedTokenizerBase, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Really train a tiny random Qwen2 model for a few steps on the CPU."""
    root = tmp_path_factory.mktemp("tiny")
    model_config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    model_dir = root / "model"
    AutoModelForCausalLM.from_config(model_config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    base = load_config(CONFIG_PATH)
    config = dataclasses.replace(
        base,
        base_model=str(model_dir),
        load_in_4bit=False,
        fp16=False,
        output_dir=str(root / "out"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=1000,
        num_generation_examples=2,
        max_new_tokens=5,
    )
    rows = [tool_row(i) for i in range(6)] + [no_tool_row(100 + i) for i in range(2)]
    validation = [tool_row(50), no_tool_row(150)]
    trainer, trained_tokenizer = train_on_rows(config, rows, validation, max_steps=3)
    return trainer, trained_tokenizer, config


def _decoded_labels(trainer: Any, tokenizer: Any, index: int) -> str:
    labels = trainer.train_dataset[index]["labels"]
    return tokenizer.decode([token for token in labels if token != -100])


def test_tiny_run_only_computes_loss_on_the_assistant_tool_call(tiny_run: Any) -> None:
    trainer, tokenizer, _ = tiny_run
    assert _decoded_labels(trainer, tokenizer, 0).strip() == (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        "</tool_call><|im_end|>"
    )


def test_tiny_run_only_computes_loss_on_the_no_tool_reply(tiny_run: Any) -> None:
    trainer, tokenizer, _ = tiny_run
    index = len(trainer.train_dataset) - 1
    assert _decoded_labels(trainer, tokenizer, index).strip() == (
        "None of the available tools fit this request.<|im_end|>"
    )


def test_tiny_run_training_inputs_match_the_inference_prompt(tiny_run: Any) -> None:
    # TRL's training template must render the same text as the model's own template.
    trainer, tokenizer, _ = tiny_run
    row = tool_row(0)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["query"]}],
        tools=json.loads(row["tools"]),
        add_generation_prompt=True,
        tokenize=False,
    )
    assert tokenizer.decode(trainer.train_dataset[0]["input_ids"]).startswith(prompt)


def test_tiny_run_trains_and_evaluates(tiny_run: Any) -> None:
    trainer, _, _ = tiny_run
    history = trainer.state.log_history
    assert trainer.state.global_step == 3
    assert any("loss" in entry for entry in history)
    assert any("eval_loss" in entry for entry in history)


def test_tiny_run_saves_only_an_adapter(tiny_run: Any) -> None:
    _, _, config = tiny_run
    output = Path(config.output_dir)
    assert (output / "adapter_config.json").exists()
    assert (output / "adapter_model.safetensors").exists()
    assert not (output / "model.safetensors").exists()


def test_tiny_run_only_trains_adapter_weights_as_float32(tiny_run: Any) -> None:
    trainer, _, _ = tiny_run
    trainable = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert all(parameter.dtype == torch.float32 for parameter in trainable.values())


def test_tiny_run_prints_generated_examples(tiny_run: Any, capfd: pytest.CaptureFixture) -> None:
    trainer, tokenizer, config = tiny_run
    train.print_generated_examples(trainer.model, tokenizer, [tool_row(7)], config.max_new_tokens)
    output = capfd.readouterr().out
    assert "QUERY:" in output and "EXPECTED:" in output and "GENERATED:" in output

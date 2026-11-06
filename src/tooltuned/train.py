"""QLoRA fine-tuning."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer
from trl.chat_template_utils import get_training_chat_template

from tooltuned.data import Row, load_hub_split

WANDB_PROJECT = "tooltuned"
END_OF_TURN = "<|im_end|>"


@dataclass(frozen=True)
class TrainConfig:
    """Training settings, read from a YAML file. Unknown or missing keys are errors."""

    base_model: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    fp16: bool
    gradient_checkpointing: bool
    max_seq_length: int
    eval_steps: int
    logging_steps: int
    save_steps: int
    save_total_limit: int
    seed: int
    dataset_config: str
    output_dir: str
    hub_model_name: str
    num_generation_examples: int
    max_new_tokens: int


def load_config(path: Path) -> TrainConfig:
    """Read a training config from a YAML file."""
    return TrainConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


def wandb_report_target() -> str:
    """Log to Weights & Biases only if its key is set, under this project's name."""
    if not os.environ.get("WANDB_API_KEY"):
        return "none"
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    return "wandb"


def is_quick_run(max_steps: int | None, train_limit: int | None) -> bool:
    """A run shortened by either flag is a test, not a real result."""
    return max_steps is not None or train_limit is not None


def tokenize_row(
    row: Row, tokenizer: PreTrainedTokenizerBase, chat_template: str | None
) -> dict[str, list[int]]:
    """Tokenize a row and mark which tokens belong to the assistant's answer."""
    tokens = tokenizer.apply_chat_template(
        json.loads(row["messages"]),
        tools=json.loads(row["tools"]),
        chat_template=chat_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    return {"input_ids": tokens["input_ids"], "assistant_masks": tokens["assistant_masks"]}


def build_trainer_dataset(
    rows: list[Row], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> tuple[Dataset, int]:
    """Tokenize rows for TRL, keeping those that fit whole, and count the rest.

    Long rows are dropped rather than truncated, because the answer the model must learn
    comes last and truncating would cut it off.

    Rows are tokenized here because TRL's documented Json column changes numbers in
    function arguments (0.3 becomes 0.30000000000000004), which would train the model on
    wrong values. TRL's training template marks the assistant's tokens, and TRL builds
    its loss labels from those masks.
    """
    template = get_training_chat_template(tokenizer)
    tokenized = [tokenize_row(row, tokenizer, template) for row in rows]
    kept = [tokens for tokens in tokenized if len(tokens["input_ids"]) <= max_length]
    return Dataset.from_list(kept), len(rows) - len(kept)


def build_lora_config(config: TrainConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )


def build_quantization_config(config: TrainConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_base_model(config: TrainConfig) -> PreTrainedModel:
    """Load the model, in 4 bits and on the first GPU when the config asks for it."""
    if not config.load_in_4bit:
        return AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=build_quantization_config(config),
        dtype=torch.float16,
        device_map={"": 0},
    )
    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def build_training_args(config: TrainConfig, max_steps: int | None) -> SFTConfig:
    """Build TRL's settings. Quick runs evaluate before they end, even if very short."""
    eval_steps = min(config.eval_steps, max_steps) if max_steps else config.eval_steps
    return SFTConfig(
        output_dir=config.output_dir,
        max_steps=max_steps if max_steps else -1,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_ratio,  # Values below 1 are read as a share of all updates.
        fp16=config.fp16,
        bf16=False,  # SFTConfig would otherwise turn this on when fp16 is off.
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=config.max_seq_length,
        # Off because TRL only allows it on untokenized data. Loss is still assistant-only,
        # through the assistant_masks column from build_trainer_dataset.
        assistant_only_loss=False,
        eos_token=END_OF_TURN,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        report_to=wandb_report_target(),
    )


def force_adapter_float32(model: PreTrainedModel) -> None:
    """Undo TRL's cast of QLoRA adapters to bf16, which the T4 can't run.

    SFTTrainer casts them whatever fp16 is set to (sft_trainer.py in trl 1.17.1). The
    fp16 gradient scaler also needs float32 weights, so they have to stay float32.
    """
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)


def build_trainer(
    config: TrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_steps: int | None,
) -> SFTTrainer:
    trainer = SFTTrainer(
        model=load_base_model(config),
        args=build_training_args(config, max_steps),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(config),
    )
    force_adapter_float32(trainer.model)
    return trainer


def expected_answer(row: Row) -> str:
    """Return a row's correct output, either its plain reply or its tool calls."""
    return row["reply"] or row["answers"]


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    row: Row,
    max_new_tokens: int,
) -> str:
    """Generate the model's answer to one row, given only its query and tools."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["query"]}],
        tools=json.loads(row["tools"]),
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    use_autocast = model.device.type == "cuda"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def print_generated_examples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[Row],
    max_new_tokens: int,
) -> None:
    model.eval()
    for index, row in enumerate(rows):
        print(f"\n===== VALIDATION EXAMPLE {index} =====")
        print(f"QUERY:\n{row['query']}\n")
        print(f"EXPECTED:\n{expected_answer(row)}\n")
        print(f"GENERATED:\n{generate_answer(model, tokenizer, row, max_new_tokens)}")


def push_adapter(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, repo_name: str, token: str
) -> str:
    """Push the adapter and tokenizer to a private repo under the token's account."""
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{repo_name}"
    api.create_repo(repo_id, private=True, exist_ok=True)
    if not api.repo_info(repo_id).private:
        raise RuntimeError(f"{repo_id} already exists and is public, so nothing was pushed.")
    model.push_to_hub(repo_id, private=True, token=token)
    tokenizer.push_to_hub(repo_id, private=True, token=token)
    return repo_id


def train_on_rows(
    config: TrainConfig,
    train_rows: list[Row],
    validation_rows: list[Row],
    max_steps: int | None = None,
) -> tuple[SFTTrainer, PreTrainedTokenizerBase]:
    """Train and save an adapter, then print generated answers."""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    train_dataset, train_dropped = build_trainer_dataset(
        train_rows, tokenizer, config.max_seq_length
    )
    eval_dataset, eval_dropped = build_trainer_dataset(
        validation_rows, tokenizer, config.max_seq_length
    )
    print(f"Train rows: {len(train_dataset)} ({train_dropped} too long, dropped)")
    print(f"Validation rows: {len(eval_dataset)} ({eval_dropped} too long, dropped)")

    trainer = build_trainer(config, tokenizer, train_dataset, eval_dataset, max_steps)
    trainer.train()
    trainer.save_model(config.output_dir)
    print_generated_examples(
        trainer.model,
        tokenizer,
        validation_rows[: config.num_generation_examples],
        config.max_new_tokens,
    )
    return trainer, tokenizer


def run_training(
    config: TrainConfig,
    max_steps: int | None = None,
    train_limit: int | None = None,
    push: bool = True,
) -> None:
    """Load the Hub dataset, train, and push the adapter unless this is a test."""
    rows: list[dict[str, Any]] = list(load_hub_split(config.dataset_config, "train"))
    validation = list(load_hub_split(config.dataset_config, "validation"))
    trainer, tokenizer = train_on_rows(config, rows[:train_limit], validation, max_steps)

    if is_quick_run(max_steps, train_limit) or not push:
        print(f"\nNot pushing to the Hub. The adapter is in {config.output_dir}")
        return
    repo_id = push_adapter(
        trainer.model, tokenizer, config.hub_model_name, os.environ["HF_TOKEN"]
    )
    print(f"\nPushed the adapter to {repo_id}")

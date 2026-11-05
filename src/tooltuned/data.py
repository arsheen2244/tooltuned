"""Load and prepare the function-calling training data."""

import json
import os
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

DATASET_NAME = "Salesforce/xlam-function-calling-60k"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

Row = dict[str, Any]
Message = dict[str, Any]


def load_raw_dataset() -> Dataset:
    """Download the dataset, which is gated and so needs HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set the HF_TOKEN environment variable to load the dataset.")
    return load_dataset(DATASET_NAME, split="train", token=token)


def row_to_messages(row: Row) -> list[Message]:
    """Turn one dataset row into a user message and an assistant tool-call message."""
    # arguments stays a dict: the chat template converts it to JSON itself.
    tool_calls = [
        {
            "type": "function",
            "function": {"name": answer["name"], "arguments": answer["arguments"]},
        }
        for answer in json.loads(row["answers"])
    ]
    return [
        {"role": "user", "content": row["query"]},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
    ]


def render_example(row: Row, tokenizer: PreTrainedTokenizerBase) -> str:
    """Return the exact text the model will learn from for one row."""
    return tokenizer.apply_chat_template(
        row_to_messages(row),
        tools=json.loads(row["tools"]),
        tokenize=False,
    )


def has_usable_fields(row: Row) -> bool:
    """Check that tools and answers parse and that every answer is a complete call."""
    try:
        tools = json.loads(row["tools"])
        answers = json.loads(row["answers"])
    except (TypeError, json.JSONDecodeError):
        return False
    if not (isinstance(tools, list) and tools and isinstance(answers, list) and answers):
        return False
    return all(
        isinstance(answer, dict)
        and isinstance(answer.get("name"), str)
        and isinstance(answer.get("arguments"), dict)
        for answer in answers
    )

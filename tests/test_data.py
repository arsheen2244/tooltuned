"""Tests for turning dataset rows into chat messages and training text."""

import json
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tooltuned import data
from tooltuned.data import (
    BASE_MODEL,
    DATASET_NAME,
    has_usable_fields,
    load_raw_dataset,
    render_example,
    row_to_messages,
)

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {"city": {"description": "City name.", "type": "str"}},
    }
]


def make_row(*answers: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": 0,
        "query": "What's the weather in Paris and Rome?",
        "answers": json.dumps(list(answers)),
        "tools": json.dumps(TOOLS),
    }


PARIS = {"name": "get_weather", "arguments": {"city": "Paris"}}
ROME = {"name": "get_weather", "arguments": {"city": "Rome"}}


class FakeTokenizer:
    def __init__(self) -> None:
        self.call: tuple[Any, dict[str, Any]] | None = None

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.call = (messages, kwargs)
        return "RENDERED"


def test_row_to_messages_builds_user_and_assistant() -> None:
    row = make_row(PARIS)
    assert row_to_messages(row) == [
        {"role": "user", "content": row["query"]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
    ]


def test_row_to_messages_keeps_parallel_calls_in_order() -> None:
    calls = row_to_messages(make_row(PARIS, ROME))[1]["tool_calls"]
    assert [call["function"]["arguments"]["city"] for call in calls] == ["Paris", "Rome"]


def test_render_example_passes_parsed_tools_and_returns_text() -> None:
    row = make_row(PARIS)
    tokenizer = FakeTokenizer()
    assert render_example(row, tokenizer) == "RENDERED"
    assert tokenizer.call == (
        row_to_messages(row),
        {"tools": TOOLS, "tokenize": False},
    )


def test_load_raw_dataset_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        load_raw_dataset()


def test_load_raw_dataset_uses_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_load(name: str, **kwargs: Any) -> str:
        calls.append((name, kwargs))
        return "DATASET"

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(data, "load_dataset", fake_load)
    assert load_raw_dataset() == "DATASET"
    assert calls == [(DATASET_NAME, {"split": "train", "token": "test-token"})]


def test_has_usable_fields_accepts_complete_row() -> None:
    assert has_usable_fields(make_row(PARIS, ROME))


@pytest.mark.parametrize(
    "changes",
    [
        {"answers": "not json"},
        {"tools": "not json"},
        {"answers": None},
        {"answers": "[]"},
        {"tools": "[]"},
        {"answers": '{"name": "f"}'},
        {"answers": '[{"name": "f"}]'},
        {"answers": '[{"arguments": {}}]'},
        {"answers": '[{"name": "f", "arguments": "{}"}]'},
    ],
)
def test_has_usable_fields_rejects_bad_rows(changes: dict[str, Any]) -> None:
    assert not has_usable_fields({**make_row(PARIS), **changes})


@pytest.fixture(scope="module")
def qwen_tokenizer() -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL)
    except OSError as error:
        pytest.skip(f"Qwen2.5 tokenizer unavailable: {error}")


def test_real_template_renders_tools_query_and_calls(
    qwen_tokenizer: PreTrainedTokenizerBase,
) -> None:
    text = render_example(make_row(PARIS, ROME), qwen_tokenizer)
    assert json.dumps(TOOLS[0]) in text
    assert "<|im_start|>user\nWhat's the weather in Paris and Rome?<|im_end|>" in text
    assert (
        '<|im_start|>assistant\n<tool_call>\n'
        '{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n'
        "</tool_call><|im_end|>"
    ) in text

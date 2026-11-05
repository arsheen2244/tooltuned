"""Tests for turning dataset rows into chat messages and training text."""

import json
import random
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tooltuned import data
from tooltuned.data import (
    BASE_MODEL,
    DATASET_NAME,
    NO_TOOL_REPLIES,
    build_final_dataset,
    has_usable_fields,
    is_clearly_unrelated,
    load_raw_dataset,
    make_no_tool_example,
    make_no_tool_examples,
    no_tool_count,
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


def topic_row(row_id: int, query: str, name: str, description: str) -> dict[str, Any]:
    """Make a valid row whose only function has no parameters."""
    tool = {"name": name, "description": description, "parameters": {}}
    return {
        "id": row_id,
        "query": query,
        "answers": json.dumps([{"name": name, "arguments": {}}]),
        "tools": json.dumps([tool]),
    }


WEATHER = topic_row(0, "Is it raining in Paris?", "get_weather", "Forecast for a city.")
SUDOKU = topic_row(1, "Solve this puzzle.", "solve_sudoku", "Solves a sudoku grid.")
MOVIES = topic_row(2, "Who directed Alien?", "movie_credits", "Film cast and crew.")
PIZZA = topic_row(3, "Order me a margherita.", "order_pizza", "Places a takeaway order.")


def test_unrelated_tools_are_accepted() -> None:
    assert is_clearly_unrelated(WEATHER, SUDOKU)


def test_shared_function_name_is_rejected() -> None:
    donor = topic_row(1, "x", "get_weather", "Sudoku grids.")
    assert not is_clearly_unrelated(WEATHER, donor)


def test_tool_sharing_a_word_with_the_query_is_rejected() -> None:
    donor = topic_row(1, "x", "city_rainfall", "Whether it is raining anywhere.")
    assert not is_clearly_unrelated(WEATHER, donor)


def test_tool_sharing_a_word_with_the_original_tools_is_rejected() -> None:
    donor = topic_row(1, "x", "historic_forecast", "Past records.")
    assert not is_clearly_unrelated(WEATHER, donor)


def test_plural_and_camel_case_words_still_count_as_shared() -> None:
    donor = topic_row(1, "x", "cityForecasts", "Past records.")
    assert not is_clearly_unrelated(WEATHER, donor)


def test_generic_words_alone_do_not_link_tools() -> None:
    first = topic_row(0, "Find my film.", "get_movie", "Fetch data on a movie.")
    second = topic_row(1, "x", "get_sudoku", "Fetch data on a sudoku.")
    assert is_clearly_unrelated(first, second)


@pytest.mark.parametrize(
    ("name", "description"),
    [("web_search", "Looks up pages."), ("lookup", "Uses Google for anything.")],
)
def test_open_ended_tools_are_rejected(name: str, description: str) -> None:
    assert not is_clearly_unrelated(WEATHER, topic_row(1, "x", name, description))


def test_no_tool_example_uses_query_from_row_and_tools_from_donor() -> None:
    example = make_no_tool_example(WEATHER, SUDOKU, 99, random.Random(0))
    assert example["id"] == 99
    assert example["query"] == WEATHER["query"]
    assert example["tools"] == SUDOKU["tools"]
    assert example["answers"] == "[]"
    assert example["reply"] in NO_TOOL_REPLIES


def test_no_tool_messages_are_plain_text() -> None:
    example = make_no_tool_example(WEATHER, SUDOKU, 99, random.Random(0))
    assert row_to_messages(example) == [
        {"role": "user", "content": WEATHER["query"]},
        {"role": "assistant", "content": example["reply"]},
    ]


def test_make_no_tool_examples_uses_each_query_once_with_new_ids() -> None:
    rows = [WEATHER, SUDOKU, MOVIES, PIZZA]
    examples = make_no_tool_examples(rows, 4, random.Random(0))
    assert sorted(row["query"] for row in examples) == sorted(row["query"] for row in rows)
    assert sorted(row["id"] for row in examples) == [4, 5, 6, 7]
    original_tools = {row["query"]: row["tools"] for row in rows}
    assert all(row["tools"] != original_tools[row["query"]] for row in examples)


def test_make_no_tool_examples_fails_clearly_when_it_cannot_reach_count() -> None:
    twin = topic_row(1, "Is it raining in Rome?", "get_weather", "Forecast for a city.")
    with pytest.raises(ValueError, match="Only made 0 of 1"):
        make_no_tool_examples([WEATHER, twin], 1, random.Random(0))


@pytest.mark.parametrize(
    ("positives", "expected"), [(9, 1), (90, 10), (58485, 6498), (0, 0)]
)
def test_no_tool_count_makes_ten_percent_of_final(positives: int, expected: int) -> None:
    assert no_tool_count(positives) == expected


def _many_rows() -> list[dict[str, Any]]:
    topics = ["sudoku", "weather", "movie", "pizza", "chess", "tennis", "guitar", "volcano"]
    letters = "abcdefghij"
    return [
        topic_row(
            index * 10 + offset,
            f"Tell me about {topic}{letter}.",
            f"{topic}{letter}",
            f"About {topic}{letter}.",
        )
        for index, topic in enumerate(topics)
        for offset, letter in enumerate(letters)
    ]


def test_build_final_dataset_is_ten_percent_no_tool_and_drops_invalid_rows() -> None:
    bad = {**WEATHER, "id": 999, "answers": "not json"}
    final = build_final_dataset([*_many_rows(), bad])
    no_tool = [row for row in final if row["reply"]]
    assert len(final) == 89
    assert len(no_tool) == 9
    assert all(row["id"] != 999 for row in final)
    assert len({row["id"] for row in final}) == len(final)
    assert all(set(row) == {"id", "query", "answers", "tools", "reply"} for row in final)


def test_build_final_dataset_is_repeatable() -> None:
    assert build_final_dataset(_many_rows()) == build_final_dataset(_many_rows())
    assert build_final_dataset(_many_rows()) != build_final_dataset(_many_rows(), seed=1)


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


def test_real_template_renders_no_tool_reply_without_a_tool_call(
    qwen_tokenizer: PreTrainedTokenizerBase,
) -> None:
    example = make_no_tool_example(WEATHER, SUDOKU, 99, random.Random(0))
    text = render_example(example, qwen_tokenizer)
    assert "solve_sudoku" in text
    assert "get_weather" not in text
    assert text.rstrip().endswith(
        f"<|im_start|>assistant\n{example['reply']}<|im_end|>"
    )
    assert "<|im_start|>assistant\n<tool_call>" not in text

"""Tests for token counting and choosing max_seq_length."""

import json
from typing import Any

import pytest
from transformers import AutoTokenizer

from tooltuned import data
from tooltuned.data import BASE_MODEL, load_train_split
from tooltuned.formatting import (
    percentile,
    recommend_max_seq_length,
    share_within,
    token_length,
)

ROW = {
    "id": 0,
    "query": "Is it raining?",
    "answers": json.dumps([{"name": "rain", "arguments": {}}]),
    "tools": json.dumps([{"name": "rain", "description": "d", "parameters": {}}]),
    "reply": "",
}


class FakeTokenizer:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        return "one two three"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[str]]:
        self.kwargs = kwargs
        return {"input_ids": text.split()}


def test_token_length_counts_rendered_tokens_without_adding_special_ones() -> None:
    tokenizer = FakeTokenizer()
    assert token_length(ROW, tokenizer) == 3
    assert tokenizer.kwargs == {"add_special_tokens": False}


@pytest.mark.parametrize(
    ("percent", "expected"),
    [(1, 1), (10, 1), (11, 2), (50, 5), (90, 9), (95, 10), (100, 10)],
)
def test_percentile_uses_nearest_rank(percent: float, expected: int) -> None:
    assert percentile(list(range(1, 11)), percent) == expected


def test_percentile_of_a_single_value() -> None:
    assert percentile([7], 50) == 7


def test_share_within_includes_the_limit() -> None:
    assert share_within([1, 2, 3, 4], 2) == 0.5
    assert share_within([1, 2, 3, 4], 4) == 1.0


@pytest.mark.parametrize("function", [lambda: percentile([], 50), lambda: share_within([], 1)])
def test_empty_lengths_fail_clearly(function: Any) -> None:
    with pytest.raises(ValueError, match="no lengths"):
        function()


def test_recommendation_keeps_exactly_the_requested_share_before_rounding() -> None:
    lengths = list(range(1, 101))
    assert recommend_max_seq_length(lengths, round_to=1) == 98
    assert share_within(lengths, 98) == 0.98


def test_recommendation_ignores_the_longest_two_percent() -> None:
    assert recommend_max_seq_length([100] * 98 + [5000, 9000], round_to=1) == 100


def test_recommendation_needs_the_next_length_above_the_boundary() -> None:
    assert recommend_max_seq_length([100] * 97 + [200, 300, 400], round_to=1) == 200


@pytest.mark.parametrize(("needed", "expected"), [(97, 104), (96, 96), (1, 8)])
def test_recommendation_rounds_up(needed: int, expected: int) -> None:
    assert recommend_max_seq_length([needed]) == expected


def test_recommendation_does_not_depend_on_input_order() -> None:
    lengths = list(range(1, 101))
    assert recommend_max_seq_length(lengths[::-1]) == recommend_max_seq_length(lengths)


def test_load_train_split_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        load_train_split()


def test_load_train_split_reads_config_from_the_accounts_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class FakeApi:
        def __init__(self, token: str) -> None:
            pass

        def whoami(self) -> dict[str, str]:
            return {"name": "someone"}

    def fake_load(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "SPLIT"

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(data, "HfApi", FakeApi)
    monkeypatch.setattr(data, "load_dataset", fake_load)
    assert load_train_split("unverified") == "SPLIT"
    assert calls == [
        (
            ("someone/tooltuned-fc-data", "unverified"),
            {"split": "train", "token": "test-token"},
        )
    ]


@pytest.mark.internet
def test_token_length_with_real_tokenizer() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    except OSError as error:
        pytest.skip(f"Qwen2.5 tokenizer unavailable: {error}")
    longer = {**ROW, "query": ROW["query"] + " And is it cold, windy and foggy in Oslo?"}
    assert token_length(longer, tokenizer) > token_length(ROW, tokenizer) > 0

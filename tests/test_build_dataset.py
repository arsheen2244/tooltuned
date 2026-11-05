"""Tests for the BFCL duplicate filter, sampling, Hub push and report updates."""

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tooltuned import data, evaluate
from tooltuned.data import (
    bfcl_match,
    build_question_index,
    char_grams,
    normalize_text,
    push_splits_to_hub,
    remove_bfcl_duplicates,
    sample_splits,
    to_hub_row,
)
from tooltuned.evaluate import CATEGORY_COLUMNS, download_bfcl_wheel, load_bfcl_questions
from tooltuned.verify import update_report, verify_row

QUESTION = "Find the area of a triangle with a base of 10 units and height of 5 units."
INDEX = build_question_index([QUESTION, "Who wrote the novel Moby Dick?"])

TOOL = {
    "name": "f",
    "description": "d",
    "parameters": {"x": {"description": "d", "type": "int"}},
}


def tool_row(row_id: int, query: str | None = None, value: Any = 1) -> dict[str, Any]:
    return {
        "id": row_id,
        "query": query or f"query number {row_id}",
        "answers": json.dumps([{"name": "f", "arguments": {"x": value}}]),
        "tools": json.dumps([TOOL]),
        "reply": "",
    }


def no_tool_row(row_id: int, query: str | None = None) -> dict[str, Any]:
    return {
        "id": row_id,
        "query": query or f"no tool query {row_id}",
        "answers": "[]",
        "tools": json.dumps([TOOL]),
        "reply": "None fit.",
    }


def test_normalize_text() -> None:
    assert normalize_text("  What's\nthe AREA,  of (5)? ") == "what s the area of 5"


def test_char_grams() -> None:
    assert char_grams("abcdefghi") == {"abcdefgh", "bcdefghi"}
    assert char_grams("short") == frozenset()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (QUESTION, "exact"),
        ("FIND the area, of a triangle with a base of 10 units and height of 5 units", "exact"),
        ("Find the area of a triangle with a base of 12 units and height of 7 units.", "near"),
        (f"First, tell me a joke. Then: {QUESTION} Also, what time is it now?", "near"),
        ("Who wrote Moby Dick?", ""),
        ("What is the weather like in Paris during the spring?", ""),
        ("Hi", ""),
    ],
)
def test_bfcl_match(query: str, expected: str) -> None:
    assert bfcl_match(query, INDEX) == expected


def test_remove_bfcl_duplicates_counts_by_match_type() -> None:
    near = "Find the area of a triangle with a base of 12 units and height of 7 units."
    keep = tool_row(3, "What is the weather like in Paris during the spring?")
    kept, removed = remove_bfcl_duplicates(
        [tool_row(1, QUESTION), tool_row(2, near), keep], INDEX
    )
    assert kept == [keep]
    assert removed == {"exact": 1, "near": 1}


def _pools() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Every 10th tool-call row fails the verifier, because x should be an int.
    positives = [
        tool_row(index, value="wrong" if index % 10 == 0 else 1) for index in range(300)
    ]
    no_tool = [no_tool_row(1000 + index) for index in range(60)]
    return positives, no_tool


def _sample(**kwargs: Any) -> dict[str, list[dict[str, Any]]]:
    positives, no_tool = _pools()
    return sample_splits(positives, no_tool, train_size=100, validation_size=20, **kwargs)


def test_sample_splits_has_requested_sizes_and_no_tool_share() -> None:
    splits = _sample()
    assert len(splits["validation"]) == 20
    assert sum(1 for row in splits["validation"] if row["reply"]) == 2
    for name in ("train_verified", "train_unverified"):
        assert len(splits[name]) == 100
        assert sum(1 for row in splits[name] if row["reply"]) == 10


def test_verified_train_and_validation_only_hold_rows_that_pass() -> None:
    splits = _sample()
    for name in ("train_verified", "validation"):
        assert all(verify_row(row)[0] for row in splits[name] if not row["reply"])


def test_unverified_train_contains_rejected_rows() -> None:
    rows = _sample()["train_unverified"]
    assert any(not verify_row(row)[0] for row in rows if not row["reply"])


def test_train_sets_only_differ_by_what_the_verifier_rejected() -> None:
    splits = _sample()
    verified_ids = {row["id"] for row in splits["train_verified"]}
    only_unverified = [r for r in splits["train_unverified"] if r["id"] not in verified_ids]
    assert only_unverified
    assert all(not verify_row(row)[0] for row in only_unverified)


def test_train_sets_share_no_tool_rows() -> None:
    splits = _sample()
    ids = [{r["id"] for r in splits[name] if r["reply"]} for name in splits if "train" in name]
    assert ids[0] == ids[1]


def test_no_query_appears_in_both_train_and_validation() -> None:
    positives, no_tool = _pools()
    # The same query as a tool-call row and a no-tool row, with different punctuation.
    no_tool = [no_tool_row(1000 + index, f"Query, number {index}!") for index in range(60)]
    splits = sample_splits(positives, no_tool, train_size=100, validation_size=20)
    validation = {normalize_text(row["query"]) for row in splits["validation"]}
    for name in ("train_verified", "train_unverified"):
        assert not validation & {normalize_text(row["query"]) for row in splits[name]}


def test_sample_splits_is_repeatable_whatever_the_input_order() -> None:
    positives, no_tool = _pools()
    first = sample_splits(positives, no_tool, train_size=100, validation_size=20)
    second = sample_splits(positives[::-1], no_tool[::-1], train_size=100, validation_size=20)
    assert first == second
    assert first != _sample(seed=1)


def test_sample_splits_fails_clearly_when_a_pool_is_too_small() -> None:
    positives, no_tool = _pools()
    with pytest.raises(ValueError, match="no-tool"):
        sample_splits(positives, no_tool[:5], train_size=100, validation_size=20)


def test_to_hub_row_adds_messages_as_json_text() -> None:
    row = to_hub_row(no_tool_row(1))
    assert json.loads(row["messages"])[1] == {"role": "assistant", "content": "None fit."}


class FakeDatasetDict:
    pushes: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, **splits: Any) -> None:
        self.splits = splits

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> None:
        self.pushes.append((repo_id, kwargs))


class FakeApi:
    is_private = True
    created: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, token: str) -> None:
        self.token = token

    def whoami(self) -> dict[str, str]:
        return {"name": "someone"}

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.created.append((repo_id, kwargs))

    def repo_info(self, repo_id: str, repo_type: str) -> Any:
        return type("Info", (), {"private": self.is_private})


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeDatasetDict.pushes = []
    FakeApi.created = []
    FakeApi.is_private = True
    monkeypatch.setattr(data, "HfApi", FakeApi)
    monkeypatch.setattr(data, "DatasetDict", FakeDatasetDict)
    monkeypatch.setattr(data.Dataset, "from_list", staticmethod(lambda rows: rows))


SPLITS = {
    "validation": [no_tool_row(1)],
    "train_verified": [no_tool_row(2)],
    "train_unverified": [no_tool_row(3)],
}


def test_push_creates_private_repo_and_both_configs(fake_hub: None) -> None:
    assert push_splits_to_hub(SPLITS, "token") == "someone/tooltuned-fc-data"
    assert FakeApi.created[0][1]["private"] is True
    assert [push[1]["config_name"] for push in FakeDatasetDict.pushes] == [
        "verified",
        "unverified",
    ]
    assert all(push[1]["private"] is True for push in FakeDatasetDict.pushes)


def test_push_refuses_an_existing_public_repo(fake_hub: None) -> None:
    FakeApi.is_private = False
    with pytest.raises(RuntimeError, match="public"):
        push_splits_to_hub(SPLITS, "token")
    assert FakeDatasetDict.pushes == []


def test_update_report_keeps_other_sections(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text('{"accepted": 5, "final_dataset": {"old": 1}}', encoding="utf-8")
    update_report(path, {"final_dataset": {"new": 2}})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "accepted": 5,
        "final_dataset": {"new": 2},
    }


def test_update_report_creates_a_new_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    update_report(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def _make_wheel(path: Path) -> None:
    entry = {
        "id": "x",
        "question": [[{"role": "system", "content": "skip"}, {"role": "user", "content": "Q1"}]],
    }
    with zipfile.ZipFile(path, "w") as wheel:
        for category in CATEGORY_COLUMNS:
            wheel.writestr(f"bfcl_eval/data/BFCL_v4_{category}.json", json.dumps(entry) + "\n")
        wheel.writestr("bfcl_eval/data/BFCL_v4_live_simple.json", "not used\n")


def test_load_bfcl_questions_reads_user_messages_of_our_categories(tmp_path: Path) -> None:
    path = tmp_path / "bfcl.whl"
    _make_wheel(path)
    assert load_bfcl_questions(path) == ["Q1"] * len(CATEGORY_COLUMNS)


def test_download_bfcl_wheel_removes_a_file_with_the_wrong_checksum(tmp_path: Path) -> None:
    path = tmp_path / "bfcl.whl"
    path.write_bytes(b"not the real wheel")
    with pytest.raises(ValueError, match="checksum"):
        download_bfcl_wheel(path)
    assert not path.exists()


def test_download_bfcl_wheel_keeps_a_matching_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bfcl.whl"
    path.write_bytes(b"content")
    digest = "ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73"
    monkeypatch.setattr(evaluate, "BFCL_WHEEL_SHA256", digest)
    assert download_bfcl_wheel(path) == path

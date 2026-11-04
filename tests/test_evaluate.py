"""Tests for reading BFCL scores and building the results table."""

from pathlib import Path

import pytest

from tooltuned.evaluate import (
    SCORE_FILE_NAME,
    format_accuracy,
    read_score_row,
    scores_to_markdown,
)

HEADER = (
    "Rank,Model,Python Simple AST,Multiple AST,Parallel AST,"
    "Parallel Multiple AST,Irrelevance Detection\n"
)
ROW = "1,Qwen,0.85,0.5,0.25,N/A,90.00%\n"


def write_scores(folder: Path, *rows: str) -> None:
    (folder / SCORE_FILE_NAME).write_text(HEADER + "".join(rows), encoding="utf-8")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.85", "85.00%"), ("0", "0.00%"), ("90.00%", "90.00%"), ("N/A", "N/A")],
)
def test_format_accuracy(raw: str, expected: str) -> None:
    assert format_accuracy(raw) == expected


def test_table_has_every_category_with_its_own_score(tmp_path: Path) -> None:
    write_scores(tmp_path, ROW)
    assert scores_to_markdown(read_score_row(tmp_path)).splitlines() == [
        "| Category | Accuracy |",
        "| --- | --- |",
        "| `simple_python` | 85.00% |",
        "| `multiple` | 50.00% |",
        "| `parallel` | 25.00% |",
        "| `parallel_multiple` | N/A |",
        "| `irrelevance` | 90.00% |",
    ]


@pytest.mark.parametrize("rows", [(), (ROW, ROW)])
def test_read_score_row_requires_exactly_one_model(
    tmp_path: Path, rows: tuple[str, ...]
) -> None:
    write_scores(tmp_path, *rows)
    with pytest.raises(ValueError, match="Expected 1 model"):
        read_score_row(tmp_path)

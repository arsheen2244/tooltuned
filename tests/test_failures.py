"""Tests for reading BFCL score files, sorting failures into types, and comparing models."""

import json
from pathlib import Path
from typing import Any

import pytest

from tooltuned import evaluate as ev

CALL = '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'
PROMPT = {"question": [[{"role": "user", "content": "What is f of 1?"}]], "function": []}


def entry(category: str = "simple_python", raw: str = CALL, **fields: Any) -> dict:
    return {"id": f"{category}_0", "test_category": category, "prompt": PROMPT,
            "model_result_raw": raw, "model_result_decoded": [{"f": {"x": 1}}], **fields}


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ({"id": "irrelevance_0", "test_category": "irrelevance", "model_result": CALL,
          "error_type": "irrelevance_error:decoder_success"}, ev.CALLED_WHEN_IT_SHOULD_NOT),
        (entry(raw="The answer is 42.", error_type="ast_decoder:decoder_wrong_output_format"),
         ev.DID_NOT_CALL),
        (entry(error_type="ast_decoder:decoder_failed"), ev.BAD_FORMAT),
        (entry(raw="<tool_call>\n{bad json}\n</tool_call>", model_result_decoded=[],
               error_type="simple_function_checker:wrong_count"), ev.BAD_FORMAT),
        (entry(error_type="simple_function_checker:wrong_func_name"), ev.WRONG_FUNCTION),
        (entry(category="parallel", error_type="parallel_function_checker_no_order:cannot_find_match",
               error=["No match.", {"Model Result Index 0": {"sub_error_type": "simple_function_checker:wrong_func_name"}}]),
         ev.WRONG_FUNCTION),
        (entry(category="parallel", error_type="parallel_function_checker_no_order:cannot_find_match",
               error=["No match.", {"Model Result Index 0": {"sub_error_type": "simple_function_checker:wrong_func_name"}},
                      {"Model Result Index 1": {"sub_error_type": "value_error:others"}}]),
         ev.WRONG_ARGUMENTS),
        (entry(category="parallel", error_type="parallel_function_checker_no_order:wrong_count"),
         ev.WRONG_COUNT),
        (entry(error_type="value_error:string"), ev.WRONG_ARGUMENTS),
        (entry(error_type="type_error:simple"), ev.WRONG_ARGUMENTS),
        (entry(error_type="simple_function_checker:missing_required"), ev.WRONG_ARGUMENTS),
        (entry(category="parallel", error_type="parallel_function_checker_no_order:cannot_find_match",
               error=["No match.", {"Model Result Index 0": {"sub_error_type": "value_error:others"}}]), ev.WRONG_ARGUMENTS),
        (entry(error_type="dict_checker:unclear"), ev.OTHER),
    ],
)
def test_classify_failure(failure: dict, expected: str) -> None:
    assert ev.classify_failure(failure) == expected


def write_score_files(folder: Path, failures_by_category: dict[str, list[dict]]) -> None:
    (folder / "score").mkdir(parents=True)
    for category in ev.CATEGORY_COLUMNS:
        items = failures_by_category.get(category, [])
        lines = [{"accuracy": 1.0, "correct_count": 10, "total_count": 10 + len(items)}, *items]
        path = ev.score_file_path(folder, category)
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def test_load_failures_skips_each_files_summary_line(tmp_path: Path) -> None:
    write_score_files(tmp_path, {"simple_python": [entry()], "parallel": [entry("parallel")] * 2})
    failures = ev.load_failures(tmp_path)
    assert len(failures) == 3
    assert all("accuracy" not in item for item in failures)


def test_count_failure_types_lists_every_type() -> None:
    counts = ev.count_failure_types([entry(error_type="value_error:string")] * 2)
    assert list(counts) == list(ev.FAILURE_TYPES)
    assert counts[ev.WRONG_ARGUMENTS] == 2 and sum(counts.values()) == 2


@pytest.mark.parametrize(
    ("only_a", "only_b", "expected"),
    [(10, 2, 158 / 4096), (5, 5, 1.0), (0, 0, 1.0), (6, 0, 2 / 64)],
)
def test_mcnemar_p_value_matches_the_binomial_formula(only_a: int, only_b: int, expected: float) -> None:
    assert ev.mcnemar_p_value(only_a, only_b) == pytest.approx(expected)


def test_paired_comparison_counts_questions_only_one_model_got_right() -> None:
    a = [{"id": "q1"}, {"id": "q2"}]
    b = [{"id": "q2"}, {"id": "q3"}, {"id": "q4"}]
    only_a, only_b, _ = ev.paired_comparison(a, b)
    assert (only_a, only_b) == (2, 1)


def test_question_and_expected_answer_are_readable() -> None:
    failure = entry(possible_answer={"id": "x", "ground_truth": [{"f": {"x": [1]}}]})
    assert ev.question_text(failure) == "What is f of 1?"
    assert ev.expected_answer(failure) == '[{"f": {"x": [1]}}]'
    assert "No tool call" in ev.expected_answer({"test_category": "irrelevance"})


def test_comparison_table_shows_changes_against_the_first_model() -> None:
    row = {column: "50.00%" for column in ev.CATEGORY_COLUMNS.values()}
    better = {**row, "Irrelevance Detection": "0.575"}
    table = ev.comparison_table({"Base": row, "Tuned": better})
    assert "| Category | Base | Tuned | Tuned vs Base |" in table
    assert "| `irrelevance` | 50.00% | 57.50% | +7.5 |" in table


def test_failure_table_has_a_row_per_type_and_a_total() -> None:
    counts = {"A": dict.fromkeys(ev.FAILURE_TYPES, 1)}
    table = ev.failure_table(counts)
    assert table.count("\n") == len(ev.FAILURE_TYPES) + 2
    assert f"| **total** | {len(ev.FAILURE_TYPES)} |" in table


def test_paired_table_keeps_tool_calling_and_irrelevance_apart() -> None:
    failures = {
        "A": [{"id": "simple_python_1", "test_category": "simple_python"}],
        "B": [{"id": "irrelevance_1", "test_category": "irrelevance"}],
    }
    table = ev.paired_table(failures, [("A", "B")])
    assert "| tool calling | A vs B | 0 | 1 | 1.000 | could be chance |" in table
    assert "| irrelevance | A vs B | 1 | 0 | 1.000 | could be chance |" in table


@pytest.mark.parametrize(("p_value", "expected"), [(0.01, "real difference"), (0.5, "could be chance")])
def test_verdict(p_value: float, expected: str) -> None:
    assert ev.verdict(p_value) == expected

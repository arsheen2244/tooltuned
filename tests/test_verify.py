"""Tests for rejecting invalid dataset rows."""

import json
from typing import Any

import pytest

from tooltuned import verify
from tooltuned.verify import (
    INVALID_ANSWERS_JSON,
    INVALID_TOOLS,
    MALFORMED_ANSWERS,
    MISSING_REQUIRED_PARAMETER,
    REASON_CODES,
    TYPE_MISMATCH,
    UNKNOWN_FUNCTION,
    UNKNOWN_PARAMETER,
    UNSUPPORTED_TYPE,
    build_report,
    is_required,
    parse_declared_type,
    reason_code,
    value_matches,
    verify_row,
)

TOOL = {
    "name": "f",
    "description": "A test function.",
    "parameters": {
        "needed": {"description": "Required.", "type": "int"},
        "extra": {"description": "Optional.", "type": "str, optional"},
    },
}


def make_row(answers: Any, tools: Any = (TOOL,)) -> dict[str, Any]:
    return {
        "id": 0,
        "query": "q",
        "answers": json.dumps(answers),
        "tools": json.dumps(list(tools)),
    }


def call(**arguments: Any) -> dict[str, Any]:
    return {"name": "f", "arguments": arguments}


def tool_with_type(declared: str) -> dict[str, Any]:
    return {"name": "f", "parameters": {"p": {"description": "", "type": declared}}}


def test_valid_row_returns_ok_and_empty_reason() -> None:
    assert verify_row(make_row([call(needed=1, extra="x")])) == (True, "")


def test_valid_row_with_parallel_calls() -> None:
    assert verify_row(make_row([call(needed=1), call(needed=2)])) == (True, "")


@pytest.mark.parametrize("answers", ["not json", '[{"name": "f"', None])
def test_rejects_answers_that_are_not_json(answers: Any) -> None:
    row = {**make_row([call(needed=1)]), "answers": answers}
    ok, reason = verify_row(row)
    assert not ok
    assert reason_code(reason) == INVALID_ANSWERS_JSON


@pytest.mark.parametrize(
    "answers",
    [
        [],
        {"name": "f", "arguments": {}},
        ["f"],
        [{"name": "f"}],
        [{"arguments": {}}],
        [{"name": "f", "arguments": "{}"}],
    ],
)
def test_rejects_malformed_answers(answers: Any) -> None:
    ok, reason = verify_row(make_row(answers))
    assert not ok
    assert reason_code(reason) == MALFORMED_ANSWERS


@pytest.mark.parametrize(
    "tools",
    [
        "not json",
        None,
        json.dumps({"name": "f"}),
        json.dumps([{"name": "f"}]),
        json.dumps([{"name": "f", "parameters": {"p": "int"}}]),
        json.dumps([{"name": "f", "parameters": {"p": {"description": "no type"}}}]),
    ],
)
def test_rejects_invalid_tools(tools: Any) -> None:
    row = {**make_row([call(needed=1)]), "tools": tools}
    ok, reason = verify_row(row)
    assert not ok
    assert reason_code(reason) == INVALID_TOOLS


def test_rejects_function_not_in_tools() -> None:
    ok, reason = verify_row(make_row([{"name": "g", "arguments": {}}]))
    assert not ok
    assert reason_code(reason) == UNKNOWN_FUNCTION


def test_rejects_when_a_later_call_is_invalid() -> None:
    ok, reason = verify_row(make_row([call(needed=1), {"name": "g", "arguments": {}}]))
    assert not ok
    assert reason_code(reason) == UNKNOWN_FUNCTION


def test_rejects_missing_required_parameter() -> None:
    ok, reason = verify_row(make_row([call(extra="x")]))
    assert not ok
    assert reason_code(reason) == MISSING_REQUIRED_PARAMETER


def test_rejects_undefined_parameter() -> None:
    ok, reason = verify_row(make_row([call(needed=1, invented=2)]))
    assert not ok
    assert reason_code(reason) == UNKNOWN_PARAMETER


def test_rejects_value_of_wrong_type() -> None:
    ok, reason = verify_row(make_row([call(needed="1")]))
    assert not ok
    assert reason_code(reason) == TYPE_MISMATCH


def test_rejects_type_it_cannot_check() -> None:
    row = make_row([call(p=1)], [tool_with_type("Banana[int]")])
    ok, reason = verify_row(row)
    assert not ok
    assert reason_code(reason) == UNSUPPORTED_TYPE


def test_rejects_unrecognised_type_modifier() -> None:
    row = make_row([call(p=1)], [tool_with_type("int, sometimes")])
    ok, reason = verify_row(row)
    assert not ok
    assert reason_code(reason) == UNSUPPORTED_TYPE


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("str", ("str", False, False)),
        ("str, optional", ("str", True, False)),
        ("int, optional, default=100", ("int", True, True)),
        ("str, default='elonmusk'", ("str", False, True)),
        ("str, optional, default 'London'", ("str", True, True)),
        ("List[Union[int, float]]", ("List[Union[int, float]]", False, False)),
        ("Tuple[float, float], optional", ("Tuple[float, float]", True, False)),
    ],
)
def test_parse_declared_type(declared: str, expected: tuple[str, bool, bool]) -> None:
    assert parse_declared_type(declared) == expected


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [
        ({"type": "int"}, True),
        ({"type": "int, optional"}, False),
        ({"type": "int", "default": 5}, False),
        ({"type": "int", "default": ""}, False),
        ({"type": "str, default='x'"}, False),
    ],
)
def test_is_required(parameter: dict[str, Any], expected: bool) -> None:
    assert is_required(parameter) is expected


@pytest.mark.parametrize(
    ("base", "value"),
    [
        ("str", "x"),
        ("int", 1),
        ("float", 1.5),
        ("float", 1),
        ("bool", True),
        ("list", [1, "a"]),
        ("List", []),
        ("List[int]", [1, 2]),
        ("List[List[str]]", [["a"], []]),
        ("List[Union[int, float]]", [1, 2.5]),
        ("Tuple[float, float]", [1.0, 2]),
        ("List[Tuple[int, int]]", [[1, 2], [3, 4]]),
        ("set", [1, 2]),
        ("Dict", {"a": 1}),
        ("Callable[[float], float]", "lambda x: x"),
    ],
)
def test_value_matches_accepts(base: str, value: Any) -> None:
    assert value_matches(base, value)


@pytest.mark.parametrize(
    ("base", "value"),
    [
        ("str", 1),
        ("str", None),
        ("int", "1"),
        ("int", 1.0),
        ("int", 0.7),
        ("int", True),
        ("float", "1.5"),
        ("float", True),
        ("bool", 1),
        ("bool", "true"),
        ("list", "[1]"),
        ("List[int]", [1, "2"]),
        ("List[int]", "[1, 2]"),
        ("List[List[str]]", ["a"]),
        ("Tuple[float, float]", [1.0]),
        ("Tuple[float, float]", [1.0, 2.0, 3.0]),
        ("Tuple[float, float]", [1.0, "2"]),
        ("set", "a"),
        ("Dict", []),
        ("Callable[[float], float]", 1),
    ],
)
def test_value_matches_rejects(base: str, value: Any) -> None:
    assert not value_matches(base, value)


@pytest.mark.parametrize(
    ("declared", "ok"),
    [("str, optional", True), ("str", False), ("str, default='x'", False)],
)
def test_null_is_only_accepted_for_optional_parameters(declared: str, ok: bool) -> None:
    assert verify_row(make_row([call(p=None)], [tool_with_type(declared)]))[0] is ok


def test_build_report_counts_each_reason_and_keeps_examples() -> None:
    good = make_row([call(needed=1)])
    bad = make_row([call(needed="1")])
    report = build_report([good, bad, make_row([call()])])

    assert report["total_rows"] == 3
    assert report["accepted"] == 1
    assert report["rejected"] == 2
    assert set(report["rejected_by_reason"]) == set(REASON_CODES)
    assert report["rejected_by_reason"][TYPE_MISMATCH] == 1
    assert report["rejected_by_reason"][MISSING_REQUIRED_PARAMETER] == 1
    assert report["rejected_by_reason"][UNKNOWN_FUNCTION] == 0

    example = report["examples_by_reason"][TYPE_MISMATCH][0]
    assert example["answers"] == bad["answers"]
    assert reason_code(example["reason"]) == TYPE_MISMATCH
    json.dumps(report)


def test_build_report_limits_examples_but_not_counts() -> None:
    bad = make_row([call(needed="1")])
    report = build_report([bad] * (verify.EXAMPLES_PER_REASON + 2))
    assert report["rejected_by_reason"][TYPE_MISMATCH] == verify.EXAMPLES_PER_REASON + 2
    assert len(report["examples_by_reason"][TYPE_MISMATCH]) == verify.EXAMPLES_PER_REASON

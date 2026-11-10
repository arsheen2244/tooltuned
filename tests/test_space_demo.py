"""Tests for the demo Space's logic in space/demo.py (no Gradio or model needed)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "space" / "demo.py"


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("space_demo", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_tools_accepts_a_list_or_a_single_tool(demo: ModuleType) -> None:
    tool = {"name": "f", "parameters": {}}
    assert demo.parse_tools(json.dumps([tool])) == [tool]
    assert demo.parse_tools(json.dumps(tool)) == [tool]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[{", "valid JSON"),
        ("[]", "at least one tool"),
        ('"f"', "at least one tool"),
        ('[{"description": "no name"}]', '"name"'),
        ("[1]", '"name"'),
    ],
)
def test_parse_tools_rejects_bad_input(demo: ModuleType, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        demo.parse_tools(text)


def test_parse_tool_calls_reads_every_block(demo: ModuleType) -> None:
    answer = (
        '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
    )
    assert demo.parse_tool_calls(answer) == [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {}},
    ]


def test_parse_tool_calls_keeps_broken_json_as_text(demo: ModuleType) -> None:
    assert demo.parse_tool_calls('<tool_call>\n{"name": "a"\n</tool_call>') == ['{"name": "a"']


def test_parse_tool_calls_handles_a_cut_off_block(demo: ModuleType) -> None:
    assert demo.parse_tool_calls('<tool_call>\n{"name": "a", "arguments": {}}') == [
        {"name": "a", "arguments": {}}
    ]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Sorry, none of these tools can help.", "No tool call"),
        ('<tool_call>{"name": "a"}</tool_call>', "1 tool call."),
        ('<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>', "2 tool calls."),
        ('<tool_call>{"name": "a"}</tool_call><tool_call>{"name"</tool_call>', "1 of them not valid"),
    ],
)
def test_summarize(demo: ModuleType, answer: str, expected: str) -> None:
    assert expected in demo.summarize(answer)


def test_examples_have_valid_tools(demo: ModuleType) -> None:
    for _, tools, query in demo.EXAMPLES:
        assert demo.parse_tools(json.dumps(tools)) == tools
        assert query.strip()

"""Check the prepared data is valid.

xLAM describes parameters with Python-style type strings rather than JSON Schema:

    {"city": {"description": "...", "type": "str, optional", "default": "Paris"}}

- Type strings look like "int", "List[Tuple[float, float]]" or
  "str, optional, default='x'".
- There is no "required" list. A parameter is treated as required when its type isn't
  marked optional and it has no default, either as a key or inside the type string.
- JSON has no tuples, sets or functions, so those arrive as lists and strings.
- float accepts int, as Python's typing does. int doesn't accept float, and bool counts
  as neither. null is only accepted for parameters marked optional.
"""

import json
import re
from collections.abc import Iterable
from typing import Any

INVALID_ANSWERS_JSON = "invalid_answers_json"
MALFORMED_ANSWERS = "malformed_answers"
INVALID_TOOLS = "invalid_tools"
UNKNOWN_FUNCTION = "unknown_function"
UNKNOWN_PARAMETER = "unknown_parameter"
MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
TYPE_MISMATCH = "type_mismatch"
UNSUPPORTED_TYPE = "unsupported_type"

REASON_CODES = (
    INVALID_ANSWERS_JSON,
    MALFORMED_ANSWERS,
    INVALID_TOOLS,
    UNKNOWN_FUNCTION,
    UNKNOWN_PARAMETER,
    MISSING_REQUIRED_PARAMETER,
    TYPE_MISMATCH,
    UNSUPPORTED_TYPE,
)

EXAMPLES_PER_REASON = 3

Row = dict[str, Any]


class UnsupportedTypeError(ValueError):
    """A declared type isn't one this module knows how to check."""


def split_top_level(text: str) -> list[str]:
    """Split on commas that aren't inside square brackets."""
    parts, current, depth = [], "", 0
    for character in text:
        depth += {"[": 1, "]": -1}.get(character, 0)
        if character == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += character
    parts.append(current.strip())
    return parts


def parse_declared_type(declared: str) -> tuple[str, bool, bool]:
    """Split a type string into its base type, whether it's optional and has a default."""
    base, *modifiers = split_top_level(declared)
    for modifier in modifiers:
        if modifier != "optional" and not modifier.startswith("default"):
            raise UnsupportedTypeError(declared)
    has_default = any(modifier.startswith("default") for modifier in modifiers)
    return base, "optional" in modifiers, has_default


def is_required(parameter: dict[str, Any]) -> bool:
    """A parameter is required unless it's marked optional or has a default."""
    _, optional, inline_default = parse_declared_type(parameter["type"])
    return not (optional or inline_default or "default" in parameter)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or isinstance(value, float)


def _matches_generic(name: str, arguments: list[str], value: Any) -> bool:
    if name == "list":
        return isinstance(value, list) and (
            not arguments or all(value_matches(arguments[0], item) for item in value)
        )
    if name == "tuple":
        return (
            isinstance(value, list)
            and len(value) == len(arguments)
            and all(map(value_matches, arguments, value))
        )
    if name == "union":
        return any(value_matches(argument, value) for argument in arguments)
    raise UnsupportedTypeError(name)


def value_matches(base: str, value: Any) -> bool:
    """Check a JSON value against a Python-style type with no optional or default parts."""
    match = re.fullmatch(r"(\w+)(?:\[(.*)\])?", base.strip())
    if match is None:
        raise UnsupportedTypeError(base)
    name = match.group(1).lower()
    if name == "callable":
        return isinstance(value, str)  # JSON can only write a function as text.
    arguments = split_top_level(match.group(2)) if match.group(2) else []
    simple_checks = {
        "str": lambda: isinstance(value, str),
        "int": lambda: _is_int(value),
        "float": lambda: _is_number(value),
        "bool": lambda: isinstance(value, bool),
        "set": lambda: isinstance(value, list),
        "dict": lambda: isinstance(value, dict),
    }
    if name in simple_checks and not arguments:
        return simple_checks[name]()
    return _matches_generic(name, arguments, value)


def _argument_matches(parameter: dict[str, Any], value: Any) -> bool:
    base, optional, _ = parse_declared_type(parameter["type"])
    return (value is None and optional) or value_matches(base, value)


def _is_well_formed_call(answer: Any) -> bool:
    return (
        isinstance(answer, dict)
        and isinstance(answer.get("name"), str)
        and isinstance(answer.get("arguments"), dict)
    )


def _is_well_formed_tool(tool: Any) -> bool:
    return (
        isinstance(tool, dict)
        and isinstance(tool.get("name"), str)
        and isinstance(tool.get("parameters"), dict)
        and all(
            isinstance(parameter, dict) and isinstance(parameter.get("type"), str)
            for parameter in tool["parameters"].values()
        )
    )


def _check_call(answer: dict[str, Any], tools: dict[str, Any]) -> str:
    """Return a rejection reason for one function call, or "" if it's valid."""
    name, arguments = answer["name"], answer["arguments"]
    if name not in tools:
        return f"{UNKNOWN_FUNCTION}: {name} isn't in the tools list"
    parameters = tools[name]["parameters"]
    for argument in arguments:
        if argument not in parameters:
            return f"{UNKNOWN_PARAMETER}: {name} doesn't define {argument}"
    for parameter_name, parameter in parameters.items():
        if is_required(parameter) and parameter_name not in arguments:
            return f"{MISSING_REQUIRED_PARAMETER}: {name} needs {parameter_name}"
    for argument, value in arguments.items():
        if not _argument_matches(parameters[argument], value):
            declared = parameters[argument]["type"]
            return f"{TYPE_MISMATCH}: {name}.{argument} is {declared!r}, got {value!r}"
    return ""


def verify_row(row: Row) -> tuple[bool, str]:
    """Return (ok, reason). The reason is "" if ok, else "<code>: <details>".

    Only the first problem found in a row is reported.
    """
    try:
        answers = json.loads(row["answers"])
    except (TypeError, json.JSONDecodeError) as error:
        return False, f"{INVALID_ANSWERS_JSON}: {error}"
    if not (isinstance(answers, list) and answers and all(map(_is_well_formed_call, answers))):
        return False, f"{MALFORMED_ANSWERS}: expected a list of name and arguments objects"
    try:
        tools = json.loads(row["tools"])
    except (TypeError, json.JSONDecodeError) as error:
        return False, f"{INVALID_TOOLS}: {error}"
    if not (isinstance(tools, list) and all(map(_is_well_formed_tool, tools))):
        return False, f"{INVALID_TOOLS}: expected a list of name and parameters objects"

    tools_by_name = {tool["name"]: tool for tool in tools}
    try:
        for answer in answers:
            reason = _check_call(answer, tools_by_name)
            if reason:
                return False, reason
    except UnsupportedTypeError as error:
        return False, f"{UNSUPPORTED_TYPE}: {error}"
    return True, ""


def reason_code(reason: str) -> str:
    """Return the code at the start of a rejection reason."""
    return reason.split(":", 1)[0]


def build_report(rows: Iterable[Row]) -> dict[str, Any]:
    """Count rejections per reason and keep the first few rejected rows for each."""
    total = 0
    counts = dict.fromkeys(REASON_CODES, 0)
    examples: dict[str, list[Row]] = {code: [] for code in REASON_CODES}
    for row in rows:
        total += 1
        ok, reason = verify_row(row)
        if ok:
            continue
        code = reason_code(reason)
        counts[code] += 1
        if len(examples[code]) < EXAMPLES_PER_REASON:
            examples[code].append({**row, "reason": reason})
    rejected = sum(counts.values())
    return {
        "total_rows": total,
        "accepted": total - rejected,
        "rejected": rejected,
        "rejected_by_reason": counts,
        "note": "Each rejected row is counted once, under the first problem found.",
        "examples_by_reason": examples,
    }

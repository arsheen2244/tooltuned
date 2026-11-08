"""Score the base and fine-tuned models on BFCL."""

import csv
import hashlib
import json
import math
import urllib.request
import zipfile
from pathlib import Path

SCORE_FILE_NAME = "data_non_live.csv"

# BFCL's questions ship inside its package, which has to stay in its own environment.
# Reading them from the pinned wheel avoids installing it here.
BFCL_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/ba/41/"
    "ed458527c770c50225b60bae3b0c3444b26804ee455fa2d8f187018d2cb2/"
    "bfcl_eval-2026.3.23-py3-none-any.whl"
)
BFCL_WHEEL_SHA256 = "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"

# Maps each BFCL test category to its column in data_non_live.csv.
CATEGORY_COLUMNS: dict[str, str] = {
    "simple_python": "Python Simple AST",
    "multiple": "Multiple AST",
    "parallel": "Parallel AST",
    "parallel_multiple": "Parallel Multiple AST",
    "irrelevance": "Irrelevance Detection",
}


def download_bfcl_wheel(path: Path) -> Path:
    """Download the pinned BFCL wheel once and check it's the expected file."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BFCL_WHEEL_URL, timeout=120) as response:
            path.write_bytes(response.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != BFCL_WHEEL_SHA256:
        path.unlink()
        raise ValueError("The BFCL wheel doesn't match its expected checksum.")
    return path


def load_bfcl_questions(wheel_path: Path) -> list[str]:
    """Read the user questions of the categories we score from the BFCL wheel."""
    questions: list[str] = []
    with zipfile.ZipFile(wheel_path) as wheel:
        for category in CATEGORY_COLUMNS:
            text = wheel.read(f"bfcl_eval/data/BFCL_v4_{category}.json").decode("utf-8")
            for line in text.splitlines():
                for turn in json.loads(line)["question"]:
                    questions += [m["content"] for m in turn if m["role"] == "user"]
    return questions


def read_score_row(results_dir: Path) -> dict[str, str]:
    """Return the single model's row from a results folder's score CSV."""
    with (results_dir / SCORE_FILE_NAME).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"Expected 1 model in {SCORE_FILE_NAME}, found {len(rows)}.")
    return rows[0]


def format_accuracy(raw: str) -> str:
    """Show a BFCL accuracy value, stored as 0.85 or 85.00%, as a percentage."""
    value = raw.strip()
    if value == "N/A" or value.endswith("%"):
        return value
    return f"{float(value) * 100:.2f}%"


def scores_to_markdown(row: dict[str, str]) -> str:
    """Build a markdown table with the accuracy of each category."""
    lines = ["| Category | Accuracy |", "| --- | --- |"]
    for category, column in CATEGORY_COLUMNS.items():
        lines.append(f"| `{category}` | {format_accuracy(row[column])} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Failure analysis: reads BFCL's per-category score files, which run_bfcl.sh copies to
# results/<output>/score/. Their first line is a summary; every other line is one failed
# question with its error type, the prompt, the raw model output and the expected answer.

CALLED_WHEN_IT_SHOULD_NOT = "called a tool when it shouldn't"
DID_NOT_CALL = "didn't call a tool when it should"
BAD_FORMAT = "bad format"
WRONG_FUNCTION = "wrong function"
WRONG_COUNT = "wrong number of calls"
WRONG_ARGUMENTS = "wrong arguments"
OTHER = "other"
FAILURE_TYPES = (
    CALLED_WHEN_IT_SHOULD_NOT,
    DID_NOT_CALL,
    BAD_FORMAT,
    WRONG_FUNCTION,
    WRONG_COUNT,
    WRONG_ARGUMENTS,
    OTHER,
)
ARGUMENT_ERRORS = ("missing_required", "unexpected_param", "missing_optional")


def read_score_file(path: Path) -> tuple[dict, list[dict]]:
    """Return a score file's summary line and its failure entries."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = [json.loads(line) for line in lines]
    return entries[0], entries[1:]


def score_file_path(results_dir: Path, category: str) -> Path:
    return results_dir / "score" / f"BFCL_v4_{category}_score.json"


def load_failures(results_dir: Path) -> list[dict]:
    """All failure entries of the categories we score, from one results folder."""
    failures: list[dict] = []
    for category in CATEGORY_COLUMNS:
        failures += read_score_file(score_file_path(results_dir, category))[1]
    return failures


def raw_output(entry: dict) -> str:
    """The model's raw text. Irrelevance entries store it under a different key."""
    return str(entry.get("model_result_raw", entry.get("model_result", "")))


def _error_types(entry: dict) -> list[str]:
    """The entry's error type, plus the inner one BFCL keeps for unmatched parallel calls."""
    types = [entry.get("error_type", "")]
    error = entry.get("error")
    details = error.values() if isinstance(error, dict) else []
    types += [d.get("sub_error_type", "") for d in details if isinstance(d, dict)]
    return types


def classify_failure(entry: dict) -> str:
    """Sort one failed BFCL question into a failure type. The checks run in order."""
    if "irrelevance" in entry.get("test_category", ""):
        return CALLED_WHEN_IT_SHOULD_NOT
    if "<tool_call>" not in raw_output(entry):
        return DID_NOT_CALL
    types = _error_types(entry)
    if types[0].startswith("ast_decoder:") or not entry.get("model_result_decoded", True):
        return BAD_FORMAT
    if any(t.endswith(":wrong_func_name") for t in types):
        return WRONG_FUNCTION
    if types[0].endswith(":wrong_count"):
        return WRONG_COUNT
    if any(t.startswith(("type_error:", "value_error:")) or t.endswith(ARGUMENT_ERRORS) for t in types):
        return WRONG_ARGUMENTS
    return OTHER


def count_failure_types(failures: list[dict]) -> dict[str, int]:
    counts = dict.fromkeys(FAILURE_TYPES, 0)
    for entry in failures:
        counts[classify_failure(entry)] += 1
    return counts


def failed_ids(failures: list[dict]) -> set[str]:
    return {str(entry["id"]) for entry in failures}


def mcnemar_p_value(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar test: could this split of disagreements be chance?

    Only questions the two models answer differently count. If neither model is better,
    each of those is a coin flip, so the p-value is the chance of a split at least this
    uneven from fair coin flips.
    """
    total = only_a + only_b
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(only_a, only_b) + 1)) / 2**total
    return min(1.0, 2 * tail)


def paired_comparison(failures_a: list[dict], failures_b: list[dict]) -> tuple[int, int, float]:
    """Questions only model A got right, only model B got right, and the McNemar p-value."""
    wrong_a, wrong_b = failed_ids(failures_a), failed_ids(failures_b)
    only_a, only_b = len(wrong_b - wrong_a), len(wrong_a - wrong_b)
    return only_a, only_b, mcnemar_p_value(only_a, only_b)


def question_text(entry: dict) -> str:
    """The user's question from a failure entry's prompt."""
    prompt = entry.get("prompt", {})
    turns = prompt.get("question", []) if isinstance(prompt, dict) else []
    return "\n".join(m["content"] for turn in turns for m in turn if m.get("role") == "user")


def expected_answer(entry: dict) -> str:
    """The accepted calls, or a note that no call was expected."""
    if "irrelevance" in entry.get("test_category", ""):
        return "No tool call: none of the given functions can answer this."
    answer = entry.get("possible_answer", {})
    truth = answer.get("ground_truth", answer) if isinstance(answer, dict) else answer
    return json.dumps(truth, ensure_ascii=False)


def accuracy_value(raw: str) -> float:
    """A BFCL accuracy, stored as 0.85 or 85.00%, as a number of percentage points."""
    value = raw.strip()
    return float(value.rstrip("%")) if value.endswith("%") else float(value) * 100


def comparison_table(rows: dict[str, dict[str, str]]) -> str:
    """Accuracy per category for each model, and each later model's change vs the first."""
    names = list(rows)
    base = names[0]
    header = ["Category", *names, *[f"{name} vs {base}" for name in names[1:]]]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for category, column in CATEGORY_COLUMNS.items():
        values = [accuracy_value(rows[name][column]) for name in names]
        changes = [f"{value - values[0]:+.1f}" for value in values[1:]]
        cells = [f"`{category}`", *[f"{v:.2f}%" for v in values], *changes]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def failure_table(counts: dict[str, dict[str, int]]) -> str:
    """Number of failures of each type, one column per model."""
    names = list(counts)
    lines = ["| Failure type | " + " | ".join(names) + " |", "|" + " --- |" * (len(names) + 1)]
    for failure_type in FAILURE_TYPES:
        cells = [str(counts[name][failure_type]) for name in names]
        lines.append(f"| {failure_type} | " + " | ".join(cells) + " |")
    totals = [str(sum(counts[name].values())) for name in names]
    lines.append("| **total** | " + " | ".join(totals) + " |")
    return "\n".join(lines)


def paired_summary(name_a: str, name_b: str, only_a: int, only_b: int, p_value: float) -> str:
    """One sentence on whether two models really differ, from a paired comparison."""
    verdict = (
        "a split this uneven is unlikely to be chance"
        if p_value < 0.05
        else "a split like this could easily be chance, so the two are not shown to differ"
    )
    return (
        f"On the same 1,240 questions, {name_a} got {only_a} right that {name_b} missed, and "
        f"{name_b} got {only_b} right that {name_a} missed (exact McNemar p = {p_value:.3f}): "
        f"{verdict}."
    )

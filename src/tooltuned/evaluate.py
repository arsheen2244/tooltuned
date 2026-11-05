"""Score the base and fine-tuned models on BFCL."""

import csv
import hashlib
import json
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

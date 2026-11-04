"""Score the base and fine-tuned models on BFCL."""

import csv
from pathlib import Path

SCORE_FILE_NAME = "data_non_live.csv"

# Maps each BFCL test category to its column in data_non_live.csv.
CATEGORY_COLUMNS: dict[str, str] = {
    "simple_python": "Python Simple AST",
    "multiple": "Multiple AST",
    "parallel": "Parallel AST",
    "parallel_multiple": "Parallel Multiple AST",
    "irrelevance": "Irrelevance Detection",
}


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

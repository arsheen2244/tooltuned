"""Print per-category accuracy from a results folder as a markdown table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tooltuned.evaluate import read_score_row, scores_to_markdown  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/print_scores.py <results folder>")
    print(scores_to_markdown(read_score_row(Path(sys.argv[1]))))


if __name__ == "__main__":
    main()

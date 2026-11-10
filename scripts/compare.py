"""Compare the base, verified and unverified models and write results/RESULTS.md.

    python scripts/compare.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tooltuned.evaluate import (  # noqa: E402
    comparison_table,
    count_failure_types,
    failure_table,
    load_failures,
    paired_table,
    read_score_row,
)

RESULTS = ROOT / "results"
MODELS = {
    "Base": RESULTS / "baseline",
    "Verified": RESULTS / "tooltuned_verified",
    "Unverified": RESULTS / "tooltuned_unverified",
}
LABELS = RESULTS / "tooltuned_verified" / "failure_labels.json"
NOTES = RESULTS / "analysis_notes.md"


def label_section() -> list[str]:
    """The most common hand-labelled failure types, if the labels exist yet."""
    if not LABELS.exists():
        return ["_Not labelled yet._"]
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    labels = [item["label"] for item in data["labels"]]
    lines = [f"From {len(labels)} randomly chosen failures of the verified model "
             f"(labelled by {data['labelled_by']}):", ""]
    ranked = Counter(labels).most_common()
    cutoff = ranked[min(3, len(ranked) - 1)][1]  # keep the top 4, plus anything tied with the 4th
    lines += [f"- **{label}**: {count}" for label, count in ranked if count >= cutoff]
    return lines


def build_report() -> str:
    rows = {name: read_score_row(folder) for name, folder in MODELS.items()}
    failures = {name: load_failures(folder) for name, folder in MODELS.items()}
    sections = [
        "# Results",
        "",
        "BFCL accuracy on 5 categories (1,240 questions). Base is `Qwen/Qwen2.5-1.5B-Instruct`. "
        "Verified and Unverified are the same QLoRA recipe (r=16, 2 epochs, 10,000 rows) trained "
        "on data with and without the `verify.py` checks. Changes are in percentage points.",
        "",
        comparison_table(rows),
        "",
        "## Are the differences real?",
        "",
        "Each pair of models answered the same questions, so this counts the questions only one "
        "of the two got right and runs an exact McNemar test: the chance of a split at least this "
        "uneven if neither model were better. Below 0.05 is read as a real difference. Tool-calling "
        "and irrelevance questions are tested apart, because fine-tuning moved them in opposite "
        "directions. The test covers question-to-question luck only, not training luck: each model "
        "was trained once.",
        "",
        paired_table(failures, [("Verified", "Base"), ("Unverified", "Base"), ("Verified", "Unverified")]),
        "",
        "## Failure types",
        "",
        "Counted automatically from BFCL's score files (see `evaluate.classify_failure`).",
        "",
        failure_table({name: count_failure_types(items) for name, items in failures.items()}),
        "",
        "## Labelled failures",
        "",
        *label_section(),
    ]
    if NOTES.exists():
        sections += ["", "## Notes", "", NOTES.read_text(encoding="utf-8").strip()]
    return "\n".join(sections) + "\n"


def main() -> None:
    (RESULTS / "RESULTS.md").write_text(build_report(), encoding="utf-8")
    print(f"Wrote {RESULTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()

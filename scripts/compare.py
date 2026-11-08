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
    paired_comparison,
    paired_summary,
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
    labels = [item["label"] for item in json.loads(LABELS.read_text(encoding="utf-8"))]
    lines = [f"From {len(labels)} randomly chosen failures of the verified model:", ""]
    lines += [f"{rank}. **{label}**: {count}" for rank, (label, count) in enumerate(Counter(labels).most_common(4), 1)]
    return lines


def build_report() -> str:
    rows = {name: read_score_row(folder) for name, folder in MODELS.items()}
    failures = {name: load_failures(folder) for name, folder in MODELS.items()}
    only_v, only_u, p_value = paired_comparison(failures["Verified"], failures["Unverified"])
    sections = [
        "# Results",
        "",
        "BFCL accuracy on 5 categories (1,240 questions). Base is `Qwen/Qwen2.5-1.5B-Instruct`. "
        "Verified and Unverified are the same QLoRA recipe (r=16, 2 epochs, 10,000 rows) trained "
        "on data with and without the `verify.py` checks. Changes are in percentage points.",
        "",
        comparison_table(rows),
        "",
        "## Did verification help?",
        "",
        paired_summary("Verified", "Unverified", only_v, only_u, p_value),
        "",
        "## Failure types",
        "",
        "Counted automatically from BFCL's score files (see `evaluate.classify_failure`).",
        "",
        failure_table({name: count_failure_types(items) for name, items in failures.items()}),
        "",
        "## Hand-labelled failures",
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

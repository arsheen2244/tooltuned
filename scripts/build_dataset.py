"""Build the train and validation sets and push them to a private Hub dataset.

Use --no-push to build and report without uploading anything.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tooltuned.data import (  # noqa: E402
    GRAM_SIZE,
    MAX_BFCL_OVERLAP,
    Row,
    build_final_dataset,
    build_question_index,
    load_raw_dataset,
    push_splits_to_hub,
    remove_bfcl_duplicates,
    sample_splits,
)
from tooltuned.evaluate import download_bfcl_wheel, load_bfcl_questions  # noqa: E402
from tooltuned.verify import update_report  # noqa: E402

SEED = 42
REPORT_PATH = ROOT / "results" / "data_report.json"
BFCL_WHEEL_PATH = ROOT / "outputs" / "bfcl" / "bfcl_eval-2026.3.23-py3-none-any.whl"


def _split_counts(rows: list[Row]) -> dict[str, int]:
    no_tool = sum(1 for row in rows if row["reply"])
    return {"rows": len(rows), "tool_call": len(rows) - no_tool, "no_tool": no_tool}


def _summary(
    raw_count: int,
    verified_count: int,
    pools: dict[str, list[Row]],
    removed: dict[str, dict[str, int]],
    bfcl_question_count: int,
    splits: dict[str, list[Row]],
) -> dict[str, Any]:
    verified_ids = {row["id"] for row in splits["train_verified"]}
    return {
        "seed": SEED,
        "raw_rows": raw_count,
        "rows_passing_verifier": verified_count,
        "no_tool_rows_generated": len(pools["no_tool_before"]),
        "bfcl_filter": {
            "bfcl_questions_compared": bfcl_question_count,
            "method": (
                f"normalized text, then character {GRAM_SIZE}-grams; removed above "
                f"{MAX_BFCL_OVERLAP:.0%} of the shorter text's grams"
            ),
            "removed_tool_call_rows": removed["positives"],
            "removed_no_tool_rows": removed["no_tool"],
        },
        "pool_after_filter": {
            "tool_call_rows": len(pools["positives"]),
            "no_tool_rows": len(pools["no_tool"]),
        },
        "splits": {name: _split_counts(rows) for name, rows in splits.items()},
        "unverified_train_rows_not_in_verified_train": sum(
            1 for row in splits["train_unverified"] if row["id"] not in verified_ids
        ),
    }


def build(token_is_set: bool) -> tuple[dict[str, list[Row]], dict[str, Any]]:
    if not token_is_set:
        sys.exit("Set the HF_TOKEN environment variable first.")
    rows = list(load_raw_dataset())
    final = build_final_dataset(rows, seed=SEED)
    no_tool_before = [row for row in final if row["reply"]]
    verified_count = len(final) - len(no_tool_before)

    questions = load_bfcl_questions(download_bfcl_wheel(BFCL_WHEEL_PATH))
    index = build_question_index(questions)
    positives, removed_positives = remove_bfcl_duplicates(
        [{**row, "reply": ""} for row in rows], index
    )
    no_tool, removed_no_tool = remove_bfcl_duplicates(no_tool_before, index)

    splits = sample_splits(positives, no_tool, seed=SEED)
    pools = {"no_tool_before": no_tool_before, "positives": positives, "no_tool": no_tool}
    removed = {"positives": removed_positives, "no_tool": removed_no_tool}
    summary = _summary(len(rows), verified_count, pools, removed, len(questions), splits)
    return splits, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-push", action="store_true", help="Don't upload to the Hub.")
    arguments = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    splits, summary = build(bool(token))
    summary["hub_repo"] = None if arguments.no_push else push_splits_to_hub(splits, token)
    update_report(REPORT_PATH, {"final_dataset": summary})

    filter_counts = summary["bfcl_filter"]
    print(f"Removed tool-call rows matching BFCL: {filter_counts['removed_tool_call_rows']}")
    print(f"Removed no-tool rows matching BFCL: {filter_counts['removed_no_tool_rows']}")
    for name, counts in summary["splits"].items():
        print(f"{name}: {counts}")
    print(f"Hub repo: {summary['hub_repo']}")
    print(f"Updated {REPORT_PATH}")


if __name__ == "__main__":
    main()

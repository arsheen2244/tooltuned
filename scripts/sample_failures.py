"""Pick random failures of a model for hand-labelling, and print one of them.

    python scripts/sample_failures.py                # pick 20 and save them
    python scripts/sample_failures.py --show 3       # print failure number 3 of the sample
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tooltuned.evaluate import (  # noqa: E402
    classify_failure,
    expected_answer,
    load_failures,
    question_text,
    raw_output,
)

FOLDER = ROOT / "results" / "tooltuned_verified"
SAMPLE = FOLDER / "failure_sample.json"


def summarise(entry: dict) -> dict[str, str]:
    """The parts of a failure a person needs to label it."""
    return {
        "id": str(entry["id"]),
        "category": entry["test_category"],
        "question": question_text(entry),
        "expected": expected_answer(entry),
        "model_output": raw_output(entry),
        "bfcl_error": str(entry.get("error_type", "")),
        "auto_type": classify_failure(entry),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", type=int, help="Print this failure (1-based) of the saved sample.")
    arguments = parser.parse_args()

    if arguments.show:
        item = json.loads(SAMPLE.read_text(encoding="utf-8"))[arguments.show - 1]
        print(json.dumps(item, indent=2, ensure_ascii=False))
        return
    failures = sorted(load_failures(FOLDER), key=lambda entry: str(entry["id"]))
    sample = random.Random(arguments.seed).sample(failures, arguments.count)
    SAMPLE.write_text(json.dumps([summarise(e) for e in sample], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {arguments.count} of {len(failures)} failures to {SAMPLE}")


if __name__ == "__main__":
    main()

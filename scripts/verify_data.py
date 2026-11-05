"""Check every dataset row and save results/data_report.json."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tooltuned.data import load_raw_dataset  # noqa: E402
from tooltuned.verify import build_report, update_report  # noqa: E402

REPORT_PATH = ROOT / "results" / "data_report.json"


def main() -> None:
    report = build_report(load_raw_dataset())
    update_report(REPORT_PATH, report)
    print(f"Accepted {report['accepted']} of {report['total_rows']}")
    for code, count in report["rejected_by_reason"].items():
        print(f"{code}: {count}")
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()

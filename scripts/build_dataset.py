"""Build data/final_dataset.jsonl and print 5 no-tool examples."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from tooltuned.data import (  # noqa: E402
    BASE_MODEL,
    build_final_dataset,
    load_raw_dataset,
    render_example,
)

OUTPUT_PATH = ROOT / "data" / "final_dataset.jsonl"


def main() -> None:
    final = build_final_dataset(list(load_raw_dataset()))
    no_tool = [row for row in final if row["reply"]]
    print(f"Final rows: {len(final)}")
    print(f"No-tool rows: {len(no_tool)} ({len(no_tool) / len(final):.2%})")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in final:
            file.write(json.dumps(row) + "\n")
    print(f"Saved {OUTPUT_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    for index, row in enumerate(no_tool[:5]):
        print(f"\n===== NO-TOOL EXAMPLE {index} =====")
        print(render_example(row, tokenizer))


if __name__ == "__main__":
    main()

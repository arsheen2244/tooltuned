"""Print raw dataset rows and the text the model will learn from them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from tooltuned.data import (  # noqa: E402
    BASE_MODEL,
    has_usable_fields,
    load_raw_dataset,
    render_example,
)


def main() -> None:
    dataset = load_raw_dataset()
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    for index in range(2):
        print(f"\n===== RAW ROW {index} =====")
        for field, value in dataset[index].items():
            print(f"[{field}] ({type(value).__name__})\n{value}\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    for index in range(3):
        print(f"\n===== RENDERED EXAMPLE {index} =====")
        print(render_example(dataset[index], tokenizer))

    unusable = sum(1 for row in dataset if not has_usable_fields(row))
    print(f"\nRows with unusable tools or answers: {unusable} of {len(dataset)}")


if __name__ == "__main__":
    main()

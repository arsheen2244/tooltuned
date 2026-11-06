"""Print token length percentiles and recommend a max_seq_length."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from tooltuned.data import BASE_MODEL, load_train_split  # noqa: E402
from tooltuned.formatting import (  # noqa: E402
    percentile,
    recommend_max_seq_length,
    share_within,
    token_length,
)

PERCENTILES = (50, 90, 95, 99)


def _print_stats(label: str, lengths: list[int]) -> None:
    ordered = sorted(lengths)
    values = [f"p{p}={percentile(ordered, p)}" for p in PERCENTILES]
    print(f"{label} ({len(ordered)} rows): {' '.join(values)} max={ordered[-1]}")


def main() -> None:
    rows = list(load_train_split())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    lengths = [token_length(row, tokenizer) for row in rows]
    tool_call = [n for n, row in zip(lengths, rows) if not row["reply"]]
    no_tool = [n for n, row in zip(lengths, rows) if row["reply"]]

    _print_stats("All", lengths)
    _print_stats("Tool-call", tool_call)
    _print_stats("No-tool", no_tool)

    limit = recommend_max_seq_length(lengths)
    kept = sum(length <= limit for length in lengths)
    print(f"Recommended max_seq_length: {limit}")
    print(f"Keeps {kept} of {len(lengths)} ({share_within(lengths, limit):.2%})")
    print(f"Over the limit: {len(lengths) - kept}")


if __name__ == "__main__":
    main()

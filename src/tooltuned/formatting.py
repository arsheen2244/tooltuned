"""Turn examples into the model's prompt format."""

import math

from transformers import PreTrainedTokenizerBase

from tooltuned.data import Row, render_example


def token_length(row: Row, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count the tokens in a rendered example, which already has its special tokens."""
    text = render_example(row, tokenizer)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def percentile(sorted_lengths: list[int], percent: float) -> int:
    """Return the nearest-rank percentile, which is always a real example's length."""
    if not sorted_lengths:
        raise ValueError("Can't take a percentile of no lengths.")
    # round() stops tiny floating-point errors from moving the rank up by one.
    rank = math.ceil(round(len(sorted_lengths) * percent / 100, 9))
    return sorted_lengths[max(rank, 1) - 1]


def share_within(lengths: list[int], limit: int) -> float:
    """Return the fraction of examples no longer than the limit."""
    if not lengths:
        raise ValueError("Can't take a share of no lengths.")
    return sum(length <= limit for length in lengths) / len(lengths)


def recommend_max_seq_length(
    lengths: list[int], keep: float = 0.98, round_to: int = 8
) -> int:
    """Return the shortest limit that keeps the given share, rounded up."""
    needed = percentile(sorted(lengths), keep * 100)
    return math.ceil(needed / round_to) * round_to

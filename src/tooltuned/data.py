"""Load and prepare the function-calling training data."""

import json
import os
import random
import re
from functools import lru_cache
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from tooltuned.verify import verify_row

DATASET_NAME = "Salesforce/xlam-function-calling-60k"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

NO_TOOL_FRACTION = 0.1
MAX_DONOR_ATTEMPTS = 25
NO_TOOL_REPLIES = (
    "None of the available tools fit this request.",
    "I can't help with this using the available tools, because none of them fit the request.",
    "The available tools don't cover this request, so I can't make a function call for it.",
)
# Common words that say nothing about a function's topic, so sharing one isn't a link.
GENERIC_WORDS = frozenset(
    """
    about above after again all also any are available based been before being both can
    could default does each for from get given had has have how into its may more most
    need not of off one only other our out over provide provided same should some such
    than that the their them then there these they this those through use used using
    want was were what when where which while who why will with would you your
    api calculate compute check data detail fetch find information info list name number
    return result retrieve specified type value
    """.split()
)
# A function like web search could answer almost any query, so it can't be "unrelated".
OPEN_ENDED_WORDS = frozenset(
    """
    answer ask assistant bing browse chat chatbot completion conversation duckduckgo
    engine gemini generate google gpt internet llm prompt query question search text
    translate translation web wikipedia
    """.split()
)

Row = dict[str, Any]
Message = dict[str, Any]


def load_raw_dataset() -> Dataset:
    """Download the dataset, which is gated and so needs HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set the HF_TOKEN environment variable to load the dataset.")
    return load_dataset(DATASET_NAME, split="train", token=token)


def row_to_messages(row: Row) -> list[Message]:
    """Turn one row into a user message and an assistant tool-call or text message."""
    user_message = {"role": "user", "content": row["query"]}
    if row.get("reply"):
        return [user_message, {"role": "assistant", "content": row["reply"]}]
    # arguments stays a dict: the chat template converts it to JSON itself.
    tool_calls = [
        {
            "type": "function",
            "function": {"name": answer["name"], "arguments": answer["arguments"]},
        }
        for answer in json.loads(row["answers"])
    ]
    return [user_message, {"role": "assistant", "content": "", "tool_calls": tool_calls}]


def render_example(row: Row, tokenizer: PreTrainedTokenizerBase) -> str:
    """Return the exact text the model will learn from for one row."""
    return tokenizer.apply_chat_template(
        row_to_messages(row),
        tools=json.loads(row["tools"]),
        tokenize=False,
    )


def has_usable_fields(row: Row) -> bool:
    """Check that tools and answers parse and that every answer is a complete call."""
    try:
        tools = json.loads(row["tools"])
        answers = json.loads(row["answers"])
    except (TypeError, json.JSONDecodeError):
        return False
    if not (isinstance(tools, list) and tools and isinstance(answers, list) and answers):
        return False
    return all(
        isinstance(answer, dict)
        and isinstance(answer.get("name"), str)
        and isinstance(answer.get("arguments"), dict)
        for answer in answers
    )


def _content_words(text: str) -> frozenset[str]:
    """Return the lowercase topic words in some text, without plural endings."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()
    words = (word.rstrip("s") for word in re.findall(r"[a-z]{3,}", spaced))
    return frozenset(word for word in words if word and word not in GENERIC_WORDS)


@lru_cache(maxsize=None)
def _tools_words(tools_json: str) -> frozenset[str]:
    """Return the topic words in tool names, descriptions and parameter names."""
    pieces = []
    for tool in json.loads(tools_json):
        pieces += [tool["name"], tool["description"], " ".join(tool["parameters"])]
    return _content_words(" ".join(pieces))


def _tool_names(tools_json: str) -> set[str]:
    return {tool["name"] for tool in json.loads(tools_json)}


def is_clearly_unrelated(row: Row, donor: Row) -> bool:
    """Check that a donor's tools can't plausibly serve the row's query.

    This compares words, so it can't understand meaning. It's deliberately strict: no
    shared function names, no open-ended tools and no topic words in common.
    """
    if _tool_names(row["tools"]) & _tool_names(donor["tools"]):
        return False
    donor_words = _tools_words(donor["tools"])
    if donor_words & OPEN_ENDED_WORDS:
        return False
    return not donor_words & (_content_words(row["query"]) | _tools_words(row["tools"]))


def make_no_tool_example(row: Row, donor: Row, new_id: int, rng: random.Random) -> Row:
    """Pair a row's query with a donor's tools and a reply that declines to call them."""
    return {
        "id": new_id,
        "query": row["query"],
        "answers": "[]",
        "tools": donor["tools"],
        "reply": rng.choice(NO_TOOL_REPLIES),
    }


def _find_donor(row: Row, rows: list[Row], rng: random.Random) -> Row | None:
    for _ in range(MAX_DONOR_ATTEMPTS):
        donor = rng.choice(rows)
        if donor is not row and is_clearly_unrelated(row, donor):
            return donor
    return None


def make_no_tool_examples(rows: list[Row], count: int, rng: random.Random) -> list[Row]:
    """Make no-tool examples, using each query at most once."""
    next_id = max(row["id"] for row in rows) + 1
    examples: list[Row] = []
    for row in rng.sample(rows, len(rows)):
        if len(examples) == count:
            break
        donor = _find_donor(row, rows, rng)
        if donor is not None:
            examples.append(make_no_tool_example(row, donor, next_id + len(examples), rng))
    if len(examples) < count:
        raise ValueError(f"Only made {len(examples)} of {count} no-tool examples.")
    return examples


def no_tool_count(positive_count: int, fraction: float = NO_TOOL_FRACTION) -> int:
    """Return how many no-tool rows make them this fraction of the final dataset."""
    return round(positive_count * fraction / (1 - fraction))


def build_final_dataset(
    rows: list[Row], fraction: float = NO_TOOL_FRACTION, seed: int = 42
) -> list[Row]:
    """Drop rows that fail verification, add no-tool examples and shuffle."""
    rng = random.Random(seed)
    positives = [{**row, "reply": ""} for row in rows if verify_row(row)[0]]
    no_tool = make_no_tool_examples(positives, no_tool_count(len(positives), fraction), rng)
    final = positives + no_tool
    rng.shuffle(final)
    return final

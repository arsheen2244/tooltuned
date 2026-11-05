"""Load and prepare the function-calling training data."""

import json
import os
import random
import re
from collections import Counter
from functools import lru_cache
from typing import Any, NamedTuple

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizerBase

from tooltuned.verify import verify_row

DATASET_NAME = "Salesforce/xlam-function-calling-60k"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

HUB_DATASET_NAME = "tooltuned-fc-data"
TRAIN_SIZE = 10_000
VALIDATION_SIZE = 500
GRAM_SIZE = 8
MAX_BFCL_OVERLAP = 0.5

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


def normalize_text(text: str) -> str:
    """Lowercase, replace punctuation with spaces and collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def char_grams(normalized: str) -> frozenset[str]:
    """Return every run of GRAM_SIZE characters in already normalized text."""
    return frozenset(
        normalized[start : start + GRAM_SIZE]
        for start in range(len(normalized) - GRAM_SIZE + 1)
    )


class QuestionIndex(NamedTuple):
    """BFCL questions, arranged so each query only compares against likely matches."""

    normalized: frozenset[str]
    gram_sets: list[frozenset[str]]
    questions_by_gram: dict[str, list[int]]


def build_question_index(questions: list[str]) -> QuestionIndex:
    normalized = [normalize_text(question) for question in questions]
    gram_sets = [char_grams(text) for text in normalized]
    questions_by_gram: dict[str, list[int]] = {}
    for index, grams in enumerate(gram_sets):
        for gram in grams:
            questions_by_gram.setdefault(gram, []).append(index)
    return QuestionIndex(frozenset(normalized), gram_sets, questions_by_gram)


def bfcl_match(query: str, index: QuestionIndex) -> str:
    """Return "exact", "near" or "" for how closely a query matches a BFCL question.

    Near means more than half of the shorter text's character 8-grams appear in the
    other, which also catches a long multi-part query containing a BFCL question.
    """
    normalized = normalize_text(query)
    if normalized in index.normalized:
        return "exact"
    grams = char_grams(normalized)
    shared = Counter(
        question for gram in grams for question in index.questions_by_gram.get(gram, ())
    )
    for question, count in shared.items():
        if count / min(len(grams), len(index.gram_sets[question])) > MAX_BFCL_OVERLAP:
            return "near"
    return ""


def remove_bfcl_duplicates(
    rows: list[Row], index: QuestionIndex
) -> tuple[list[Row], dict[str, int]]:
    """Drop rows whose query matches a BFCL question, and count them by match type."""
    kept: list[Row] = []
    removed = {"exact": 0, "near": 0}
    for row in rows:
        match = bfcl_match(row["query"], index)
        if match:
            removed[match] += 1
        else:
            kept.append(row)
    return kept, removed


def _first_rows(rows: list[Row], count: int, description: str) -> list[Row]:
    if len(rows) < count:
        raise ValueError(f"Need {count} {description} rows, only have {len(rows)}.")
    return rows[:count]


def sample_splits(
    positives: list[Row],
    no_tool: list[Row],
    seed: int = 42,
    train_size: int = TRAIN_SIZE,
    validation_size: int = VALIDATION_SIZE,
    fraction: float = NO_TOOL_FRACTION,
) -> dict[str, list[Row]]:
    """Sample validation, verified train and unverified train sets.

    positives holds every tool-call row, including ones the verifier rejects.
    Both train sets read the same shuffled order and share their no-tool rows, so they
    differ only where the verifier rejected a row. No query appears in both train and
    validation, so validation only measures unseen queries.
    """
    rng = random.Random(seed)
    ordered = rng.sample(sorted(positives, key=lambda row: row["id"]), len(positives))
    ordered_no_tool = rng.sample(sorted(no_tool, key=lambda row: row["id"]), len(no_tool))
    verified = [row for row in ordered if verify_row(row)[0]]

    validation_no_tool = round(validation_size * fraction)
    validation = _first_rows(ordered_no_tool, validation_no_tool, "no-tool") + _first_rows(
        verified, validation_size - validation_no_tool, "verified"
    )
    seen = {normalize_text(row["query"]) for row in validation}

    def unseen(rows: list[Row]) -> list[Row]:
        return [row for row in rows if normalize_text(row["query"]) not in seen]

    train_no_tool = round(train_size * fraction)
    shared_no_tool = _first_rows(unseen(ordered_no_tool), train_no_tool, "no-tool")
    train_positive = train_size - train_no_tool
    splits = {
        "validation": validation,
        "train_verified": shared_no_tool
        + _first_rows(unseen(verified), train_positive, "verified"),
        "train_unverified": shared_no_tool
        + _first_rows(unseen(ordered), train_positive, "unverified"),
    }
    return {name: rng.sample(rows, len(rows)) for name, rows in splits.items()}


def to_hub_row(row: Row) -> Row:
    """Add the chat messages, as a JSON string because their arguments vary by row."""
    return {**row, "messages": json.dumps(row_to_messages(row))}


def _to_dataset_dict(train: list[Row], validation: list[Row]) -> DatasetDict:
    return DatasetDict(
        train=Dataset.from_list([to_hub_row(row) for row in train]),
        validation=Dataset.from_list([to_hub_row(row) for row in validation]),
    )


def push_splits_to_hub(splits: dict[str, list[Row]], token: str) -> str:
    """Push both configs to a private dataset under the token's account."""
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{HUB_DATASET_NAME}"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise RuntimeError(f"{repo_id} already exists and is public, so nothing was pushed.")
    for config, train in (
        ("verified", splits["train_verified"]),
        ("unverified", splits["train_unverified"]),
    ):
        _to_dataset_dict(train, splits["validation"]).push_to_hub(
            repo_id, config_name=config, private=True, token=token
        )
    return repo_id

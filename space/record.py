"""Run the demo examples through the model and save the answers to examples.json.

The hosted Space is a static page, so it shows these recorded answers.
Usage: python record.py [float32|bfloat16]
"""

import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

from demo import EXAMPLES, MAX_NEW_TOKENS, MODEL_ID, answer, load_model, parse_tool_calls, summarize

OUTPUT = Path(__file__).with_name("examples.json")


def main(dtype_name: str) -> None:
    tokenizer, model = load_model(dtype_name)
    records = []
    for label, tools, query in EXAMPLES:
        start = time.perf_counter()
        raw = answer(tokenizer, model, tools, query)
        seconds = round(time.perf_counter() - start, 1)
        records.append({
            "label": label,
            "tools": tools,
            "query": query,
            "raw": raw,
            "calls": parse_tool_calls(raw),
            "summary": summarize(raw),
            "seconds": seconds,
        })
        print(f"{label} ({seconds}s): {summarize(raw)}")
    recording = {
        "model": MODEL_ID,
        "model_revision": HfApi().model_info(MODEL_ID).sha,
        "dtype": dtype_name,
        "device": "CPU",
        "decoding": f"greedy, at most {MAX_NEW_TOKENS} new tokens",
        "examples": records,
    }
    OUTPUT.write_text(json.dumps(recording, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "float32")

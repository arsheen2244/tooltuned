"""Load the merged model from the Hub and check it makes one correct tool call.

    python scripts/sanity_check.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tooltuned.merge import MERGED_MODEL_NAME, account_repo_id  # noqa: E402
from tooltuned.train import generate_answer  # noqa: E402

TOOL = {"name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"city": {"description": "City name.", "type": "str"}}}
EXPECTED = {"name": "get_weather", "arguments": {"city": "Paris"}}

token = os.environ["HF_TOKEN"]
repo_id = account_repo_id(MERGED_MODEL_NAME, token)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # CPUs are slow at fp16.
model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype, token=token)
model.to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
row = {"query": "What's the weather like in Paris right now?", "tools": json.dumps([TOOL])}
answer = generate_answer(model, tokenizer, row, max_new_tokens=64)
print(answer)
call = json.loads(answer.split("<tool_call>")[1].split("</tool_call>")[0]) if "<tool_call>" in answer else None
print("PASS: correct tool call" if call == EXPECTED else f"FAIL: expected {EXPECTED}")
sys.exit(0 if call == EXPECTED else 1)

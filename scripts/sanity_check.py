"""Load a merged model from the Hub and check it makes one correct tool call.

    python scripts/sanity_check.py
    python scripts/sanity_check.py --config configs/qlora_r16_unverified.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tooltuned.merge import account_repo_id  # noqa: E402
from tooltuned.train import generate_answer, load_config  # noqa: E402

TOOL = {"name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"city": {"description": "City name.", "type": "str"}}}
EXPECTED = {"name": "get_weather", "arguments": {"city": "Paris"}}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=ROOT / "configs" / "qlora_r16.yaml")
token = os.environ["HF_TOKEN"]
repo_id = account_repo_id(load_config(parser.parse_args().config).merged_model_name, token)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # CPUs are slow at fp16.
model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype, token=token)
model.to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
row = {"query": "What's the weather like in Paris right now?", "tools": json.dumps([TOOL])}
answer = generate_answer(model, tokenizer, row, max_new_tokens=64)
print(f"{repo_id}:\n{answer}")
call = json.loads(answer.split("<tool_call>")[1].split("</tool_call>")[0]) if "<tool_call>" in answer else None
print("PASS: correct tool call" if call == EXPECTED else f"FAIL: expected {EXPECTED}")
sys.exit(0 if call == EXPECTED else 1)

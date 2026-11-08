"""Merge the trained adapter into the base model and push the result to a private Hub repo.

Needs no GPU (it runs on the CPU in a few minutes) and about 4 GB of free memory:
    python scripts/merge_and_push.py
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from tooltuned.merge import MERGED_MODEL_NAME, account_repo_id, merge_adapter  # noqa: E402
from tooltuned.train import load_config, push_private  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "qlora_r16.yaml")
    parser.add_argument("--save-dir", type=Path, help="Also save the merged model here.")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    token = os.environ["HF_TOKEN"]
    adapter = account_repo_id(config.hub_model_name, token)
    print(f"Merging {adapter} into {config.base_model} ...")
    model = merge_adapter(config.base_model, adapter, token)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    if arguments.save_dir:
        model.save_pretrained(arguments.save_dir)
        tokenizer.save_pretrained(arguments.save_dir)
        print(f"Saved the merged model to {arguments.save_dir}")
    repo_id = push_private(model, tokenizer, MERGED_MODEL_NAME, token)
    print(f"Pushed the merged model to {repo_id}")


if __name__ == "__main__":
    main()

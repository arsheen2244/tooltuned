"""Merge the trained adapter into the base model and push the result to a private Hub repo.

Needs no GPU (it runs on the CPU in a few minutes) and about 4 GB of free memory:
    python scripts/merge_and_push.py --save-dir /kaggle/working/merged
    python scripts/merge_and_push.py --config configs/qlora_r16_unverified.yaml
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tooltuned.merge import (  # noqa: E402
    account_repo_id,
    merge_adapter,
    push_folder_private,
    save_merged,
)
from tooltuned.train import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "qlora_r16.yaml")
    parser.add_argument(
        "--save-dir", type=Path, default=ROOT / "outputs" / "merged",
        help="Where to save the merged model before uploading it. Replaced if it exists.",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    token = os.environ["HF_TOKEN"]
    adapter = account_repo_id(config.hub_model_name, token)
    print(f"Merging {adapter} into {config.base_model} ...")
    model = merge_adapter(config.base_model, adapter, token)
    save_merged(model, config.base_model, arguments.save_dir, token)
    print(f"Saved the merged model to {arguments.save_dir}")
    repo_id = push_folder_private(arguments.save_dir, config.merged_model_name, token)
    print(f"Pushed the merged model to {repo_id}")


if __name__ == "__main__":
    main()

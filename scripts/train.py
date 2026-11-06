"""Fine-tune the base model and push the adapter.

Quick test, which never pushes:
    python scripts/train.py --max-steps 20 --train-limit 200
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before torch loads, so a 2-GPU Kaggle notebook trains on one GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tooltuned.train import load_config, run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "qlora_r16.yaml")
    parser.add_argument("--max-steps", type=int, help="Stop after this many updates.")
    parser.add_argument("--train-limit", type=int, help="Use only this many training rows.")
    parser.add_argument("--no-push", action="store_true", help="Don't push the adapter.")
    arguments = parser.parse_args()

    run_training(
        load_config(arguments.config),
        max_steps=arguments.max_steps,
        train_limit=arguments.train_limit,
        push=not arguments.no_push,
    )


if __name__ == "__main__":
    main()

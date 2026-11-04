# ToolTuned

Portfolio project for AI Engineering roles. Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA so it calls functions more accurately, and measure it against the base model on the Berkeley Function Calling Leaderboard (BFCL).

The owner is a beginner: explain new files and concepts briefly and plainly.

## Folder layout

```
src/tooltuned/
  data.py        # load and prepare the training data
  verify.py      # check the prepared data is valid
  formatting.py  # turn examples into the model's prompt format
  train.py       # QLoRA fine-tuning
  evaluate.py    # BFCL scoring against the base model
scripts/         # command-line entry points
notebooks/       # Kaggle notebook instructions
configs/         # YAML configs
tests/           # pytest tests
results/         # scores (committed)
data/            # datasets (git-ignored)
outputs/         # checkpoints and adapters (git-ignored)
```

## Constraints

- **Where things run:** training and eval run on a free Kaggle notebook (T4 GPU, 16 GB). Data prep runs on a CPU laptop, so it must not need a GPU.
- **Precision:** the T4 has no bf16 support, so use fp16.
- **Python:** 3.11, with the src layout above.
- **Secrets:** never put secrets in code. Read `HF_TOKEN` and `WANDB_API_KEY` from environment variables.
- **Code style:** keep functions small and typed.
- **Tests:** every module gets tests, added alongside the code they test.
- **BFCL:** runs in its own environment, created by `scripts/run_bfcl.sh`, because its `vllm` needs a different `torch` from `requirements.txt`. BFCL has no built-in Qwen2.5 entry, so `scripts/bfcl_cli.py` registers `Qwen/Qwen2.5-1.5B-Instruct-FC`.

## Setup and tests

Windows:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the tests from the repo root:

```bash
python -m pytest
```

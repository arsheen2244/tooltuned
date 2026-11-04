# ToolTuned

Fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA to call functions more accurately, measured against the base model on the Berkeley Function Calling Leaderboard (BFCL).

## Setup

Requires Python 3.11. On Windows:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set `HF_TOKEN` and `WANDB_API_KEY` as environment variables.

## Tests

```bash
python -m pytest
```

## Results

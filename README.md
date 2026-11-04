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

Baseline for the base model, before fine-tuning, on a T4 GPU with fp16:

| Category | Accuracy |
| --- | --- |
| `simple_python` | 86.50% |
| `multiple` | 84.00% |
| `parallel` | 74.50% |
| `parallel_multiple` | 72.50% |
| `irrelevance` | 72.08% |

These come from running BFCL with a custom model registration, because BFCL has no built-in Qwen2.5 entry, so they aren't official leaderboard results. The run covers these five categories only. BFCL counts categories that weren't run as 0, so the overall accuracy columns in [`results/baseline/`](results/baseline/) understate the model and shouldn't be quoted.

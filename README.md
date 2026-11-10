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

BFCL accuracy on 5 categories (1,240 questions), on a T4 GPU with fp16. **Verified** is the QLoRA model (r=16, 2 epochs, 10,000 rows) trained on data that passed `verify.py`; **Unverified** is the same recipe on data without those checks, as an ablation.

| Category | Questions | Base | Verified | Unverified |
| --- | --- | --- | --- | --- |
| `simple_python` | 400 | 86.50% | 87.50% | 89.25% |
| `multiple` | 200 | 84.00% | 87.00% | 88.50% |
| `parallel` | 200 | 74.50% | 78.00% | 81.00% |
| `parallel_multiple` | 200 | 72.50% | 75.50% | 77.50% |
| `irrelevance` | 240 | 72.50% | 57.92% | 55.00% |

- **Fine-tuning improved tool calling:** both models gained 1–6.5 points on all four tool-calling categories. A paired test on the same questions says the gains are real.
- **It hurt knowing when *not* to call a tool:** `irrelevance` fell by 15–18 points. The training set's no-tool examples were clearly unrelated, while BFCL's are near misses on the same topic. In a labelled sample of failures, most irrelevance errors were near-miss calls.
- **Verification did not help:** the unverified model did better on tool calling (36 vs 16 questions only one model got right, p = 0.008). With one training run each, training luck can't be ruled out.

Full table, paired tests, failure types and labelled failures: [`results/RESULTS.md`](results/RESULTS.md), built by `python scripts/compare.py`. Raw scores are in [`results/`](results/).

These come from running BFCL with a custom model registration, because BFCL has no built-in Qwen2.5 entry, so they aren't official leaderboard results. BFCL counts categories that weren't run as 0, so the overall accuracy columns in the CSV files understate every model and shouldn't be quoted.

### Training

![Training and validation loss](results/tooltuned_verified/loss_curves.svg)

Training the verified model took 3 h 48 min on one T4. Validation loss on 487 held-out rows fell from 0.0240 at step 200 to 0.0191 and stayed flat over the last 400 updates, so the model did not overfit. The small rise at step 400 came while the learning rate was near its peak. The numbers come from the training log, saved in [`loss_history.json`](results/tooltuned_verified/loss_history.json); `python scripts/plot_loss.py results/tooltuned_verified` redraws the chart.

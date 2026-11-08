# Kaggle notebook setup

Before you start, set these in the notebook's right panel:

- **Accelerator:** GPU T4 x2
- **Internet:** On
- **Secrets (Add-ons > Secrets):** attach `HF_TOKEN` and `WANDB_API_KEY`

Paste these 3 cells at the top of every notebook and run them in order.

## Cell 1: download and run the setup script

```text
!curl -fsSL https://raw.githubusercontent.com/arsheen2244/tooltuned/main/scripts/kaggle_setup.sh -o /tmp/kaggle_setup.sh && bash /tmp/kaggle_setup.sh
```

The repo isn't on Kaggle yet, so this fetches the script directly from GitHub. The script then clones the repo.

## Cell 2: load secrets into the notebook

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
for name in ("HF_TOKEN", "WANDB_API_KEY"):
    os.environ[name] = secrets.get_secret(name)
```

Anything `!bash` exports disappears when that command ends, so the notebook needs its own copy. Later `!` commands inherit these values.

## Cell 3: move into the repo and check the GPU

```text
%cd /kaggle/working/tooltuned
!nvidia-smi -L
!git log --oneline -1
```

You should see at least one Tesla T4 and your latest commit.

## Baseline BFCL run

Takes roughly 20 to 40 minutes:

```text
!bash scripts/run_bfcl.sh "Qwen/Qwen2.5-1.5B-Instruct-FC" "" baseline
```

Then download `results/baseline/` from the Output panel and put it in your repo's `results/` folder.

## Training

Run a quick test first. It trains for 20 updates on 200 rows and never pushes to the Hub:

```text
!python scripts/train.py --max-steps 20 --train-limit 200
```

Check that it finishes, the loss is a real number, and the 5 generated answers print. Then start the full run, which pushes the adapter to a private repo when it finishes:

```text
!python scripts/train.py
```

Add `--no-push` to keep the adapter on Kaggle only. Logging to Weights & Biases turns on when `WANDB_API_KEY` is set, which Cell 2 does.

## Merge and check

These need no GPU, so they also run on a laptop with `HF_TOKEN` set. The first merges the adapter into the base model and pushes the result to a private repo, `<your-username>/tooltuned-qwen2.5-1.5b`. The second loads that model and checks it makes one correct tool call:

```text
!python scripts/merge_and_push.py
!python scripts/sanity_check.py
```

Add `--save-dir /kaggle/working/merged` to the first command to also keep a local copy for BFCL.

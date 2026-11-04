#!/usr/bin/env bash
# Prepares a Kaggle notebook. See notebooks/README.md for how to run it.
set -euo pipefail

REPO_URL="https://github.com/arsheen2244/tooltuned.git"
REPO_DIR="/kaggle/working/tooltuned"

read_secret() {
  python -c 'import sys; from kaggle_secrets import UserSecretsClient; print(UserSecretsClient().get_secret(sys.argv[1]))' "$1"
}

if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

pip install --quiet -r "$REPO_DIR/requirements.txt"

HF_TOKEN="$(read_secret HF_TOKEN)"
WANDB_API_KEY="$(read_secret WANDB_API_KEY)"
export HF_TOKEN WANDB_API_KEY

python -c 'import os; from huggingface_hub import login; login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)'

nvidia-smi -L || { echo "No GPU found. Set Accelerator to GPU T4 x2." >&2; exit 1; }
python -c 'import torch; print("torch", torch.__version__, "sees GPU:", torch.cuda.is_available())'

echo "Setup complete."

#!/usr/bin/env bash
# Scores a model on BFCL. Usage:
#   bash scripts/run_bfcl.sh <model name> <local model path or ""> <output folder name>
set -euo pipefail

BFCL_VERSION="2026.3.23"
CATEGORIES="simple_python,multiple,parallel,parallel_multiple,irrelevance"
VENV_DIR="/tmp/bfcl-venv"
export BFCL_PROJECT_ROOT="/kaggle/working/bfcl"

usage() {
  echo 'Usage: bash scripts/run_bfcl.sh <model name> <local model path or ""> <output folder name>' >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
MODEL="$1"
LOCAL_MODEL_PATH="$2"
OUTPUT="$3"

[[ -n "$MODEL" ]] || usage
if [[ ! "$OUTPUT" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Output folder name may only use letters, numbers, dots, dashes and underscores." >&2
  exit 2
fi
if [[ -n "$LOCAL_MODEL_PATH" && ! -d "$LOCAL_MODEL_PATH" ]]; then
  echo "Local model path not found: $LOCAL_MODEL_PATH" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# vllm needs torch==2.6.0, which clashes with requirements.txt, so BFCL gets its own environment.
if [[ ! -x "$VENV_DIR/bin/bfcl" ]]; then
  python -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --quiet "bfcl-eval[oss_eval_vllm]==$BFCL_VERSION"
fi
export PATH="$VENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH

GENERATE_ARGS=(
  --model "$MODEL"
  --test-category "$CATEGORIES"
  --backend vllm
  --num-gpus 1
  --result-dir "result/$OUTPUT"
)
if [[ -n "$LOCAL_MODEL_PATH" ]]; then
  GENERATE_ARGS+=(--local-model-path "$LOCAL_MODEL_PATH")
fi

python "$REPO_DIR/scripts/bfcl_cli.py" generate "${GENERATE_ARGS[@]}"
python "$REPO_DIR/scripts/bfcl_cli.py" evaluate \
  --model "$MODEL" \
  --test-category "$CATEGORIES" \
  --result-dir "result/$OUTPUT" \
  --score-dir "score/$OUTPUT"

mkdir -p "$REPO_DIR/results/$OUTPUT"
cp "$BFCL_PROJECT_ROOT/score/$OUTPUT"/data_*.csv "$REPO_DIR/results/$OUTPUT/"

python "$REPO_DIR/scripts/print_scores.py" "$REPO_DIR/results/$OUTPUT"

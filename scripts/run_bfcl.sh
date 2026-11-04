#!/usr/bin/env bash
# Scores a model on BFCL. Usage:
#   bash scripts/run_bfcl.sh <model name> <local model path or ""> <output folder name>
set -euo pipefail

BFCL_VERSION="2026.3.23"
BFCL_TRANSFORMERS_VERSION="4.51.3"
BFCL_OPENAI_VERSION="1.90.0"
BFCL_SOUNDFILE_VERSION="0.14.0"
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

# BFCL's vllm needs Python below 3.13 and torch==2.6.0, which clash with Kaggle's defaults,
# so it gets its own Python 3.11 environment. vllm 0.8.5 is from 2025, so transformers and
# openai are pinned to versions close to its release rather than today's latest.
if [[ ! -x "$VENV_DIR/bin/bfcl" ]]; then
  pip install --quiet uv
  uv venv --clear --python 3.11 "$VENV_DIR"
  uv pip install --quiet --python "$VENV_DIR/bin/python" \
    "bfcl-eval[oss_eval_vllm]==$BFCL_VERSION" \
    "transformers==$BFCL_TRANSFORMERS_VERSION" \
    "openai==$BFCL_OPENAI_VERSION" \
    "soundfile==$BFCL_SOUNDFILE_VERSION"  # qwen-agent imports this without declaring it.
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

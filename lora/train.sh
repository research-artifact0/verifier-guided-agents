#!/usr/bin/env bash
# Train every paper LoRA variant using hyperparameters from .env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-126}"

module load conda/latest cuda/12.6
if [[ -n "${CUDA_HOME:-}" ]]; then
  NVHPC_ROOT="$(cd "${CUDA_HOME}/../.." && pwd)"
  CUDA_PATHS="${NVHPC_ROOT}/math_libs/lib64:${CUDA_HOME}/lib64"
  export LD_LIBRARY_PATH="${CUDA_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda is unavailable'"
python -c "import bitsandbytes.cextension as e; assert e.lib is not None and getattr(e.lib, 'compiled_with_cuda', False), 'bitsandbytes CUDA is unavailable'"

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/paper}"

train_variant() {
  local variant="$1"
  local pairs="$2"
  echo "[$(date '+%H:%M:%S')] ${variant}: ${pairs}"
  python -m lora.train --paper --tensorboard --pairs "$pairs" --out "$variant"
}

[[ -f "${DATA_DIR}/a_beta_all.jsonl" ]] || {
  echo "ERROR: missing ${DATA_DIR}/a_beta_all.jsonl" >&2
  exit 1
}

train_variant filter_on  "${DATA_DIR}/filter_on.jsonl"
train_variant filter_off "${DATA_DIR}/filter_off.jsonl"
train_variant core       "${DATA_DIR}/a_beta_core.jsonl"
train_variant aux        "${DATA_DIR}/a_beta_aux.jsonl"
train_variant all        "${DATA_DIR}/a_beta_all.jsonl"
train_variant rw         "${DATA_DIR}/a_beta_rw.jsonl"

python -m eval.run_table \
  --table 5 \
  --merge \
  --checkpoint-dir "runs/${RUN_ID}/lora"

echo "Training complete: runs/${RUN_ID}/lora"

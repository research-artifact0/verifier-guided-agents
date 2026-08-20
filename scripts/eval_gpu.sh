#!/usr/bin/env bash
# GPU eval wrapper (tables 1-7 suite or single table via run_table).
#   ./scripts/eval_gpu.sh --paper --episodes 12 --checkpoint-dir lora_3b ...
#   ./scripts/eval_gpu.sh --table 1 --paper --variant all --episodes 12 ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/load_env.sh
source "$ROOT/scripts/load_env.sh"
_load_env "$ROOT"
# shellcheck source=scripts/setup_cuda_ld.sh
source "$ROOT/scripts/setup_cuda_ld.sh"

module load conda/latest cuda/12.6
setup_cuda_ld
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: no GPU — run on a GPU node" >&2
  exit 1
fi

./scripts/check_gpu_train.sh

if [[ "${1:-}" == "--table" ]]; then
  exec python -m eval.run_table "$@"
fi
if [[ "${1:-}" == "--extended" ]]; then
  shift
  export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
  exec python -m eval.extended_eval "$@"
fi
exec python eval/run_paper_tables.py "$@"

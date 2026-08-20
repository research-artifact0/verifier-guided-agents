#!/usr/bin/env bash
# Run lora.train on a GPU node with correct CUDA + bitsandbytes env.
# Usage: ./scripts/train_gpu.sh --paper --tensorboard --max-steps 60 --pairs ... --out ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/load_env.sh
source "$ROOT/scripts/load_env.sh"
_load_sal_env "$ROOT"
# shellcheck source=scripts/setup_cuda_ld.sh
source "$ROOT/scripts/setup_cuda_ld.sh"

module load conda/latest cuda/12.6
setup_cuda_ld
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: no GPU — run on a GPU node (salloc/sbatch), not login7" >&2
  exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda unavailable after module load cuda/12.6'"

exec python -m lora.train "$@"

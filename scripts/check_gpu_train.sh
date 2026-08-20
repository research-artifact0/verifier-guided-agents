#!/usr/bin/env bash
# Quick preflight before lora.train on a GPU node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/load_env.sh
source "$ROOT/scripts/load_env.sh"
_load_sal_env "$ROOT"
# shellcheck source=scripts/setup_cuda_ld.sh
source "$ROOT/scripts/setup_cuda_ld.sh"
module load conda/latest cuda/12.6 2>/dev/null || true
setup_cuda_ld
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal
cd "$ROOT"

echo "=== nvidia-smi ==="
nvidia-smi -L
echo "=== torch CUDA ==="
python -c "import torch; print('cuda_available', torch.cuda.is_available(), 'cuda', torch.version.cuda)"
echo "=== bitsandbytes ==="
python -c "
import lora.gpu_env  # noqa: F401
import bitsandbytes.cextension as e
lib = e.lib
ok = lib is not None and getattr(lib, 'compiled_with_cuda', False)
print('bnb_cuda', ok, 'BNB_CUDA_VERSION', __import__('os').environ.get('BNB_CUDA_VERSION'))
assert ok, 'bitsandbytes CPU fallback — run: module load cuda/12.6 && source scripts/setup_cuda_ld.sh'
print('OK')
"

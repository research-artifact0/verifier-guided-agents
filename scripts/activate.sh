#!/usr/bin/env bash
# Source this file to load .env, CUDA libs, and conda env `sal`.
#   source scripts/activate.sh
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
echo "sal env ready ($(python --version 2>&1))"

#!/usr/bin/env bash
# Source after: module load cuda/12.6
# Adds NVHPC math_libs (libcublas) + cuda lib64 to LD_LIBRARY_PATH for bitsandbytes.
setup_cuda_ld() {
  export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-126}"
  local paths=()
  if [[ -n "${CUDA_HOME:-}" ]]; then
    local cuda_lib="${CUDA_HOME}/lib64"
    local nvhpc_root
    nvhpc_root="$(cd "${CUDA_HOME}/../.." && pwd)"
    local math_lib="${nvhpc_root}/math_libs/lib64"
    [[ -d "$math_lib" ]] && paths+=("$math_lib")
    [[ -d "$cuda_lib" ]] && paths+=("$cuda_lib")
  elif [[ -d /modules/opt/linux-ubuntu24.04-x86_64/nvhpc/Linux_x86_64/24.9/math_libs/lib64 ]]; then
    paths+=("/modules/opt/linux-ubuntu24.04-x86_64/nvhpc/Linux_x86_64/24.9/math_libs/lib64")
    paths+=("/modules/opt/linux-ubuntu24.04-x86_64/nvhpc/Linux_x86_64/24.9/cuda/12.6/lib64")
  fi
  if ((${#paths[@]})); then
    local joined
    joined=$(IFS=:; echo "${paths[*]}")
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  module load conda/latest cuda/12.6 2>/dev/null || true
  setup_cuda_ld
  echo "BNB_CUDA_VERSION=${BNB_CUDA_VERSION}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
fi

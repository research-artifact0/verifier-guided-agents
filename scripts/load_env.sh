#!/usr/bin/env bash
# Load sal/.env (HF token etc.) for GPU scripts. Safe to source multiple times.
_load_env() {
  local root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local env_file="${root}/.env"

  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi

  # Defaults for UMass scratch layout (overridden by .env when set).
  export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
  export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-126}"
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" 2>/dev/null || true

  if [[ -n "${hf_token:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$hf_token"
  fi
  if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  _load_env "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  echo "HF_HOME: ${HF_HOME:-unset}"
  echo "HF_TOKEN set: $([[ -n ${HF_TOKEN:-} ]] && echo yes || echo no)"
fi

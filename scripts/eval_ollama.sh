#!/usr/bin/env bash
# Table eval via local Ollama (no HF / bitsandbytes).
# Ollama must run on the SAME node as this script (127.0.0.1:11434).
#
#   ./scripts/eval_ollama.sh --table 1 --variant base --episodes 12 \
#     --ollama-model llama3.1:8b --run-dir eval/runs/frontier_llama8b_ollama
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_ROOT="${OLLAMA_ROOT:-/scratch/workspace/eunbiyoon_umass_edu-paper/ollama-local}"
cd "$ROOT"
# shellcheck source=scripts/load_env.sh
source "$ROOT/scripts/load_env.sh"
_load_env "$ROOT"

module load conda/latest 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

_ollama_ok() {
  curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

if ! _ollama_ok; then
  echo "Ollama not up at ${OLLAMA_URL} — starting from ${OLLAMA_ROOT} ..."
  if [[ ! -x "${OLLAMA_ROOT}/scripts/start.sh" ]]; then
    echo "ERROR: missing ${OLLAMA_ROOT}/scripts/start.sh" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${OLLAMA_ROOT}/env.sh"
  "${OLLAMA_ROOT}/scripts/start.sh"
  sleep 2
fi

if ! _ollama_ok; then
  echo "ERROR: Ollama still not reachable at ${OLLAMA_URL}" >&2
  echo "  Eval 노드(예: ials-gpu034)에서 Ollama를 켜야 합니다. login7에서 켠 서버는 GPU 노드에서 안 보입니다." >&2
  echo "  수동: source ${OLLAMA_ROOT}/env.sh && ${OLLAMA_ROOT}/scripts/start.sh" >&2
  echo "  로그: ${OLLAMA_ROOT}/logs/ollama.log" >&2
  tail -15 "${OLLAMA_ROOT}/logs/ollama.log" 2>/dev/null || true
  exit 1
fi

echo "Ollama OK at ${OLLAMA_URL}"
exec python -m eval.run_table --mode ollama "$@"

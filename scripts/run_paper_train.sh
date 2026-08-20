#!/usr/bin/env bash
# Paper LoRA training: Qwen2.5-3B, data/paper/*.jsonl, TensorBoard + checkpoints.
#
# Usage:
#   ./scripts/run_paper_train.sh                  # all 6 trainable variants
#   ./scripts/run_paper_train.sh --variant all
#   ./scripts/run_paper_train.sh --merge-only
#
# Outputs: runs/<session_id>/lora/<variant>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="data/paper"
SESSION_ID="${RUN_ID:-}"
SINGLE_VARIANT=""
MERGE_ONLY=0
SKIP_MERGE=0
RESUME=0
RESUME_CKPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)     SINGLE_VARIANT="$2"; shift 2 ;;
    --merge-only)  MERGE_ONLY=1; shift ;;
    --skip-merge)  SKIP_MERGE=1; shift ;;
    --resume)      RESUME=1; shift ;;
    --resume-from-checkpoint) RESUME_CKPT="$2"; shift 2 ;;
    --data-dir)    DATA_DIR="$2"; shift 2 ;;
    --session)     SESSION_ID="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

setup_env() {
  # shellcheck source=scripts/activate.sh
  source "${ROOT}/scripts/activate.sh"
  export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
  if [[ -z "$SESSION_ID" ]]; then
    SESSION_ID="$(date -u +%Y%m%d_%H%M%S)"
  fi
  export RUN_ID="$SESSION_ID"
  LORA_ROOT="runs/${SESSION_ID}/lora"
  mkdir -p "$LORA_ROOT"
  log "Session: runs/${SESSION_ID}/lora/"
  if ! nvidia-smi -L >/dev/null 2>&1; then
    log "ERROR: no GPU visible — module load cuda/12.6 and use a GPU node (salloc/sbatch)"
    exit 1
  fi
  python -c "import torch; assert torch.cuda.is_available(), 'torch sees no CUDA'"
}

train_variant() {
  local variant="$1"
  local pairs="$2"
  log "=== ${variant}: ${pairs} -> runs/${SESSION_ID}/lora/${variant} ==="
  local -a extra=()
  if [[ -n "$RESUME_CKPT" ]]; then
    extra+=(--resume-from-checkpoint "$RESUME_CKPT")
  elif [[ "$RESUME" -eq 1 ]]; then
    extra+=(--resume)
  fi
  python -m lora.train \
    --paper \
    --tensorboard \
    --pairs "${pairs}" \
    --out "${variant}" \
    --epochs "${TRAIN_EPOCHS}" \
    "${extra[@]}"
}

train_all() {
  if [[ ! -f "${DATA_DIR}/a_beta_all.jsonl" ]]; then
    log "Missing ${DATA_DIR}/ — run: python scripts/prepare_paper_data.py --from-1000"
    exit 1
  fi

  if [[ -n "$SINGLE_VARIANT" ]]; then
    case "$SINGLE_VARIANT" in
      core)       train_variant core       "${DATA_DIR}/a_beta_core.jsonl" ;;
      aux)        train_variant aux        "${DATA_DIR}/a_beta_aux.jsonl" ;;
      all)        train_variant all        "${DATA_DIR}/a_beta_all.jsonl" ;;
      rw)         train_variant rw         "${DATA_DIR}/a_beta_rw.jsonl" ;;
      filter_on)  train_variant filter_on  "${DATA_DIR}/filter_on.jsonl" ;;
      filter_off) train_variant filter_off "${DATA_DIR}/filter_off.jsonl" ;;
      *) echo "Unknown variant: $SINGLE_VARIANT"; exit 1 ;;
    esac
  else
    train_variant filter_on  "${DATA_DIR}/filter_on.jsonl"
    train_variant filter_off "${DATA_DIR}/filter_off.jsonl"
    train_variant core       "${DATA_DIR}/a_beta_core.jsonl"
    train_variant aux        "${DATA_DIR}/a_beta_aux.jsonl"
    train_variant all        "${DATA_DIR}/a_beta_all.jsonl"
    train_variant rw         "${DATA_DIR}/a_beta_rw.jsonl"
  fi

  if [[ "$SKIP_MERGE" -eq 0 && -z "$SINGLE_VARIANT" || "$SINGLE_VARIANT" == "all" || "$SINGLE_VARIANT" == "aux" ]]; then
    if [[ -f "runs/${SESSION_ID}/lora/aux/adapter_config.json" && -f "runs/${SESSION_ID}/lora/all/adapter_config.json" ]]; then
      log "Merging AUX + ALL -> runs/${SESSION_ID}/lora/merge"
      python -m eval.run_table --table 5 --merge --checkpoint-dir "runs/${SESSION_ID}/lora"
    fi
  fi
}

main() {
  setup_env
  if [[ "$MERGE_ONLY" -eq 1 ]]; then
    python -m eval.run_table --table 5 --merge --checkpoint-dir "runs/${SESSION_ID}/lora"
    exit 0
  fi
  train_all
  log "Done. TensorBoard: tensorboard --logdir runs/${SESSION_ID}/lora --port 6006"
}

main "$@"

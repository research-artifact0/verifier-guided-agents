#!/usr/bin/env bash
# Full paper protocol: prepare data -> train 3B LoRA (10 ep) -> merge -> eval Tables 1-7
#
# Usage:
#   ./scripts/run_paper_protocol.sh              # data + train + eval (very long)
#   ./scripts/run_paper_protocol.sh --data-only
#   ./scripts/run_paper_protocol.sh --train-only
#   ./scripts/run_paper_protocol.sh --eval-only --session <id>
#   ./scripts/run_paper_protocol.sh --variant rw
#
# Outputs: runs/<session_id>/lora/<variant>/ and runs/<session_id>/eval/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ONLY=0
TRAIN_ONLY=0
EVAL_ONLY=0
SINGLE_VARIANT=""
EPISODES=12
SESSION_ID="${RUN_ID:-}"
VARIANTS="base,filter_on,filter_off,core,aux,all,rw,merge"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-only)  DATA_ONLY=1; shift ;;
    --train-only) TRAIN_ONLY=1; shift ;;
    --eval-only)  EVAL_ONLY=1; shift ;;
    --variant)    SINGLE_VARIANT="$2"; shift 2 ;;
    --episodes)   EPISODES="$2"; shift 2 ;;
    --session)    SESSION_ID="$2"; shift 2 ;;
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

if [[ "$DATA_ONLY" -eq 1 && "$TRAIN_ONLY" -eq 1 ]]; then
  echo "Use only one of --data-only / --train-only / --eval-only" >&2
  exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

setup_env() {
  log "Activating conda env (sal)..."
  # shellcheck source=scripts/activate.sh
  source "${ROOT}/scripts/activate.sh"
  export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
  export RUN_LABEL="paper-protocol"
  if [[ -z "$SESSION_ID" ]]; then
    if [[ "$EVAL_ONLY" -eq 1 && -f runs/latest.json ]]; then
      SESSION_ID="$(python -c "import json; print(json.load(open('runs/latest.json'))['run_id'])")"
    else
      SESSION_ID="$(date -u +%Y%m%d_%H%M%S)"
    fi
  fi
  export RUN_ID="$SESSION_ID"
  LORA_ROOT="runs/${SESSION_ID}/lora"
  EVAL_ROOT="runs/${SESSION_ID}/eval"
  mkdir -p "$LORA_ROOT" "$EVAL_ROOT"
  log "Session: runs/${SESSION_ID}/"
}

prepare_data() {
  log "Preparing data/paper/*.jsonl from data/*_1000 (paper Table 5 counts)..."
  python scripts/prepare_paper_data.py --from-1000
}

train_variant() {
  local variant="$1"
  local pairs="$2"
  log "Training ${variant} -> runs/${SESSION_ID}/lora/${variant}"
  python -m lora.train \
    --paper \
    --tensorboard \
    --pairs "${pairs}" \
    --out "${variant}" \
    --epochs "${TRAIN_EPOCHS}"
}

train_all() {
  if [[ -n "$SINGLE_VARIANT" ]]; then
    case "$SINGLE_VARIANT" in
      core)  train_variant core  data/paper/a_beta_core.jsonl ;;
      aux)   train_variant aux   data/paper/a_beta_aux.jsonl ;;
      all)   train_variant all   data/paper/a_beta_all.jsonl ;;
      rw)    train_variant rw    data/paper/a_beta_rw.jsonl ;;
      filter_on)  train_variant filter_on  data/paper/filter_on.jsonl ;;
      filter_off) train_variant filter_off data/paper/filter_off.jsonl ;;
      *) echo "Unknown variant: $SINGLE_VARIANT"; exit 1 ;;
    esac
    return
  fi

  train_variant filter_on  data/paper/filter_on.jsonl
  train_variant filter_off data/paper/filter_off.jsonl
  train_variant core       data/paper/a_beta_core.jsonl
  train_variant aux        data/paper/a_beta_aux.jsonl
  train_variant all        data/paper/a_beta_all.jsonl
  train_variant rw         data/paper/a_beta_rw.jsonl

  log "Merging AUX + ALL -> runs/${SESSION_ID}/lora/merge"
  python -m eval.run_table --table 5 --merge --checkpoint-dir "runs/${SESSION_ID}/lora"
}

run_eval() {
  log "Eval Tables 1-7 (paper mode, episodes=${EPISODES})"
  python eval/run_paper_tables.py \
    --paper \
    --episodes "${EPISODES}" \
    --variants "${VARIANTS}" \
    --lora-dir "runs/${SESSION_ID}/lora" \
    --staging-dir "runs/${SESSION_ID}/eval/staging" \
    --run-id "${SESSION_ID}"
  log "Results: runs/${SESSION_ID}/eval/result.md + latex.md"
}

main() {
  setup_env

  if [[ "$DATA_ONLY" -eq 1 ]]; then
    prepare_data
    log "Data prep complete."
    return
  fi

  if [[ "$EVAL_ONLY" -eq 1 ]]; then
    if [[ ! -f "runs/${SESSION_ID}/lora/all/adapter_config.json" ]]; then
      echo "ERROR: no checkpoints under runs/${SESSION_ID}/lora/all/. Run training first." >&2
      exit 1
    fi
    run_eval
    log "Eval complete."
    return
  fi

  if [[ ! -f data/paper/a_beta_all.jsonl ]]; then
    prepare_data
  fi
  train_all

  if [[ "$TRAIN_ONLY" -eq 1 ]]; then
    log "Training complete."
    return
  fi

  if [[ ! -f "runs/${SESSION_ID}/lora/all/adapter_config.json" ]]; then
    echo "ERROR: no checkpoints under runs/${SESSION_ID}/lora/all/. Run training first." >&2
    exit 1
  fi
  run_eval
  log "Paper protocol complete."
}

main "$@"

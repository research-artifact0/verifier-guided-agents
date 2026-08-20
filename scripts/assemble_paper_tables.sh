#!/usr/bin/env bash
# Assemble PAPER.pdf Tables 2-7 from best error-free metrics on disk (no GPU).
#   ./scripts/assemble_paper_tables.sh
#   ./scripts/assemble_paper_tables.sh --out paper_tables_20260630
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python eval/assemble_tables.py \
  --variants base,core,aux,all,rw,merge,filter_on,filter_off \
  "$@"

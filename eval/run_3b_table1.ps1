# Qwen2.5-3B Table 1 smoke: generate -> LoRA -> eval
# Run from project root with venv active.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Stage 1: DPO generate (3B, pd-classic) ==="
python run_pipeline.py generate --config dpo/configs/sarl_3b.yaml --prepare
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Stage 2: LoRA DPO train (3B paper LoRA) ==="
python run_pipeline.py train --paper --pairs dpo/data/a_beta_all.jsonl --out lora_3b/all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Stage 3: Table 1 eval (3B + lora_3b/all) ==="
python run_pipeline.py evaluate --table 1 --paper --variant all --episodes 1 `
  --out results/table1_3b_metrics.json
exit $LASTEXITCODE

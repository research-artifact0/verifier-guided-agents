# SARL — Strategic Agent Learning

This repository trains six Qwen2.5-3B DPO/LoRA variants and evaluates the paper's Tables 1–7. Run every command below from the repository root.

## Requirements

- Linux GPU node with an NVIDIA GPU and CUDA 12.6
- Conda environment named `sal`
- Python 3.10 or newer
- Hugging Face access to `Qwen/Qwen2.5-3B-Instruct`; a token is also required for gated frontier models

On the UMass cluster, create the environment once:

```bash
module load conda/latest cuda/12.6
conda create -n sal python=3.11 -y
conda activate sal
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cp .env.example .env
# Set HF_TOKEN in .env.
```

For each new shell or GPU allocation:

```bash
source scripts/activate.sh
bash scripts/check_gpu_train.sh
```

`scripts/activate.sh` loads `.env`, CUDA 12.6, the `sal` Conda environment, and the CUDA library paths. The GPU check must finish successfully before training or LoRA evaluation.

## Reproduce training

### 1. Prepare the six datasets

The source files must exist under `data/*_1000/`. They are merged, deduplicated, deterministically ordered, and sampled to the paper's Table 5 sizes:

```bash
python scripts/prepare_paper_data.py --from-1000
```

Expected files and pair counts:

| Variant | File | Pairs |
|---|---|---:|
| `filter_on` | `data/paper/filter_on.jsonl` | 388 |
| `filter_off` | `data/paper/filter_off.jsonl` | 407 |
| `core` | `data/paper/a_beta_core.jsonl` | 503 |
| `aux` | `data/paper/a_beta_aux.jsonl` | 613 |
| `all` | `data/paper/a_beta_all.jsonl` | 1,338 |
| `rw` | `data/paper/a_beta_rw.jsonl` | 1,749 |

The command also writes `data/paper/latest_manifest.json`. Preparation fails rather than silently undersampling when a source pool is too small.

### 2. Train

Choose a session ID once so training and evaluation use the same directory:

```bash
export SAL_RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
./scripts/run_paper_train.sh --session "$SAL_RUN_ID"
```

This trains, in order, `filter_on`, `filter_off`, `core`, `aux`, `all`, and `rw`, then creates the `merge` adapter from `aux` and `all`. Each trainable variant uses:

- base model: `Qwen/Qwen2.5-3B-Instruct`
- DPO epochs: 10
- batch size: 1; gradient accumulation: 8
- learning rate: `5e-5`; cosine schedule; warm-up ratio: `0.05`
- DPO beta: `0.1`; maximum sequence length: 12,288
- LoRA rank: 16; alpha: 32; dropout: 0.05; targets: `all-linear`
- 4-bit base-model loading and gradient checkpointing
- checkpoint interval: 20 optimizer steps; retained checkpoints: 100
- TensorBoard logging enabled

Train one variant:

```bash
./scripts/run_paper_train.sh --session "$SAL_RUN_ID" --variant all
```

Resume the latest unfinished checkpoint for that output, or name a checkpoint explicitly:

```bash
./scripts/run_paper_train.sh --session "$SAL_RUN_ID" --variant all --resume
./scripts/run_paper_train.sh --session "$SAL_RUN_ID" --variant all \
  --resume-from-checkpoint "runs/$SAL_RUN_ID/lora/all/adapter/checkpoint-100"
```

Useful alternatives:

```bash
# Do not build the merged adapter.
./scripts/run_paper_train.sh --session "$SAL_RUN_ID" --skip-merge

# Build only the merged adapter after aux and all exist.
./scripts/run_paper_train.sh --session "$SAL_RUN_ID" --merge-only

# Inspect training logs.
tensorboard --logdir "runs/$SAL_RUN_ID/lora" --port 6006
```

## Reproduce evaluation

The full paper evaluation uses 12 episodes per environment, seed 42, and a 192-token generation limit. It evaluates the base model plus the six trained variants and merged adapter:

```bash
export SAL_RUN_ID=<training-session-id>
./scripts/eval_gpu.sh \
  --paper \
  --run-id "$SAL_RUN_ID" \
  --lora-dir "runs/$SAL_RUN_ID/lora" \
  --staging-dir "runs/$SAL_RUN_ID/eval/staging" \
  --variants base,filter_on,filter_off,core,aux,all,rw,merge \
  --episodes 12 \
  --seed 42 \
  --max-tokens 192
```

Evaluation creates symlinks in `staging/`, evaluates Tables 1–7, and writes JSON tables, raw metrics/logs, `result.md`, and `latex.md` under `runs/$SAL_RUN_ID/eval/`. Add `--copy` if the staging directory must contain copies instead of symlinks.

Resume an interrupted evaluation with the same session and arguments:

```bash
./scripts/eval_gpu.sh \
  --paper --resume --run-id "$SAL_RUN_ID" \
  --lora-dir "runs/$SAL_RUN_ID/lora" \
  --variants base,filter_on,filter_off,core,aux,all,rw,merge \
  --episodes 12 --seed 42 --max-tokens 192
```

List adapters detected for a session without loading a model:

```bash
./scripts/eval_gpu.sh --paper --list --run-id "$SAL_RUN_ID" \
  --lora-dir "runs/$SAL_RUN_ID/lora"
```

Run one table/variant directly:

```bash
./scripts/eval_gpu.sh --table 1 --mode lora --variant all \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --checkpoint-dir "runs/$SAL_RUN_ID/lora" \
  --episodes 12 --seed 42 --max-tokens 192 \
  --lora-r 16 --lora-alpha 32 --lora-target all-linear
```

A CPU-only structural smoke test is available; its heuristic scores are not paper results:

```bash
python -m eval.run_paper_tables \
  --mode heuristic --episodes 1 --variants base --run-id smoke
```

For the separate 1,000-episode robustness evaluation, see `eval/EXTENDED_EVAL.md`.

## One-command protocol

The protocol wrapper prepares missing data, trains, merges, and evaluates in one session:

```bash
./scripts/run_paper_protocol.sh --session "$(date -u +%Y%m%d_%H%M%S)"
```

Stage-only modes:

```bash
./scripts/run_paper_protocol.sh --data-only
./scripts/run_paper_protocol.sh --train-only --session <session-id>
./scripts/run_paper_protocol.sh --eval-only --session <session-id>
```

## Output layout

```text
runs/<session-id>/
├── lora/
│   ├── filter_on/        published adapter; checkpoints are under adapter/
│   ├── filter_off/
│   ├── core/
│   ├── aux/
│   ├── all/
│   ├── rw/
│   └── merge/
└── eval/
    ├── staging/          adapter symlinks or copies
    ├── tables/
    ├── metrics/
    ├── logs/
    ├── result.md
    └── latex.md
```

`runs/latest.json` points to the most recently completed training session. Large datasets, model outputs, `.env`, and Python caches are intentionally ignored by Git.

## Troubleshooting

```bash
module load cuda/12.6
source scripts/setup_cuda_ld.sh
setup_cuda_ld
bash scripts/check_gpu_train.sh
```

If model loading reports `AutoProcessor`, Torch, or TorchVision errors, reinstall the matched pair shown in Requirements. If a gated model returns HTTP 401/403, verify `HF_TOKEN` in `.env` and that the account has accepted the model license.
# verifier-guided-agents

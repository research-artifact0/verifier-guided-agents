# SARL: Strategic Agent Learning
## 📖 Overview

SARL trains language-model agents to act strategically across cooperative and competitive games. The repository provides the complete pipeline for preparing preference data, training Qwen2.5-3B LoRA adapters with Direct Preference Optimization (DPO), and reproducing Tables 1–7.

The pipeline includes three stages:

1. **Preference Data Construction:** Merge, deduplicate, and deterministically sample game trajectories into six training datasets.
2. **Strategic Agent Training:** Train six DPO/LoRA variants and construct a merged adapter.
3. **Game Evaluation:** Evaluate in-distribution and held-out strategic games using the paper protocol.

### 🎮 Featured Games

- **Social dilemmas:** Prisoner's Dilemma and Stag Hunt
- **Coordination and competition:** Battle of the Sexes and Matching Pennies
- **Sequential interaction:** Negotiation and Tic-Tac-Toe
- **Held-out games:** Auction, Divide-the-Dollar, Beauty Contest, and one-stage IPD

---

## 🚀 Method

SARL uses preference pairs of the form `prompt / chosen / rejected` to optimize a 4-bit Qwen2.5-3B base model with DPO. The paper protocol trains the following variants:

| Variant | Training pairs | Description |
|---|---:|---|
| `filter_on` | 388 | Filtered preference data |
| `filter_off` | 407 | Unfiltered preference data |
| `core` | 503 | Core A+β subset |
| `aux` | 613 | Auxiliary A+β subset |
| `all` | 1,338 | Full A+β set |
| `rw` | 1,749 | Reward-weighted extension |
| `merge` | — | Equal-weight merge of `aux` and `all` |

---

## 🛠️ Installation

The project requires Python 3.10 or newer, an NVIDIA GPU, and CUDA 12.6. On the UMass cluster, create the environment once:

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

Initialize each new shell or GPU allocation:

```bash
source scripts/activate.sh
bash scripts/check_gpu_train.sh
```

The Hugging Face account must have access to `Qwen/Qwen2.5-3B-Instruct`. Gated frontier-model evaluation also requires the corresponding model license.

---

## ⚡ Training

Use the following commands to reproduce the paper training pipeline.

### 1. Prepare Preference Data

Source trajectories must be available under `data/*_1000/`.

```bash
python scripts/prepare_paper_data.py --from-1000
```

The command writes the six datasets and `latest_manifest.json` to `data/paper/`. It fails if any source pool is too small instead of silently undersampling.

### 2. Train Strategic Agents

Set one session ID and use it for both training and evaluation:

```bash
export SAL_RUN_ID="$(date -u +%Y%m%d_%H%M%S)"

# Train all six variants and merge AUX + ALL.
./scripts/run_paper_train.sh --session "$SAL_RUN_ID"

# Train one variant.
./scripts/run_paper_train.sh \
  --session "$SAL_RUN_ID" \
  --variant all
```

The paper configuration is:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Epochs: 10
- Per-device batch size: 1
- Gradient accumulation: 8
- Learning rate: `5e-5`
- Scheduler and warm-up: cosine, `0.05`
- DPO beta: `0.1`
- Maximum sequence length: 12,288
- LoRA: rank 16, alpha 32, dropout 0.05, `all-linear` targets
- Checkpoints: every 20 optimizer steps, retaining up to 100
- Quantization: 4-bit

### Resume Training

```bash
# Resume the latest unfinished checkpoint for this variant.
./scripts/run_paper_train.sh \
  --session "$SAL_RUN_ID" \
  --variant all \
  --resume

# Resume an explicit checkpoint.
./scripts/run_paper_train.sh \
  --session "$SAL_RUN_ID" \
  --variant all \
  --resume-from-checkpoint \
  "runs/$SAL_RUN_ID/lora/all/adapter/checkpoint-100"
```

### Monitoring

Track training with TensorBoard:

```bash
tensorboard --logdir "runs/$SAL_RUN_ID/lora" --port 6006
```

---

## 🧪 Evaluation

### 1. Full Paper Evaluation

Evaluate the base model, six trained variants, and merged adapter over Tables 1–7:

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

The paper evaluation uses 12 episodes per environment, random seed 42, and a 192-token generation limit. Adapter symlinks are assembled under `staging/`; pass `--copy` when physical copies are required.

### 2. Resume Evaluation

Use the identical session ID and evaluation arguments:

```bash
./scripts/eval_gpu.sh \
  --paper \
  --resume \
  --run-id "$SAL_RUN_ID" \
  --lora-dir "runs/$SAL_RUN_ID/lora" \
  --variants base,filter_on,filter_off,core,aux,all,rw,merge \
  --episodes 12 \
  --seed 42 \
  --max-tokens 192
```

### 3. Single-Table Evaluation

```bash
./scripts/eval_gpu.sh \
  --table 1 \
  --mode lora \
  --variant all \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --checkpoint-dir "runs/$SAL_RUN_ID/lora" \
  --episodes 12 \
  --seed 42 \
  --max-tokens 192 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-target all-linear
```

### 4. Smoke Test

The heuristic mode checks the evaluation pipeline without loading a GPU model. Its scores are not paper results.

```bash
python -m eval.run_paper_tables \
  --mode heuristic \
  --episodes 1 \
  --variants base \
  --run-id smoke
```

For the separate 1,000-episode robustness protocol, see [Extended Evaluation](eval/EXTENDED_EVAL.md).

---

## 🔄 End-to-End Pipeline

Run data preparation, training, merging, and evaluation in one session:

```bash
./scripts/run_paper_protocol.sh \
  --session "$(date -u +%Y%m%d_%H%M%S)"
```

Individual stages can also be run independently:

```bash
# Data preparation only
./scripts/run_paper_protocol.sh --data-only

# Training only
./scripts/run_paper_protocol.sh --train-only --session <session-id>

# Evaluation only
./scripts/run_paper_protocol.sh --eval-only --session <session-id>
```

---

## 📊 Outputs

All artifacts from one experiment share a session directory:

```text
runs/<session-id>/
├── lora/
│   ├── filter_on/
│   ├── filter_off/
│   ├── core/
│   ├── aux/
│   ├── all/
│   ├── rw/
│   └── merge/
└── eval/
    ├── staging/
    ├── tables/
    ├── metrics/
    ├── logs/
    ├── result.md
    └── latex.md
```

Each trainable variant stores its resumable checkpoints under `adapter/checkpoint-*`. `runs/latest.json` points to the most recently completed training session.

---

## 🔧 Troubleshooting

Verify the CUDA environment and GPU visibility:

```bash
module load cuda/12.6
source scripts/setup_cuda_ld.sh
setup_cuda_ld
bash scripts/check_gpu_train.sh
```

- For `AutoProcessor`, Torch, or TorchVision import errors, reinstall the matched Torch/TorchVision versions listed above.
- For Hugging Face HTTP 401/403 errors, verify `HF_TOKEN` in `.env` and confirm model-license access.

# Method notes — length-normalized DPO & Elo-ranking pairs

Auto-generated from local training manifests and JSONL pair files.

## Training objective (length-normalized DPO)

- Base model (paper): `Qwen/Qwen2.5-3B-Instruct`
- DPO temperature β = `0.1`
- LoRA: r=16, α=32, target=`all-linear`
- Max sequence length: `12288` tokens (paper)
- Local max sequence length: `8192` tokens

**Length-normalized DPO.** Preference loss divides each sequence log-likelihood
by its token length before forming the Bradley–Terry margin, so longer
chain-of-thought traces are not automatically preferred. In code this is
the standard TRL `DPOTrainer` objective with per-example `max_length`
truncation (`lora/train.py`). Report mean chosen vs rejected lengths below
to justify the normalization.

## Pair corpora by variant

| Variant | pairs (paper ref) | pairs (local file) | chosen tok (mean) | rejected tok (mean) |
|---------|-------------------|--------------------|-------------------|---------------------|
| filter_on | 388 | 388 | 1042.1 | 628.9 |
| filter_off | 407 | 407 | 976.1 | 615.4 |
| core | 503 | 503 | 1044.3 | 643.0 |
| aux | 613 | 613 | 1089.4 | 668.6 |
| all | 1338 | 1338 | 1057.6 | 647.7 |
| rw | 1749 | 1749 | 1156.2 | 705.7 |

## Elo-ranking pairs (RW variant)

- `a_beta_all.jsonl`: 5384 pairs
- `a_beta_rw.jsonl`: 10768 pairs (2.00× ALL)
- Mean reward margin (chosen − rejected): 2.004

**How RW extends ALL.** RW keeps the oracle-vs-blind preference pairs from ALL
and adds reversed / additional comparisons so each prompt can contribute
multiple ranked outcomes. Pairs are filtered with the same A+β manifest
(core/aux splits) but the RW export doubles coverage for length-normalized
DPO training (`config.TRAINING_VARIANTS['rw']`, paper: 1749 pairs).

> Note: `dpo/` rollout code is not in this checkout. Regenerate Elo-ranked
> trajectories with `python run_pipeline.py generate` once `dpo/` is restored.

### Opponent mix in RW file (top)

- always_defect: 1550
- epsilon_greedy: 1474
- random: 1388
- always_cooperate: 1382
- random_bargainer: 1044
- grim_trigger: 886
- equal_split_bargainer: 768
- tit_for_tat: 718

## Suggested paper sentences (draft)

1. *Length-normalized DPO.* We optimize a length-normalized DPO objective
   (β=0.1) on Qwen2.5-3B with LoRA adapters, truncating
   completions to 12288 tokens and normalizing log-probabilities
   by response length to reduce verbosity bias.

2. *Elo-ranking pairs.* Starting from blind (3B) and oracle (7B) rollouts,
   we rank trajectories by cumulative reward and emit additional preference
   pairs for the RW corpus (≈1.3× more pairs than ALL), keeping the same
   prompt distribution while enriching hard comparisons.

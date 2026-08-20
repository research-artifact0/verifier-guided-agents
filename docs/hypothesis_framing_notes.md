# Framing notes — de-emphasize opponent-prediction hypothesis

Guidance for revising Intro / Method / Discussion. Auto-generated from paper reference tables and pair-file structure.

## What to remove or soften

- Claims that the **main contribution** is teaching models to **predict opponents inside trajectories**.
- Framing Hypothesis B as a **prospective** core hypothesis rather than a 
retrospective ablation.
- Language implying `[Prior]` / `[Update]` blocks are learned **belief-state tracking** 
rather than an optional reasoning scaffold copied from oracle traces.

## What to keep

- **A+β**: blind vs oracle preference learning on strategic games.
- Oracle traces provide **higher-reward decision patterns** via solver-pinned actions.
- Hypothesis B (filter-on vs filter-off) as a **secondary ablation**: does requiring 
the reasoning scaffold to align with the solver action help?

## Evidence for reframing

| Item | filter-on | filter-off |
|------|-----------|------------|
| Pairs (local) | 5384 | 4587 |
| `[Prior]` rate | 92.8% | 92.0% |
| Paper fc(HO) | 0.36 | 0.71 |
| Paper CR(HO) | -0.95 | -0.23 |

Filter-off achieves **higher** fc(HO) than filter-on (0.71 vs 0.36) while CR is similar, 
so opponent-prediction scaffolding is **not** the primary driver of headline gains.

## Suggested replacement framing

1. *Primary claim.* Length-normalized DPO on oracle–blind preference pairs improves 
strategic play in matrix and repeated games by imitating solver-backed reasoning traces.

2. *Reasoning format.* The `[Prior]`/`[Update]`/`[EV]` template structures oracle 
outputs; we do **not** claim the fine-tuned model maintains calibrated beliefs—only 
that the format correlates with better actions in training data.

3. *Hypothesis B (one sentence).* We retrospectively tested whether filtering pairs 
to those whose chain-of-thought already agrees with the solver action changes 
metrics; this ablation did not outperform the unfiltered corpus on headline CR.

## Lines to delete / rewrite (checklist)

- [ ] Intro: any sentence starting with “We hypothesize that models learn to predict…”
- [ ] Method: move opponent-prediction from **contribution bullet** to **oracle trace format** footnote
- [ ] Discussion: drop Hypothesis B as “future work”; keep as past ablation only
- [ ] `how_to_run.md`: rename “Hypothesis B” comment to “filter-on / filter-off ablation”

# Method notes — best-response action selection (oracle solver)

Draft for Method §3 (oracle trajectory generation). The rollout solver lives in `dpo/` (not in this checkout); details below are reconstructed from pair files and eval code.

## Overview

Each training prompt is a game state. We run two policies:

- **Blind** (3B): free-form chain-of-thought → action.
- **Oracle** (7B): same scaffold, but the final action is **pinned** to the 
  game solver's best response (BR) under the rollout opponent model.

Preference pairs prefer oracle when cumulative reward exceeds blind.

## Solver procedure (per decision point)

1. **Enumerate legal actions** from the environment spec (e.g. `C`/`D`, 125 negotiation proposals, bid grid).

2. **Instantiate opponent model.** For matrix games, use the scripted opponent 
in the rollout (TFT, grim trigger, …) or a uniform/mixed prior for one-shot games. 
For negotiation, opponent type is one of `equal_split_bargainer`, 
`random_bargainer`, `nash_bargainer` (see rejected-side traces in pair files).

3. **Score each candidate action.** For 2×2 stage games, compute expected payoff 
under the opponent model (Eq. 1 in the paper). For repeated games with a known 
opponent policy, BR is the action that maximizes stage payoff given the 
opponent's realized or predicted move (see Table 6 stag-hunt TFT example in 
`eval/run_table.py`: oracle plays the payoff-maximizing response to TFT's last action).

4. **Select BR.** `a* = argmax_a EU(a | opponent_model, history)`. 
Ties broken deterministically (first legal action in list order).

5. **Pin action in oracle trace.** The 7B model writes `[Prior]` / `[Update]` / 
`[EV]` reasoning; post-processing replaces `<action>…</action>` with the solver 
output if the model disagrees. Filter-on pairs (`filter_on.jsonl`) keep only 
examples where the reasoning **already** ends at the solver action (69 explicit “solver-pinned” mentions in 5384 filter-on pairs).

## Implementation anchors in this repo

| Component | Location | Role |
|-----------|----------|------|
| 2×2 exploitability / opponent BR | `eval/exploitability.py` | `br_opp = argmax_{a1} u_opp(a0,a1)` |
| Stag-hunt oracle demo | `eval/run_table.py` Table 6 | BR vs realized TFT action |
| Negotiation payoff | `env/games.py` | feasibility check + weighted sum |
| Action parsing | `env/agents.py` | `<action>` extraction, negotiation `[a,b,c]` |

## Scaffold statistics

- ALL pairs with `[Prior]` block: **92.8%** (5384 total pairs)
- Filter-on pairs with `[Prior]`: **92.8%**

## Suggested Method sentences (draft)

1. *Oracle solver.* At each decision point we compute a best response with a 
lightweight game solver: legal actions are scored by expected utility under the 
rollout opponent model (scripted policy or bargainer type), and the maximizing 
action is written into the oracle trajectory. The 7B oracle model produces 
chain-of-thought text; if its emitted action disagrees with the solver, we 
overwrite the `<action>` tag with the solver choice before building preference pairs.

2. *Repeated games.* When the opponent policy is deterministic given history 
(e.g. Tit-for-Tat), the BR step reduces to maximizing the stage payoff against 
the opponent's next action implied by their strategy and the observed history.

3. *Negotiation.* BR is computed against the sampled bargainer type: e.g. 
equal-split plays `(2,2,2)`, Nash-type plays max-weight issue with complementary 
slack for the opponent. Feasibility (`a_i + a'_i ≤ 4`) is enforced before scoring.

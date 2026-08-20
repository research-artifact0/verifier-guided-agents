# Negotiation analysis — why training does not beat base

Auto-generated from `config.PAPER_TABLE2`, training pairs, and eval code.

## Observed results (paper Table 2)

| Variant | negotiation CR | vs base |
|---------|----------------|---------|
| base | 5.75 | — |
| core | 3.17 | -2.58 |
| aux | 3.46 | -2.29 |
| all | 3.08 | -2.67 |
| rw | 4.67 | -1.08 |
| merge | 4.04 | -1.71 |

Base CR = **5.75**; best trained variant (RW) = **4.67** — still below base. ALL drops to **3.08**.

## Training-pair statistics (`a_beta_all.jsonl`, negotiation only)

- Negotiation pairs: **1137**
- Mean chosen reward (oracle): **4.90**
- Mean rejected reward (blind): **4.04**
- Zero-reward chosen rate: **66.8%** (capacity violation → both players get 0)
- Oracle/blind **tie** rate (same outcome): **62.2%** → weak preference signal for DPO
- Pair outcomes: {'ties': 707, 'oracle_wins': 257, 'blind_wins': 173}

**Oracle vs blind action profiles**

| Source | Top proposals |
|--------|---------------|
| Oracle (chosen) | `[2,2,2]` (162), `[1,1,1]` (64), `[2,2,0]` (64) |
| Blind (rejected) | `[4,0,0]` (223), `[2,2,0]` (220), `[1,1,1]` (131) |

Oracle rollouts favor **conservative, coordinating** splits (e.g. `[2,2,2]`), while blind rollouts favor **greedy** claims (e.g. `[4,0,0]`) that often violate capacity when paired with a random opponent.

## Mechanistic explanations (for Results / Discussion)

1. **Sparse, discontinuous reward.** Negotiation pays 0 unless every issue satisfies `claim_self + claim_opp ≤ 4`. With a random opponent, ~75–80% of proposal pairs violate capacity, so reward is mostly zero and gradients are noisy.

2. **Weak preference pairs.** A large share of negotiation DPO pairs are ties (oracle and blind reach the same cumulative reward). DPO cannot push the policy toward higher negotiation payoff when chosen ≈ rejected.

3. **Large action space + format burden.** Agents must emit a valid `[a,b,c]` tuple from 125 proposals inside `<action>…</action>`. PD/BoS gains dominate training signal; negotiation coordination is under-represented and harder to imitate.

4. **Eval opponent mismatch.** Rollout opponents for negotiation include typed bargainers (`equal_split`, `random_bargainer`, `nash_bargainer`), but eval uses `random` over 125 proposals (`env/games.py`). Policies tuned to oracle bargainer types do not necessarily maximize CR against uniform random proposals.

5. **No positive transfer on aggregate CR.** Even when oracle traces show higher per-episode reward, length-normalized DPO on mixed-env corpora does not move negotiation CR above the base 3B model; other games (PD, auction) absorb most of the capacity gain.

## Suggested Discussion paragraph (draft)

*Negotiation underperformance.* Table 2 shows negotiation is the only ID game where A+β training **does not** improve over the base model (base CR 5.75 vs. ALL 3.08). We attribute this to (i) all-or-nothing payoffs that yield zero reward in most random pairings, (ii) a high rate of tied oracle–blind preference pairs that provide little learning signal, and (iii) a 125-action output space that is harder to master via imitation than 2-action matrix games. Future work could oversample feasible agreements, use dense shaping rewards, or eval against the same bargainer taxonomy used in rollouts.

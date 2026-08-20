# Table 6 — Per-(model, env) final-round coupling rate (paper_v01.pdf)

Source run: `eval/runs/paper_tables_assembled`  
Model: `Qwen/Qwen2.5-3B-Instruct` · train_ep=1 · eval_ep=12 · LoRA r=16 α=32

**paper_v01.pdf Table 6** (p.13): final-round coupling $f_c$ per model × env, parsable rounds only ($n=12$ episodes/cell).

> Repo `tables/table6.json` is **Appendix Table 8** (stag-hunt anti-coordination example). paper_v01 numbering differs.

## Your eval (local $f_c$)

Coupling was **not scored** in your runs: every variant has `per_game_fc = null` (0 parsable `[EV]` rounds in model CoT). Eval logs store actions only, not reasoning text.

| Model | pd-c | pd-t | pd-h | stag | nego | bos | mp | ttt | auct | dd | p-b | ipd |
|-------|-----:|-----:|-----:|-----:|-----:|----:|---:|----:|-----:|---:|----:|----:|
| Bedrock Haiku 4.5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Llama 3.1–8B (B) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Gemma 3–27B | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Llama 3.3–70B | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Qwen 2.5–3B base | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Qwen 2.5–3B + A+β-on | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Qwen 2.5–3B + A+β-off | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## paper_v01.pdf reference (target values)

| Model | pd-c | pd-t | pd-h | stag | nego | bos | mp | ttt | auct | dd | p-b | ipd |
|-------|-----:|-----:|-----:|-----:|-----:|----:|---:|----:|-----:|---:|----:|----:|
| Bedrock Haiku 4.5 | 0.87 | 0.90 | 0.91 | 0.89 | 0.00 | 1.00 | 1.00 | n/a | 0.00 | n/a | 0.00 | 0.92 |
| Llama 3.1–8B (B) | 0.90 | 0.43 | 1.00 | 0.43 | 0.00 | 1.00 | 0.71 | 0.00 | n/a | 0.00 | n/a | 0.67 |
| Gemma 3–27B | 0.38 | 0.75 | 0.83 | 0.91 | 0.00 | 1.00 | 0.91 | 0.00 | n/a | 0.00 | n/a | 1.00 |
| Llama 3.3–70B | 0.60 | 0.33 | 0.80 | 0.50 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 |
| Qwen 2.5–3B base | 1.00 | 0.80 | 0.83 | 0.75 | n/a | 0.78 | 0.75 | 0.00 | n/a | 0.00 | 0.00 | 1.00 |
| Qwen 2.5–3B + A+β-on | 0.83 | 0.33 | 0.50 | 1.00 | 0.00 | 1.00 | 0.60 | n/a | 0.00 | 0.00 | 0.00 | 0.67 |
| Qwen 2.5–3B + A+β-off | 0.25 | 0.33 | 0.33 | 0.00 | 0.00 | 0.88 | 0.88 | 0.00 | n/a | n/a | 0.00 | 0.83 |

## LaTeX (paper_v01.pdf caption)

Local table (your run — fill after coupling re-score):

```latex
\begin{table}[t]
\centering
\caption{Per-(model, env) final-round coupling rate, parsable rounds only. $n=12$ episodes per cell.}
\label{tab:coupling-by-game}
\small
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
Model & pd-c & pd-t & pd-h & stag & nego & bos & mp & ttt & auct & dd & p-b & ipd \\
\midrule
Bedrock Haiku 4.5 & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a \\
Llama 3.1--8B (B) & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a \\
Gemma 3--27B & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a \\
Llama 3.3--70B & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a \\
Qwen 2.5--3B base & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a \\
Qwen 2.5--3B + A+$\\beta$-on & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a \\
Qwen 2.5--3B + A+$\\beta$-off & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a & n/a \\
\bottomrule
\end{tabular}
\end{table}
```

Structured JSON: `eval/runs/paper_tables_assembled/tables/table6_paper_format.json`

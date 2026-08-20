# Table 4 — Frontier benchmark (CR + TFT-CR)

Source run: `eval/runs/paper_tables_assembled`  
Tag: `Qwen2.5-3B · train_ep=1 · eval_ep=12 · LoRA r=16 α=32`

**paper_v01.pdf Table 4** — 동일 12-env eval, **로컬 rollout만** (paper 참조값 없음).

| Model | CR(ID) | CR(HO) | TFT-CR |
|-------|-------:|-------:|-------:|
| Claude 3.5 Haiku | — | — | — |
| Gemma 3 27B | +14.85 | +43.60 | +29.31 |
| Llama 3.1 70B | — | — | +30.93 |
| Llama 3.1 8B | +16.54 | +39.48 | +25.85 |
| Qwen 2.5-3B base | +12.07 | +49.49 | +31.50 |

## LaTeX

```latex
\begin{table}[t]
\centering
\caption{Frontier benchmark: CR(ID), CR(HO), and mean CR vs.\ Tit-for-Tat (TFT-CR). Local eval only.}
\label{tab:frontier-cr}
\begin{tabular}{lrrr}
\toprule
Model & CR(ID) & CR(HO) & TFT-CR \\
\midrule
Claude 3.5 Haiku & --- & --- & --- \\
Gemma 3 27B & +14.85 & +43.60 & +29.31 \\
Llama 3.1 70B & --- & --- & +30.93 \\
Llama 3.1 8B & +16.54 & +39.48 & +25.85 \\
Qwen 2.5-3B base & +12.07 & +49.49 & +31.50 \\
\bottomrule
\end{tabular}
\end{table}
```

## Sources

- `gemma_27b`: merged frontier_gemma27b logs (near-complete); TFT-CR from 16 TFT episodes
- `llama_70b`: TFT-CR only from 15 TFT episodes (full CR not yet merged)
- `llama_8b`: `frontier_llama8b/tables/table1_base.json`; TFT-CR from 17 TFT episodes
- `qwen_3b_base`: `20260630_180322` base metrics; TFT-CR from 10 TFT episodes
- `haiku`: not evaluated locally

JSON: `eval/runs/paper_tables_assembled/tables/table4.json`

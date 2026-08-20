# Table 3 — HO Strategic (per-variant CR)

Source run: `eval/runs/final(includefilderoffandon)\`

| Variant | HO Strategic CR |
|---------|-----------------|
| base | +65.46 |
| core | +68.96 |
| aux | +68.96 |
| all | +68.96 |
| rw | +71.68 |
| merge | +68.96 |

## LaTeX row snippet

HO Strategic & +65.46 & +68.96 & +68.96 & +68.96 & \textbf{+71.68} & +68.96 \\

## Paper reference (`config.PAPER_TABLE3`)

**HO Strategic is missing from `config.PAPER_TABLE3`.** Add the row below to `config.py` after verifying paper-scale (3B) numbers:

```python
    "HO Strategic": {"base": 65.46, "core": 68.96, "aux": 68.96, "all": 68.96, "rw": 71.68, "merge": 68.96},
```

> For paper.pdf Table 3, re-run with `./scripts/run_paper_protocol.sh --eval-only` once `runs/<session>/lora/` exists.

"""
Reproduce SARL paper Tables 1–3 (LoRA-based evaluation).

Usage:
  python table_results.py --mode heuristic          # fast, no GPU model load
  python table_results.py --mode lora               # Qwen2.5-0.5B + LoRA (same as lora.py)
  python table_results.py --mode lora --smoke       # run lora backward check first
  python table_results.py --compare-paper           # print paper reference rows
  python table_results.py --episodes 2              # quick debug (paper uses 12)

Paper training uses Qwen2.5-3B LoRA r=16 alpha=32 DPO; this script defaults to the
working local setup from lora.py (0.5B 4-bit, r=8, alpha=16).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.agents import HeuristicAgent, LoRALLMAgent
from config import (
    ALL_GAMES,
    HO_GAMES,
    ID_GAMES,
    PAPER_TABLE1,
    PAPER_TABLE2,
    PAPER_TABLE3,
)
from eval.metrics import EvalMetrics, evaluate_agent
from lora.utils import attach_lora, load_base_model, load_lora_adapter, smoke_test_backward


def fmt(x: float) -> str:
    if x >= 0:
        return f"+{x:.2f}" if abs(x) < 100 else f"{x:.2f}"
    return f"{x:.2f}"


def print_table1(rows: dict[str, EvalMetrics], label_map: dict[str, str] | None = None) -> None:
    label_map = label_map or {}
    print("\n=== Table 1: Headline results ===")
    print(f"{'Model':<28} {'fc(ID)':>8} {'fc(HO)':>8} {'CR(ID)':>10} {'CR(HO)':>10}")
    print("-" * 68)
    for name, m in rows.items():
        display = label_map.get(name, name)
        print(
            f"{display:<28} {m.fc_id:>8.2f} {m.fc_ho:>8.2f} "
            f"{fmt(m.cr_id):>10} {fmt(m.cr_ho):>10}"
        )


def print_table2(rows: dict[str, EvalMetrics], variant: str = "base") -> None:
    print(f"\n=== Table 2: Per-game CR (variant={variant}) ===")
    header = f"{'Game':<22} {'CR':>8}"
    if variant in PAPER_TABLE2["pd-classic"]:
        header += f" {'paper':>8}"
    print(header)
    print("-" * 40)
    m = rows[variant]
    for game in ALL_GAMES:
        cr = m.per_game_cr.get(game, 0.0)
        line = f"{game:<22} {cr:>8.2f}"
        if game in PAPER_TABLE2 and variant in PAPER_TABLE2[game]:
            line += f" {PAPER_TABLE2[game][variant]:>8.2f}"
        print(line)


def print_table3(rows: dict[str, EvalMetrics], variant: str = "base") -> None:
    print(f"\n=== Table 3: Per-opponent-axis CR (variant={variant}) ===")
    print(f"{'Set Axis':<18} {'CR':>8}")
    print("-" * 28)
    m = rows[variant]
    for key in sorted(m.per_axis_cr):
        print(f"{key:<18} {m.per_axis_cr[key]:>8.2f}")


def print_paper_reference() -> None:
    print("\n=== Paper reference (Qwen2.5-3B + DPO LoRA) ===")
    print_table1(
        {
            k: EvalMetrics(
                fc_id=v["fc_id"],
                fc_ho=v["fc_ho"],
                cr_id=v["cr_id"],
                cr_ho=v["cr_ho"],
            )
            for k, v in PAPER_TABLE1.items()
        },
        {
            "base": "Qwen 2.5-3B base",
            "rw": "Qwen 2.5-3B + A+β-RW",
            "merge": "Qwen 2.5-3B + A+β-MERGE",
            "haiku": "Bedrock Haiku 4.5",
        },
    )


def build_agent(args):
    if args.mode == "heuristic":
        return HeuristicAgent(), "heuristic-base"

    model, tokenizer = load_base_model(model_id=args.model_id, use_4bit=not args.no_4bit)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        tag = Path(args.adapter).name
    else:
        model = attach_lora(model)
        tag = "lora-untrained"

    if args.smoke:
        loss = smoke_test_backward(model, tokenizer)
        vram = model.get_base_model().model.embed_tokens.weight.device
        print(f"LoRA smoke: loss={loss:.4f} (lora.py baseline ~4.49)")

    return LoRALLMAgent(model, tokenizer, max_new_tokens=args.max_tokens), tag


def main() -> None:
    parser = argparse.ArgumentParser(description="SARL Table 1-3 evaluation (LoRA)")
    parser.add_argument("--mode", choices=["heuristic", "lora"], default="heuristic")
    parser.add_argument("--model-id", default=None, help="HF model id (default: config.MODEL_ID)")
    parser.add_argument("--adapter", default=None, help="Path to trained LoRA adapter")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run lora.py-style backward check")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--compare-paper", action="store_true")
    parser.add_argument("--out", default="results/table_metrics.json")
    args = parser.parse_args()

    if args.compare_paper:
        print_paper_reference()
        return

    agent, tag = build_agent(args)
    print(f"Evaluating variant={tag} mode={args.mode} episodes/env={args.episodes}")

    metrics = evaluate_agent(agent, n_episodes=args.episodes, seed_base=args.seed)
    rows = {tag: metrics}

    print_table1(rows, {tag: f"local ({tag})"})
    print_table2(rows, tag)
    print_table3(rows, tag)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        tag: {
            "fc_id": metrics.fc_id,
            "fc_ho": metrics.fc_ho,
            "cr_id": metrics.cr_id,
            "cr_ho": metrics.cr_ho,
            "per_game_cr": metrics.per_game_cr,
            "per_axis_cr": metrics.per_axis_cr,
        }
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {out}")

    if args.mode == "lora" and not args.adapter:
        print(
            "\nNote: untrained LoRA ~= base model. For paper Table 1 rows (RW/MERGE), "
            "train DPO adapters then run with --adapter lora/aux and merge via lora_utils.merge_lora_adapters."
        )


if __name__ == "__main__":
    main()

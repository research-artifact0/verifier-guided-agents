"""SARL paper full pipeline: Generate DPO pairs -> LoRA train -> Evaluate.

  # DPO generate (Ollama qwen2.5:3b blind + qwen2.5:7b oracle)
  python run_pipeline.py smoke --prepare    # PD×3, default 16 ep
  python run_pipeline.py paper --prepare    # PD×3, default 1000 ep
  python run_pipeline.py generate --config dpo/configs/pd.yaml --n-episodes 64 --prepare

  python run_pipeline.py train --paper --pairs dpo/data/a_beta_all.jsonl --out lora_3b/all
  python run_pipeline.py evaluate --table 1 --paper --variant all --episodes 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def _py(*args: str) -> int:
    cmd = [sys.executable, *args]
    print("$", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("BNB_CUDA_VERSION", "126")
    return subprocess.call(cmd, cwd=ROOT, env=env)


def _append_paper_train_args(argv: list[str], ns: argparse.Namespace) -> None:
    if getattr(ns, "paper", False):
        argv.append("--paper")
    if getattr(ns, "model_id", None):
        argv.extend(["--model-id", ns.model_id])
    if getattr(ns, "lora_r", None) is not None:
        argv.extend(["--lora-r", str(ns.lora_r)])
    if getattr(ns, "lora_alpha", None) is not None:
        argv.extend(["--lora-alpha", str(ns.lora_alpha)])
    if getattr(ns, "lora_target", None):
        argv.extend(["--lora-target", ns.lora_target])


def _append_paper_eval_args(argv: list[str], ns: argparse.Namespace) -> None:
    if getattr(ns, "checkpoint_dir", None):
        argv.extend(["--checkpoint-dir", str(ns.checkpoint_dir)])
    elif getattr(ns, "paper", False):
        argv.extend(["--checkpoint-dir", str(config.LORA_3B_DIR)])
    if getattr(ns, "paper", False):
        if not getattr(ns, "model_id", None):
            argv.extend(["--model-id", config.PAPER_MODEL_ID])
        if getattr(ns, "lora_r", None) is None:
            argv.extend(["--lora-r", str(config.PAPER_LORA_R)])
        if getattr(ns, "lora_alpha", None) is None:
            argv.extend(["--lora-alpha", str(config.PAPER_LORA_ALPHA)])
        if not getattr(ns, "lora_target", None):
            argv.extend(["--lora-target", config.PAPER_LORA_TARGET_MODULES])
    else:
        if getattr(ns, "model_id", None):
            argv.extend(["--model-id", ns.model_id])
        if getattr(ns, "lora_r", None) is not None:
            argv.extend(["--lora-r", str(ns.lora_r)])
        if getattr(ns, "lora_alpha", None) is not None:
            argv.extend(["--lora-alpha", str(ns.lora_alpha)])
        if getattr(ns, "lora_target", None):
            argv.extend(["--lora-target", ns.lora_target])


def cmd_generate(ns: argparse.Namespace) -> int:
    argv = ["-m", "dpo.generate"]
    if ns.config:
        argv.extend(["--config", str(ns.config)])
    if ns.suite:
        argv.extend(["--suite", str(ns.suite)])
    if ns.prepare:
        argv.append("--prepare")
    if ns.pair_only:
        argv.append("--pair-only")
    if ns.max_pairs is not None:
        argv.extend(["--max-pairs", str(ns.max_pairs)])
    if getattr(ns, "n_episodes", None) is not None:
        argv.extend(["--n-episodes", str(ns.n_episodes)])
    else:
        argv.extend(["--n-episodes", str(config.DEFAULT_N_EPISODES)])
    if ns.quiet:
        argv.append("--quiet")
    return _py(*argv)


def cmd_train(ns: argparse.Namespace) -> int:
    if ns.train_all:
        argv = ["-m", "eval.run_table", "--table", "5", "--train"]
        _append_paper_eval_args(argv, ns)
        return _py(*argv)
    argv = ["-m", "lora.train"]
    if ns.pairs:
        argv.extend(["--pairs", str(ns.pairs)])
    if ns.out:
        argv.extend(["--out", str(ns.out)])
    if ns.smoke_only:
        argv.append("--smoke-only")
    if ns.epochs is not None:
        argv.extend(["--epochs", str(ns.epochs)])
    if getattr(ns, "max_length", None) is not None:
        argv.extend(["--max-length", str(ns.max_length)])
    if getattr(ns, "grad_accum", None) is not None:
        argv.extend(["--grad-accum", str(ns.grad_accum)])
    _append_paper_train_args(argv, ns)
    return _py(*argv)


def cmd_evaluate(ns: argparse.Namespace) -> int:
    argv = ["-m", "eval.run_table", "--table", str(ns.table)]
    if ns.variant:
        argv.extend(["--variant", ns.variant])
    if ns.compare_paper:
        argv.append("--compare-paper")
    if ns.mode:
        argv.extend(["--mode", ns.mode])
    if ns.episodes is not None:
        argv.extend(["--episodes", str(ns.episodes)])
    if getattr(ns, "out", None):
        argv.extend(["--out", str(ns.out)])
    if getattr(ns, "run_dir", None):
        argv.extend(["--run-dir", str(ns.run_dir)])
    if getattr(ns, "log_file", None):
        argv.extend(["--log-file", str(ns.log_file)])
    if getattr(ns, "quiet", False):
        argv.append("--quiet")
    _append_paper_eval_args(argv, ns)
    return _py(*argv)


def cmd_all(ns: argparse.Namespace) -> int:
    rc = cmd_generate(ns)
    if rc != 0:
        return rc
    rc = cmd_train(argparse.Namespace(
        train_all=True, pairs=None, out=None, smoke_only=False, epochs=None,
        paper=getattr(ns, "paper", False), model_id=None, checkpoint_dir=None,
        lora_r=None, lora_alpha=None, lora_target=None,
    ))
    if rc != 0:
        return rc
    ns.table = 1
    return cmd_evaluate(ns)


def _generate_ns(config_path: Path, ns: argparse.Namespace, *, default_episodes: int) -> argparse.Namespace:
    n_episodes = getattr(ns, "n_episodes", None)
    if n_episodes is None:
        n_episodes = default_episodes
    return argparse.Namespace(
        config=config_path,
        suite=None,
        prepare=not getattr(ns, "no_prepare", False),
        pair_only=getattr(ns, "pair_only", False),
        max_pairs=getattr(ns, "max_pairs", None),
        n_episodes=n_episodes,
        quiet=getattr(ns, "quiet", False),
    )


def cmd_smoke(ns: argparse.Namespace) -> int:
    return cmd_generate(_generate_ns(config.DPO_CONFIG_PD, ns, default_episodes=config.DEFAULT_N_EPISODES))


def cmd_paper(ns: argparse.Namespace) -> int:
    return cmd_generate(_generate_ns(config.DPO_CONFIG_PD, ns, default_episodes=config.PAPER_N_EPISODES))


def cmd_all_3b(ns: argparse.Namespace) -> int:
    gen_ns = argparse.Namespace(
        no_prepare=False,
        pair_only=False,
        max_pairs=None,
        n_episodes=getattr(ns, "n_episodes", None) or config.DEFAULT_N_EPISODES,
        quiet=getattr(ns, "quiet", False),
    )
    rc = cmd_generate(_generate_ns(config.DPO_CONFIG_PD, gen_ns, default_episodes=config.DEFAULT_N_EPISODES))
    if rc != 0:
        return rc
    rc = cmd_train(argparse.Namespace(
        train_all=False,
        pairs=config.DPO_DATA_DIR / "a_beta_all.jsonl",
        out=config.LORA_3B_DIR / "all",
        smoke_only=False,
        epochs=ns.epochs,
        paper=True,
        model_id=None,
        lora_r=None,
        lora_alpha=None,
        lora_target=None,
    ))
    if rc != 0:
        return rc
    return cmd_evaluate(argparse.Namespace(
        table=1,
        variant="all",
        compare_paper=False,
        mode="lora",
        episodes=ns.episodes,
        out=str(config.RESULTS_DIR / "table1_3b_metrics.json"),
        log_file=None,
        quiet=ns.quiet,
        paper=True,
        model_id=None,
        checkpoint_dir=None,
        lora_r=None,
        lora_alpha=None,
        lora_target=None,
    ))


def _add_paper_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Qwen2.5-3B + paper LoRA (r=16, alpha=32, all-linear, lora_3b/)",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-target", default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description="SARL: dpo -> lora -> evaluate")
    sub = parser.add_subparsers(dest="stage", required=True)

    g = sub.add_parser("generate", help="Stage 1: trajectories + DPO pairs -> dpo/")
    gx = g.add_mutually_exclusive_group(required=True)
    gx.add_argument("--config", type=Path)
    gx.add_argument("--suite", type=Path)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--pair-only", action="store_true")
    g.add_argument("--max-pairs", type=int, default=None)
    g.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help=f"Rollout episodes per env (default: {config.DEFAULT_N_EPISODES})",
    )
    g.add_argument("--quiet", action="store_true")

    t = sub.add_parser("train", help="Stage 2: DPO LoRA -> lora/")
    t.add_argument("--train-all", action="store_true", help="Train all variants from config")
    t.add_argument("--pairs", type=Path)
    t.add_argument("--out", type=Path)
    t.add_argument("--smoke-only", action="store_true")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=f"Max tokens per DPO example (default: {config.LOCAL_MAX_SEQ_LENGTH} local)",
    )
    t.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help=f"Gradient accumulation (default: {config.LOCAL_GRADIENT_ACCUMULATION} local)",
    )
    _add_paper_flags(t)

    e = sub.add_parser("evaluate", help="Stage 3: Tables 1-7")
    e.add_argument("--table", type=int, default=1, choices=range(1, 8))
    e.add_argument("--variant", default=None)
    e.add_argument("--compare-paper", action="store_true")
    e.add_argument("--mode", default="lora", choices=["heuristic", "lora"])
    e.add_argument("--episodes", type=int, default=None)
    e.add_argument("--out", type=Path, default=None, help="Metrics JSON output path")
    e.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Write under eval/runs/<dir>/ (auto timestamp if --out omitted)",
    )
    e.add_argument("--log-file", type=Path, default=None, help="Eval progress log path")
    e.add_argument("--quiet", action="store_true", help="No progress log during eval")
    _add_paper_flags(e)

    a = sub.add_parser("all", help="generate --prepare + train-all + table 1")
    ax = a.add_mutually_exclusive_group(required=True)
    ax.add_argument("--config", type=Path)
    ax.add_argument("--suite", type=Path)
    a.add_argument("--prepare", action="store_true", default=True)
    _add_paper_flags(a)

    a3 = sub.add_parser(
        "all-3b",
        help="smoke --prepare -> train lora_3b/all -> table1 eval",
    )
    a3.add_argument("--n-episodes", type=int, default=None)
    a3.add_argument("--epochs", type=int, default=None)
    a3.add_argument("--episodes", type=int, default=1, help="Eval episodes per env")
    a3.add_argument("--quiet", action="store_true")

    smoke = sub.add_parser(
        "smoke",
        help=f"Generate PD×3 data (default {config.DEFAULT_N_EPISODES} ep/env) -> dpo/data/*.jsonl",
    )
    paper = sub.add_parser(
        "paper",
        help=f"Generate PD×3 paper data (default {config.PAPER_N_EPISODES} ep/env) -> dpo/data/*.jsonl",
    )
    for p in (smoke, paper):
        p.add_argument("--prepare", action="store_true", default=True)
        p.add_argument("--no-prepare", action="store_true", help="Skip dpo/data export")
        p.add_argument("--pair-only", action="store_true")
        p.add_argument("--max-pairs", type=int, default=None)
        p.add_argument(
            "--n-episodes",
            type=int,
            default=None,
            help="Override default episode count (smoke: 16, paper: 1000)",
        )
        p.add_argument("--quiet", action="store_true")

    ns = parser.parse_args()
    if ns.stage == "smoke":
        return cmd_smoke(ns)
    if ns.stage == "paper":
        return cmd_paper(ns)
    if ns.stage == "generate":
        return cmd_generate(ns)
    if ns.stage == "train":
        return cmd_train(ns)
    if ns.stage == "evaluate":
        return cmd_evaluate(ns)
    if ns.stage == "all":
        return cmd_all(ns)
    if ns.stage == "all-3b":
        return cmd_all_3b(ns)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

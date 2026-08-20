#!/usr/bin/env python3
"""Export paper Table 5 pair counts from local JSONL rollouts.

Sources:
  --from-1000   merge data/*_1000/*.jsonl (default out: data/paper/)
  (default)     merge data/*.jsonl at repo root (default out: dpo/data/)

Paper targets (config.TRAINING_VARIANTS):
  filter_on=388, filter_off=407, core=503, aux=613, all=1338, rw=1749
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

PAPER_COUNTS = {
    "filter_on": config.TRAINING_VARIANTS["filter_on"]["pairs"],
    "filter_off": config.TRAINING_VARIANTS["filter_off"]["pairs"],
    "core": config.TRAINING_VARIANTS["core"]["pairs"],
    "aux": config.TRAINING_VARIANTS["aux"]["pairs"],
    "all": config.TRAINING_VARIANTS["all"]["pairs"],
    "rw": config.TRAINING_VARIANTS["rw"]["pairs"],
}

SEED = 42


def _row_key(row: dict) -> str:
    meta = row.get("meta") or {}
    group = meta.get("group")
    group_s = json.dumps(group, sort_keys=True) if group is not None else ""
    return "|".join(
        [
            row.get("prompt", "")[:256],
            row.get("chosen", "")[:128],
            meta.get("_source_env", ""),
            group_s,
        ]
    )


def _pair_fingerprint(row: dict) -> str:
    return "|".join([row.get("prompt", ""), row.get("chosen", ""), row.get("rejected", "")])


def _stable_order(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: hashlib.sha256(_row_key(r).encode()).hexdigest())


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        fp = _pair_fingerprint(row)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(row)
    return out


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = json.loads(json.dumps(row))
            meta = clean.get("meta")
            if isinstance(meta, dict):
                meta.pop("_source_env", None)
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _take(rows: list[dict], n: int) -> list[dict]:
    if n > len(rows):
        raise ValueError(f"need {n} rows but only {len(rows)} available")
    return rows[:n]


def _reward_margin(row: dict) -> float:
    meta = row.get("meta") or {}
    cr = meta.get("cumulative_reward_chosen", meta.get("chosen_reward"))
    rr = meta.get("cumulative_reward_rejected", meta.get("rejected_reward"))
    if cr is None or rr is None:
        return 0.0
    return float(cr) - float(rr)


def _env_dirs(data_root: Path) -> list[Path]:
    return sorted(
        d for d in data_root.iterdir() if d.is_dir() and d.name.endswith("_1000")
    )


def _merge_env_rows(env_dirs: list[Path], filename: str) -> list[dict]:
    rows: list[dict] = []
    per_env: dict[str, int] = {}
    for env_dir in env_dirs:
        path = env_dir / filename
        if not path.is_file():
            continue
        n = 0
        for row in _load_jsonl(path):
            tagged = json.loads(json.dumps(row))
            meta = tagged.setdefault("meta", {})
            meta["_source_env"] = env_dir.name
            rows.append(tagged)
            n += 1
        per_env[env_dir.name] = n
    return rows, per_env


def _resolve_flat_src(name: str) -> Path:
    for candidate in (ROOT / "data" / name, ROOT / "dpo" / "data" / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def _load_pool(*, from_1000: bool) -> tuple[dict[str, list[dict]], dict]:
    meta: dict = {"source": "data/*_1000" if from_1000 else "data/*.jsonl"}

    if from_1000:
        env_dirs = _env_dirs(ROOT / "data")
        if not env_dirs:
            raise FileNotFoundError("No data/*_1000 directories under data/")
        meta["env_dirs"] = [d.name for d in env_dirs]
        filter_on, fo_env = _merge_env_rows(env_dirs, "filter_on.jsonl")
        filter_off, ff_env = _merge_env_rows(env_dirs, "filter_off.jsonl")
        abeta, ab_env = _merge_env_rows(env_dirs, "a_beta_all.jsonl")
        rw, rw_env = _merge_env_rows(env_dirs, "a_beta_rw.jsonl")
        meta["per_env_raw"] = {
            "filter_on": fo_env,
            "filter_off": ff_env,
            "a_beta_all": ab_env,
            "a_beta_rw": rw_env,
        }
    else:
        filter_on = _load_jsonl(_resolve_flat_src("filter_on.jsonl"))
        filter_off = _load_jsonl(_resolve_flat_src("filter_off.jsonl"))
        abeta = _load_jsonl(_resolve_flat_src("a_beta_all.jsonl"))
        rw = _load_jsonl(_resolve_flat_src("a_beta_rw.jsonl"))

    return {
        "filter_on": _dedupe(filter_on),
        "filter_off": _dedupe(filter_off),
        "abeta": _dedupe(abeta),
        "rw": _dedupe(rw),
    }, meta


def _env_breakdown(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        env = (row.get("meta") or {}).get("_source_env", "unknown")
        counts[env] = counts.get(env, 0) + 1
    return dict(sorted(counts.items()))


def build_corpora(out_dir: Path, *, from_1000: bool) -> dict[str, int]:
    pools, src_meta = _load_pool(from_1000=from_1000)

    filter_on = _take(_stable_order(pools["filter_on"]), PAPER_COUNTS["filter_on"])
    filter_off = _take(_stable_order(pools["filter_off"]), PAPER_COUNTS["filter_off"])

    abeta_sorted = _stable_order(pools["abeta"])
    core = _take(abeta_sorted, PAPER_COUNTS["core"])
    aux = _take(abeta_sorted[PAPER_COUNTS["core"] :], PAPER_COUNTS["aux"])
    all_rows = _take(abeta_sorted, PAPER_COUNTS["all"])

    all_fps = {_pair_fingerprint(r) for r in all_rows}
    extras = [r for r in pools["rw"] if _pair_fingerprint(r) not in all_fps]
    extras.sort(key=lambda r: (-_reward_margin(r), _row_key(r)))
    rw_rows = all_rows + _take(extras, PAPER_COUNTS["rw"] - len(all_rows))

    outputs = {
        "filter_on.jsonl": filter_on,
        "filter_off.jsonl": filter_off,
        "a_beta_core.jsonl": core,
        "a_beta_aux.jsonl": aux,
        "a_beta_all.jsonl": all_rows,
        "a_beta_rw.jsonl": rw_rows,
    }

    counts: dict[str, int] = {}
    per_env_selected: dict[str, dict[str, int]] = {}
    for fname, rows in outputs.items():
        dest = out_dir / fname
        _write_jsonl(dest, rows)
        counts[fname] = len(rows)
        per_env_selected[fname] = _env_breakdown(rows)
        print(f"  {dest.relative_to(ROOT)}: {len(rows)} pairs")

    manifest = {
        "source": src_meta,
        "seed": SEED,
        "paper_counts": PAPER_COUNTS,
        "counts": counts,
        "per_env_selected": per_env_selected,
        "notes": (
            "Merged from data/*_1000 in env-name order, deduped, then deterministically "
            "subsampled to paper Table 5 counts. CORE/AUX are disjoint sequential slices "
            "of the merged A+β pool; ALL is a larger prefix; RW = ALL + top reward-margin extras."
        ),
    }
    (out_dir / "latest_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare paper-scale JSONL corpora")
    parser.add_argument(
        "--from-1000",
        action="store_true",
        help="Merge from data/*_1000/ instead of flat data/*.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/paper with --from-1000, else dpo/data)",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        out_dir = ROOT / "data" / "paper" if args.from_1000 else config.DPO_DATA_DIR
    else:
        out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    label = "data/*_1000" if args.from_1000 else "data/*.jsonl"
    print(f"Writing paper corpora ({label}) -> {out_dir}")
    build_corpora(out_dir, from_1000=args.from_1000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

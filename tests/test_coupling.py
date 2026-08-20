"""Unit tests for paper Appendix D coupling parser."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.coupling import coupling_for_round, parse_ev_slot


def test_pattern_a_colon():
    text = "[EV] C: 0.5*3 + 0.5*0 = 1.5\nD: 0.5*5 + 0.5*1 = 3.0\n[Decision] Play D"
    evs = parse_ev_slot(text)
    assert "C" in evs and "D" in evs
    assert evs["D"] > evs["C"]
    assert coupling_for_round(text, "D").coupled
    assert not coupling_for_round(text, "C").coupled


def test_pattern_b_ev_wrapper():
    text = "[EV] EV(C) = 0.30(3.0) + 0.40(0) = 1.2\nEV(D) = 0.30(5.0) + 0.40(1) = 1.9"
    assert parse_ev_slot(text)["D"] > parse_ev_slot(text)["C"]
    assert coupling_for_round(text, "D").coupled


def test_pattern_c_prose():
    text = (
        "[EV] Expected payoff for 'C' is 0.6*3 + 0.4*0 = 1.8\n"
        "Total EV for 'D' = 0.6*5 + 0.4*1 = 3.4"
    )
    assert coupling_for_round(text, "D").coupled
    assert not coupling_for_round(text, "C").coupled


def test_training_jsonl_sample():
    path = ROOT / "data/paper/a_beta_core.jsonl"
    if not path.is_file():
        return
    coupled = parsable = 0
    n = 0
    for line in path.open(encoding="utf-8"):
        if n >= 50:
            break
        d = json.loads(line)
        thinking = d["chosen"]
        import re

        m = re.search(r"<think>(.*?)</think>", thinking, re.S)
        reasoning = m.group(1) if m else thinking
        act = re.search(r"<action>(.*?)</action>", thinking, re.S)
        action = act.group(1).strip() if act else ""
        c = coupling_for_round(reasoning, action)
        if c.parsable:
            parsable += 1
            coupled += int(c.coupled)
        n += 1
    assert parsable >= 10, f"expected parsable EV rounds in training data, got {parsable}/{n}"

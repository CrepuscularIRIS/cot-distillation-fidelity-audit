#!/usr/bin/env python3
"""Calibration tool: compare Kimi judge scores with lead-analyst (Claude) scores.

Reads pilot_judged.jsonl, fetches the original traces, and for each one:
- Shows the problem + reasoning excerpt + Kimi's scores + evidence
- Claude (the lead analyst running this script's output) records agreement/disagreement

The output is outputs/calibration.jsonl — each row has the Kimi scores, the
calibration scores, and whether they agree on CCR closure (within ±1 = "agree").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def load_judged() -> list[dict]:
    rows = []
    with (OUT / "pilot_judged.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summary_report(rows: list[dict]) -> None:
    """Print a compact calibration overview."""
    cal = []
    with (OUT / "calibration.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cal.append(json.loads(line))
    if not cal:
        print("No calibration data yet.")
        return

    agree = sum(1 for c in cal if abs(c["kimi_ccr"] - c["claude_ccr"]) <= 1)
    exact = sum(1 for c in cal if c["kimi_ccr"] == c["claude_ccr"])
    n = len(cal)
    print(f"\n{'='*60}")
    print(f"Calibration: {n} traces | exact-match={exact}/{n} ({100*exact/n:.0f}%) | ±1-agree={agree}/{n} ({100*agree/n:.0f}%)")
    for c in cal:
        flag = "✓" if abs(c["kimi_ccr"] - c["claude_ccr"]) <= 1 else "✗"
        print(f"  {flag} {c['dataset']:9s} kimi_ccr={c['kimi_ccr']} claude_ccr={c['claude_ccr']} "
              f"coupling: {c.get('kimi_coupling','?')}/{c.get('claude_coupling','?')} "
              f"depth: {c.get('kimi_depth','?')}/{c.get('claude_depth','?')}")


def main() -> None:
    rows = load_judged()
    if not rows:
        print("No judged traces yet. Run the pilot first.")
        return

    print(f"Loaded {len(rows)} judged traces for calibration review.\n")
    print("For each trace below, I (Claude, the lead analyst) will record my own")
    print("scores in outputs/calibration.jsonl based on reading the reasoning excerpt.\n")

    for i, r in enumerate(rows):
        print(f"\n{'='*60}")
        print(f"Trace {i+1}/{len(rows)} | {r['dataset']} | teacher={r['teacher']} | uid={r['uid']}")
        print(f"Reasoning chars: {r['reasoning_chars']} | Crit density/1k: {r['crit_density_1k']}")
        print(f"\nKimi scores:")
        for k in ("ccr_closure", "monitoring_control_coupling", "causal_depth_of_critique",
                   "answer_change", "verification_independence", "failure_type",
                   "reasoning_topology", "uncertainty_stage"):
            print(f"  {k}: {r.get(k, '?')}")
        print(f"\nEvidence: {str(r.get('evidence',''))[:400]}")
        print(f"Rationale: {str(r.get('rationale',''))[:300]}")

    print(f"\n{'='*60}")
    print(f"All {len(rows)} traces printed. Use this output to record calibration scores.")
    print("Write calibration.jsonl rows with fields: uid, dataset, kimi_ccr, claude_ccr,")
    print("kimi_coupling, claude_coupling, kimi_depth, claude_depth, notes.")

    if (OUT / "calibration.jsonl").exists():
        summary_report(rows)


if __name__ == "__main__":
    main()

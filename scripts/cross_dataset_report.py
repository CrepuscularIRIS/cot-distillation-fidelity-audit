#!/usr/bin/env python3
"""Cross-dataset comparison report: reads all *_judged.jsonl files and produces
a unified comparison table + the convergence analysis for Round 4.

Outputs:
- outputs/cross_dataset_report.json — structured comparison
- prints a markdown table to stdout for the report
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cross")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

DATASETS_ORDER = ["deepseek", "glm", "claude46_ti", "claude47_ti", "nohurry_opus", "gemini"]
DATASET_LABELS = {
    "deepseek": "DeepSeek-V4",
    "glm": "GLM-5.1",
    "claude46_ti": "Claude-4.6-TI",
    "claude47_ti": "Claude-4.7-TI",
    "nohurry_opus": "Opus-4.6-filt",
    "gemini": "Gemini-3.1",
}


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_mean(vals: list) -> float:
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def safe_median(vals: list) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def load_dataset_rows(name: str) -> list[dict]:
    # Try both naming conventions
    for pattern in [f"{name}_judged.jsonl", "pilot_judged.jsonl"]:
        path = OUT / pattern
        if not path.exists():
            continue
        rows = []
        for r in iter_jsonl(path):
            status = r.get("status", "judged")
            if status != "judged" and "ccr_closure" not in r:
                continue
            if pattern == "pilot_judged.jsonl":
                if r.get("dataset") != name:
                    continue
            if isinstance(r.get("ccr_closure"), int):
                rows.append(r)
        if rows:
            return rows
    return []


def analyze_dataset(name: str) -> dict | None:
    rows = load_dataset_rows(name)
    if not rows:
        return None

    ccr = [r["ccr_closure"] for r in rows]
    coupling = [r["monitoring_control_coupling"] for r in rows if isinstance(r.get("monitoring_control_coupling"), int)]
    depth = [r["causal_depth_of_critique"] for r in rows if isinstance(r.get("causal_depth_of_critique"), int)]
    density = [r["crit_density_1k"] for r in rows if isinstance(r.get("crit_density_1k"), (int, float))]

    topo = Counter(r.get("reasoning_topology", "?") for r in rows)
    failure = Counter(r.get("failure_type") for r in rows if r.get("failure_type"))
    ans_change = Counter(r.get("answer_change", "?") for r in rows)

    planning = [r["planning_depth"] for r in rows if isinstance(r.get("planning_depth"), int)]
    plan_exec = [r["plan_execution_consistency"] for r in rows if isinstance(r.get("plan_execution_consistency"), int)]
    divergent = [r["divergent_score"] for r in rows if isinstance(r.get("divergent_score"), int)]
    templated = sum(1 for r in rows if r.get("templated_divergence") == 1)
    creative = sum(1 for r in rows if r.get("creative_destruction") == 1)

    return {
        "n": len(rows),
        "ccr_mean": safe_mean(ccr),
        "ccr_median": safe_median(ccr),
        "coupling_mean": safe_mean(coupling),
        "depth_mean": safe_mean(depth),
        "density_mean": safe_mean(density),
        "topology": dict(topo.most_common()),
        "loop_pct": round(100 * topo.get("loop", 0) / len(rows), 1),
        "failure_types": dict(failure.most_common()),
        "answer_change": dict(ans_change.most_common()),
        "planning_mean": safe_mean(planning),
        "plan_exec_mean": safe_mean(plan_exec),
        "divergent_mean": safe_mean(divergent),
        "templated_divergence_count": templated,
        "creative_destruction_count": creative,
    }


def main() -> None:
    report = {}
    for name in DATASETS_ORDER:
        result = analyze_dataset(name)
        if result:
            report[name] = result
            log.info("%-15s n=%d ccr=%.2f coupling=%.2f density=%.1f loop=%.1f%%",
                     name, result["n"], result["ccr_mean"], result["coupling_mean"],
                     result["density_mean"], result["loop_pct"])

    if not report:
        log.warning("no data found")
        return

    (OUT / "cross_dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Markdown table
    print("\n## Cross-Dataset Comparison\n")
    print("| Dataset | n | CCR mean | CCR med | Coupling | Depth | Density/1k | Loop% | Plan | Divergent |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in DATASETS_ORDER:
        r = report.get(name)
        if not r:
            continue
        label = DATASET_LABELS.get(name, name)
        print(f"| {label} | {r['n']} | {r['ccr_mean']} | {r['ccr_median']} | "
              f"{r['coupling_mean']} | {r['depth_mean']} | {r['density_mean']} | "
              f"{r['loop_pct']}% | {r['planning_mean']} | {r['divergent_mean']} |")

    # Paradox check
    print("\n## Density vs Closure Paradox Check\n")
    items = [(name, r["density_mean"], r["ccr_mean"]) for name, r in report.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    for name, dens, ccr in items:
        arrow = "HIGH density LOW closure" if dens > 5 and ccr < 2 else "OK" if ccr >= 2 else "low both"
        print(f"  {DATASET_LABELS.get(name, name):20s}  density={dens:5.1f}  ccr={ccr:.2f}  -> {arrow}")

    # Hypothesis survival
    print("\n## Hypothesis Survival (preliminary)\n")
    if "deepseek" in report and "glm" in report:
        ds, gl = report["deepseek"], report["glm"]
        print(f"  H1 (topology): GLM loop%={gl['loop_pct']}% vs DS loop%={ds['loop_pct']}%")
        print(f"  H2 (inertia): density->closure paradox {'CONFIRMED' if ds['density_mean'] > gl['density_mean'] and ds['ccr_mean'] < gl['ccr_mean'] else 'MIXED'}")
    if "gemini" in report:
        g = report["gemini"]
        print(f"  Gemini templated divergence: {g['templated_divergence_count']}/{g['n']} traces flagged")

    log.info("wrote outputs/cross_dataset_report.json")


if __name__ == "__main__":
    main()

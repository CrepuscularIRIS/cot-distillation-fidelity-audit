#!/usr/bin/env python3
"""Compute mindset/Opus structural metrics (CCS/PDD/IS/EAR/OED/RT/RAE) on the
FULL judged traces, and cross-check convergent validity vs the LLM CCR.

Reproduces the exact judged traces (deterministic sampling) to get full reasoning,
joins with opus_all.jsonl ccr by (dataset, uid).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from distill_audit import structural  # noqa: E402
from distill_audit.runner_core import sample_traces  # noqa: E402
import run_pilot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("structural")
OUT = ROOT / "outputs"

DATASET_JOBS = [  # same as prep_batches.py
    ("claude46_ti", 40, None, None, 200), ("claude47_ti", 40, None, None, 200),
    ("nohurry_opus", 40, None, None, 200), ("gemini", 40, None, None, 200),
    ("kimi", 40, None, 60000, 200), ("qwen", 40, None, None, 200),
    ("angrygiraffe", 40, "full_train.jsonl", None, 200), ("roman_claude", 40, None, None, 60),
]
METRICS = ["ccs", "pdd", "inertia_slope", "ear", "oed", "rt_topology_density",
           "rt_branch_merge_ratio", "rt_back_ref_depth", "rae_bigram_entropy", "rae_routinization"]


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None
           and not (isinstance(x, float) and math.isnan(x))]
    if len(pts) < 4:
        return None
    xs2, ys2 = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = sum(xs2) / len(xs2), sum(ys2) / len(ys2)
    cov = sum((xs2[i] - mx) * (ys2[i] - my) for i in range(len(xs2)))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs2)); sy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    return round(cov / (sx * sy), 3) if sx * sy else None


def main():
    # ccr by (dataset, uid) from the Opus judge
    ccr = {}
    for line in (OUT / "opus_all.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if isinstance(r.get("ccr_closure"), int):
                ccr[(r["dataset"], r["uid"])] = r["ccr_closure"]

    # full traces: paired (deepseek+glm) + each dataset
    full: list = []  # (dataset, uid, reasoning)
    pairs = run_pilot.load_pairs(60)
    ds_ids = {d for d, _ in pairs}
    glm_lk = {g for _, g in pairs} | {d for d, g in pairs if d != g}
    ds_tr = run_pilot.traces_by_uid("deepseek", ds_ids)
    glm_tr = run_pilot.traces_by_uid("glm", glm_lk, file_filter="main")
    for ds_id, glm_id in pairs:
        if ds_id in ds_tr:
            full.append(("deepseek", ds_id, ds_tr[ds_id].reasoning))
        gt = glm_tr.get(glm_id) or glm_tr.get(ds_id)
        if gt:
            full.append(("glm", gt.uid, gt.reasoning))
    log.info("paired full traces: %d", len(full))
    for name, n, ff, scan, minc in DATASET_JOBS:
        for t in sample_traces(name, n, file_filter=ff, scan_limit=scan, min_chars=minc):
            full.append((name, t.uid, t.reasoning))
    log.info("total full traces: %d", len(full))

    rows = []
    with (OUT / "structural_metrics.jsonl").open("w") as fh:
        for dataset, uid, reasoning in full:
            m = structural.all_metrics(reasoning)
            m.update({"dataset": dataset, "uid": uid, "reasoning_chars": len(reasoning),
                      "llm_ccr": ccr.get((dataset, uid))})
            rows.append(m)
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    # per-dataset means
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    order = ["qwen", "glm", "kimi", "claude47_ti", "deepseek", "gemini", "claude46_ti",
             "nohurry_opus", "angrygiraffe", "roman_claude"]
    print("\n=== STRUCTURAL METRICS per dataset (mindset/Opus, computed on full traces) ===")
    hdr = ["dataset", "n", "llm_ccr", "CCS", "PDD", "IS", "EAR", "OED", "RT_dens", "RAE_ent", "RAE_rout"]
    print("  " + " ".join(f"{h:>9s}" for h in hdr))

    def avg(rs, k):
        vs = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and not (isinstance(r[k], float) and math.isnan(r[k]))]
        return round(sum(vs) / len(vs), 2) if vs else None

    ds_summary = {}
    for ds in order:
        if ds not in by_ds:
            continue
        rs = by_ds[ds]
        vals = [ds, len(rs), avg(rs, "llm_ccr"), avg(rs, "ccs"), avg(rs, "pdd"), avg(rs, "inertia_slope"),
                avg(rs, "ear"), avg(rs, "oed"), avg(rs, "rt_topology_density"),
                avg(rs, "rae_bigram_entropy"), avg(rs, "rae_routinization")]
        ds_summary[ds] = dict(zip(hdr, vals))
        print("  " + " ".join(f"{str(v):>9s}" for v in vals))

    # convergent validity: each structural metric vs LLM ccr (trace-level)
    print("\n=== CONVERGENT VALIDITY: Pearson(structural metric, LLM CCR) ===")
    corr = {}
    for k in METRICS:
        c = pearson([r.get(k) for r in rows], [r.get("llm_ccr") for r in rows])
        corr[k] = c
        print(f"  {k:22s}: {c}")

    (OUT / "report_structural.json").write_text(json.dumps(
        {"per_dataset": ds_summary, "convergent_validity_pearson_vs_llm_ccr": corr,
         "n_traces": len(rows)}, indent=2, ensure_ascii=False))
    log.info("wrote outputs/structural_metrics.jsonl + report_structural.json")


if __name__ == "__main__":
    main()

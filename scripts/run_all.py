#!/usr/bin/env python3
"""Run the full audit: paired round (GLM<->DeepSeek) + all standalone datasets.

Concurrent (thread-pooled) judging, one output file per job (no collisions),
fully resumable. Single long-running process -> robust to subagent teardown.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from distill_audit.runner_core import judge_many, sample_traces  # noqa: E402
import run_pilot  # noqa: E402  (reuse load_pairs / traces_by_uid)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(ROOT / "outputs" / "run_all.log")])
log = logging.getLogger("run_all")
OUT = ROOT / "outputs"

# Standalone dataset jobs: (name, sample_n, file_filter, scan_limit, min_chars)
JOBS = [
    ("claude46_ti", 200, None, None, 200),
    ("claude47_ti", 200, None, None, 200),
    ("nohurry_opus", 250, None, None, 200),
    ("gemini", 200, None, None, 200),
    ("kimi", 200, None, 60000, 200),
    ("qwen", 633, None, None, 200),
    ("angrygiraffe", 200, "full_train.jsonl", None, 200),
    ("roman_claude", 250, None, None, 60),  # direct-answer baseline: keep short ones
]


def write_summary(name: str) -> None:
    path = OUT / f"{name}_judged.jsonl"
    if not path.exists():
        return
    rows = [r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())
            if isinstance(r.get("ccr_closure"), int)]
    n_err = sum(1 for l in path.read_text().splitlines() if l.strip()
                and json.loads(l).get("status") == "error")
    if not rows:
        return
    ccr = [r["ccr_closure"] for r in rows]
    coupling = [r["monitoring_control_coupling"] for r in rows if isinstance(r.get("monitoring_control_coupling"), int)]
    depth = [r["causal_depth_of_critique"] for r in rows if isinstance(r.get("causal_depth_of_critique"), int)]
    density = [r["crit_density_1k"] for r in rows]
    topo = Counter(r.get("reasoning_topology") for r in rows)
    fail = Counter(r.get("failure_type") for r in rows if r.get("failure_type"))
    summary = {
        "n_judged": len(rows), "n_errors": n_err,
        "ccr_mean": round(sum(ccr) / len(ccr), 3),
        "coupling_mean": round(sum(coupling) / max(len(coupling), 1), 3),
        "depth_mean": round(sum(depth) / max(len(depth), 1), 3),
        "density_mean": round(sum(density) / len(density), 2),
        "topology": dict(topo.most_common()),
        "loop_pct": round(100 * topo.get("loop", 0) / len(rows), 1),
        "failure_types": dict(fail.most_common()),
    }
    (OUT / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("SUMMARY %s: n=%d ccr=%.2f coupling=%.2f density=%.1f loop=%.1f%% topo=%s",
             name, summary["n_judged"], summary["ccr_mean"], summary["coupling_mean"],
             summary["density_mean"], summary["loop_pct"], summary["topology"])


def run_paired(n_pairs: int, max_tokens: int, conc: int) -> None:
    log.info("=== PAIRED ROUND: GLM <-> DeepSeek (%d pairs) ===", n_pairs)
    pairs = run_pilot.load_pairs(n_pairs)
    ds_ids = {d for d, _ in pairs}
    glm_lookups = {g for _, g in pairs} | {d for d, g in pairs if d != g}
    ds_tr = run_pilot.traces_by_uid("deepseek", ds_ids)
    glm_tr = run_pilot.traces_by_uid("glm", glm_lookups, file_filter="main")
    log.info("loaded %d ds + %d glm traces", len(ds_tr), len(glm_tr))
    items: list = []
    for ds_id, glm_id in pairs:
        dt = ds_tr.get(ds_id)
        gt = glm_tr.get(glm_id) or glm_tr.get(ds_id)
        tag = {"pair_ds": ds_id, "pair_glm": glm_id}
        if dt is not None:
            items.append((dt, tag))
        if gt is not None:
            items.append((gt, tag))
    judge_many(items, OUT / "pilot_judged.jsonl", max_tokens=max_tokens, concurrency=conc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--only", type=str, default=None, help="comma list of jobs; 'paired' included")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    if only is None or "paired" in only:
        run_paired(args.pairs, args.max_tokens, args.concurrency)

    for name, n, ff, scan, minc in JOBS:
        if only is not None and name not in only:
            continue
        log.info("=== DATASET: %s (sample %d) ===", name, n)
        try:
            traces = sample_traces(name, n, file_filter=ff, scan_limit=scan, min_chars=minc)
            log.info("%s: sampled %d traces", name, len(traces))
            judge_many([(t, {}) for t in traces], OUT / f"{name}_judged.jsonl",
                       max_tokens=args.max_tokens, concurrency=args.concurrency)
            write_summary(name)
        except Exception as e:  # noqa: BLE001
            log.error("job %s failed: %s", name, e)

    log.info("=== ALL JOBS DONE ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Round 6 / H4 step 2 (model-agnostic): given a model's generated traces jsonl,
(a) compute the zero-API structural + aesthetic metrics for a quick closure read,
(b) write judge batches in the SAME item format the audited 440 traces used, so the
model's CCR is directly comparable.

Usage:  python scripts/h4_prep_and_struct.py --traces outputs/h4_glm9b_traces.jsonl --tag glm9b
Then judge outputs/batches/<tag>_*.json with the standard Opus rubric workflow.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from distill_audit import structural, aesthetics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("h4_prep")

OUT = ROOT / "outputs"
BATCH_DIR = OUT / "batches"
HEAD_TAIL = 4500
BATCH_SIZE = 5


def trunc(text: str, n: int = HEAD_TAIL) -> str:
    if len(text) <= n * 2:
        return text
    return f"{text[:n]}\n\n[... {len(text) - n*2} chars elided ...]\n\n{text[-n:]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="path to <model>_traces.jsonl")
    ap.add_argument("--tag", required=True, help="model label (used for outputs)")
    args = ap.parse_args()

    traces = Path(args.traces)
    if not traces.exists():
        log.error("missing %s", traces); return
    recs = [json.loads(x) for x in traces.read_text().splitlines() if x.strip()]
    log.info("[%s] loaded %d traces", args.tag, len(recs))

    rae, rtden, oed_hits, n_closed = [], [], 0, 0
    metrics_path = OUT / f"{args.tag}_metrics.jsonl"
    with metrics_path.open("w") as fh:
        for r in recs:
            reasoning = r.get("reasoning", "")
            m = structural.all_metrics(reasoning)
            a = aesthetics.all_aesthetics(reasoning)
            row = {"uid": r["uid"], "dataset": args.tag, **m, **a,
                   "reasoning_chars": r.get("reasoning_chars", len(reasoning)),
                   "closed_think": r.get("closed_think")}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if m.get("rae_bigram_entropy") is not None:
                rae.append(m["rae_bigram_entropy"])
            if m.get("rt_topology_density") is not None:
                rtden.append(m["rt_topology_density"])
            if structural.OVERTHROW.search(reasoning):
                oed_hits += 1
            n_closed += bool(r.get("closed_think"))

    def avg(v):
        return round(sum(v) / len(v), 3) if v else None

    log.info("=== [%s] structural read (zero-API CCR proxy) ===", args.tag)
    log.info("  n=%d  closed_think=%d/%d", len(recs), n_closed, len(recs))
    log.info("  RAE_entropy mean=%s   (best dataset-rank predictor: qwen 2.85 .. roman 1.0)", avg(rae))
    log.info("  RT_density  mean=%s   (qwen 0.32 .. roman 0.0)", avg(rtden))
    log.info("  traces with overthrow marker: %d/%d (%.0f%%)",
             oed_hits, len(recs), 100 * oed_hits / max(len(recs), 1))

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob(f"{args.tag}_*.json"):
        old.unlink()
    items = [{"uid": r["uid"], "dataset": args.tag, "teacher": args.tag,
              "domain": r.get("domain", ""), "reasoning_chars": r.get("reasoning_chars", 0),
              "problem": (r.get("problem") or "")[:1500], "reasoning": trunc(r.get("reasoning", "")),
              "answer": (r.get("answer") or "")[:1200]} for r in recs if r.get("reasoning")]
    nb = 0
    for k in range(0, len(items), BATCH_SIZE):
        (BATCH_DIR / f"{args.tag}_{k // BATCH_SIZE:03d}.json").write_text(
            json.dumps(items[k:k + BATCH_SIZE], ensure_ascii=False, indent=1))
        nb += 1
    log.info("[%s] wrote %d judge batches (%d traces) -> outputs/batches/%s_*.json",
             args.tag, nb, len(items), args.tag)


if __name__ == "__main__":
    main()

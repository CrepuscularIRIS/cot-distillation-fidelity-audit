#!/usr/bin/env python3
"""Round 6 / H4 step 2: given outputs/qwopus_traces.jsonl (model-generated),
(a) compute the zero-API structural + aesthetic metrics for a quick closure read,
(b) write judge batches in the SAME item format the other 440 traces used, so the
Qwopus CCR is directly comparable.

Run (plain python, no GPU/API):  python scripts/h4_prep_and_struct.py
Then judge outputs/batches/qwopus_h4_*.json with the standard Opus rubric workflow.
"""
from __future__ import annotations

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
TRACES = OUT / "qwopus_traces.jsonl"
METRICS = OUT / "qwopus_metrics.jsonl"
BATCH_DIR = OUT / "batches"
HEAD_TAIL = 4500
BATCH_SIZE = 5

# dataset CCR baselines (from report_cross_dataset.json) for the H4 read-out
BASELINE_CCR = {"qwen": 2.075, "glm": 1.667, "kimi": 1.30, "deepseek": 0.60,
                "claude46_ti": 0.425, "nohurry_opus": 0.175, "roman_claude": 0.025}
# Qwopus training mix (model card): primarily Claude-Distillation + Kimi + Qwen
TRAIN_MIX = ["(primarily) Claude-Distillation ~low", "Kimi 1.30", "Qwen 2.08"]


def trunc(text: str, n: int = HEAD_TAIL) -> str:
    if len(text) <= n * 2:
        return text
    return f"{text[:n]}\n\n[... {len(text) - n*2} chars elided ...]\n\n{text[-n:]}"


def main() -> None:
    if not TRACES.exists():
        log.error("missing %s — run gen_qwopus_traces.py first", TRACES); return
    recs = [json.loads(x) for x in TRACES.read_text().splitlines() if x.strip()]
    log.info("loaded %d qwopus traces", len(recs))

    # --- structural + aesthetic metrics (zero API) ---
    rae, rtden, dens, oed_hits, ear_hits, n_closed = [], [], [], 0, 0, 0
    with METRICS.open("w") as fh:
        for r in recs:
            reasoning = r.get("reasoning", "")
            m = structural.all_metrics(reasoning)
            a = aesthetics.all_aesthetics(reasoning)
            row = {"uid": r["uid"], "dataset": "qwopus_gen", **m, **a,
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

    log.info("=== Qwopus structural read (zero-API proxy) ===")
    log.info("  n=%d  closed_think=%d/%d", len(recs), n_closed, len(recs))
    log.info("  RAE_entropy mean=%s  (dataset-rank predictor; qwen 2.85 .. roman 1.0)", avg(rae))
    log.info("  RT_density  mean=%s  (qwen 0.32 .. roman 0.0)", avg(rtden))
    log.info("  traces with an overthrow marker: %d/%d (%.0f%%)",
             oed_hits, len(recs), 100 * oed_hits / max(len(recs), 1))
    log.info("  training mix: %s", TRAIN_MIX)
    log.info("  -> H4: low RAE/RT/overthrow ~ inherits the low-closure (Claude-distill) majority;")
    log.info("         high ~ generalizes closure from the qwen/kimi minority.")

    # --- judge batches (same item format as prep_batches.trace_to_item) ---
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("qwopus_h4_*.json"):
        old.unlink()
    items = [{"uid": r["uid"], "dataset": "qwopus_gen", "teacher": "Qwopus3.6-27B-v1-preview",
              "domain": r.get("domain", ""), "reasoning_chars": r.get("reasoning_chars", 0),
              "problem": (r.get("problem") or "")[:1500], "reasoning": trunc(r.get("reasoning", "")),
              "answer": (r.get("answer") or "")[:1200]} for r in recs if r.get("reasoning")]
    nb = 0
    for k in range(0, len(items), BATCH_SIZE):
        (BATCH_DIR / f"qwopus_h4_{k // BATCH_SIZE:03d}.json").write_text(
            json.dumps(items[k:k + BATCH_SIZE], ensure_ascii=False, indent=1))
        nb += 1
    log.info("wrote %d judge batches (%d traces) -> outputs/batches/qwopus_h4_*.json", nb, len(items))


if __name__ == "__main__":
    main()

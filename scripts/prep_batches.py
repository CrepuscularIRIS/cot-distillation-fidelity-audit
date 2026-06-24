#!/usr/bin/env python3
"""Prepare trace batches on disk for the Opus-4.8 judge workflow.

Samples the paired round (GLM<->DeepSeek) + every standalone dataset, truncates
long reasoning head+tail, and writes small JSON batch files that workflow judge
agents will Read and score. Emits outputs/batches/manifest.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from distill_audit.runner_core import sample_traces  # noqa: E402
from distill_audit.schema import Trace  # noqa: E402
import run_pilot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prep")

OUT = ROOT / "outputs"
BATCH_DIR = OUT / "batches"
BATCH_SIZE = 5
HEAD_TAIL = 4500  # chars each end for long traces

# (name, sample_n, file_filter, scan_limit, min_chars)
DATASET_JOBS = [
    ("claude46_ti", 40, None, None, 200),
    ("claude47_ti", 40, None, None, 200),
    ("nohurry_opus", 40, None, None, 200),
    ("gemini", 40, None, None, 200),
    ("kimi", 40, None, 60000, 200),
    ("qwen", 40, None, None, 200),
    ("angrygiraffe", 40, "full_train.jsonl", None, 200),
    ("roman_claude", 40, None, None, 60),
]
N_PAIRS = 60


def trunc(text: str, head_tail: int = HEAD_TAIL) -> str:
    if len(text) <= head_tail * 2:
        return text
    return (f"{text[:head_tail]}\n\n[... {len(text) - head_tail*2} chars elided ...]\n\n"
            f"{text[-head_tail:]}")


def trace_to_item(t: Trace, extra: dict) -> dict:
    return {
        "uid": t.uid, "dataset": t.dataset, "teacher": t.teacher,
        "domain": t.domain, "reasoning_chars": t.reasoning_chars,
        "problem": t.problem[:1500],
        "reasoning": trunc(t.reasoning),
        "answer": t.answer[:1200],
        **extra,
    }


def write_batches(items: list[dict], job: str, manifest: list) -> None:
    for k in range(0, len(items), BATCH_SIZE):
        chunk = items[k:k + BATCH_SIZE]
        fname = f"{job}_{k // BATCH_SIZE:03d}.json"
        (BATCH_DIR / fname).write_text(json.dumps(chunk, ensure_ascii=False, indent=1))
        manifest.append({"file": fname, "job": job, "n": len(chunk),
                         "uids": [c["uid"] for c in chunk]})


def main() -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("*.json"):
        old.unlink()
    manifest: list = []

    # Paired round
    log.info("preparing paired round (%d pairs)...", N_PAIRS)
    pairs = run_pilot.load_pairs(N_PAIRS)
    ds_ids = {d for d, _ in pairs}
    glm_lookups = {g for _, g in pairs} | {d for d, g in pairs if d != g}
    ds_tr = run_pilot.traces_by_uid("deepseek", ds_ids)
    glm_tr = run_pilot.traces_by_uid("glm", glm_lookups, file_filter="main")
    paired_items: list[dict] = []
    for ds_id, glm_id in pairs:
        tag = {"pair_ds": ds_id, "pair_glm": glm_id}
        dt, gt = ds_tr.get(ds_id), glm_tr.get(glm_id) or glm_tr.get(ds_id)
        if dt:
            paired_items.append(trace_to_item(dt, tag))
        if gt:
            paired_items.append(trace_to_item(gt, tag))
    write_batches(paired_items, "paired", manifest)
    log.info("paired: %d traces -> %d batches", len(paired_items),
             (len(paired_items) + BATCH_SIZE - 1) // BATCH_SIZE)

    # Standalone datasets
    for name, n, ff, scan, minc in DATASET_JOBS:
        traces = sample_traces(name, n, file_filter=ff, scan_limit=scan, min_chars=minc)
        items = [trace_to_item(t, {}) for t in traces]
        write_batches(items, name, manifest)
        log.info("%s: %d traces -> %d batches", name, len(items),
                 (len(items) + BATCH_SIZE - 1) // BATCH_SIZE)

    (BATCH_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    total = sum(m["n"] for m in manifest)
    log.info("=== MANIFEST: %d batches, %d traces total ===", len(manifest), total)


if __name__ == "__main__":
    main()

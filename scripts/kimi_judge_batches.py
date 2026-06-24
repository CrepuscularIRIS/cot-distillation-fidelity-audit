#!/usr/bin/env python3
"""Judge the SAME batch files Opus scored, but with Kimi-for-coding, for a large
Opus-vs-Kimi calibration overlap (strengthens RQ3, which Codex flagged as thin).

Reads outputs/batches/<file>.json, scores each trace with Kimi via the same rubric
Opus used, writes outputs/kimi_scores/<file>.json (same order -> index-join).
Resumable per-batch; concurrency-limited + 429-backoff to respect quota.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from distill_audit.judge import client, rubric  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "outputs" / "kimi_judge.log")])
log = logging.getLogger("kimi")

BATCH = ROOT / "outputs" / "batches"
KSCORES = ROOT / "outputs" / "kimi_scores"


def build_prompt(entry: dict) -> str:
    """Mirror exactly what the Opus agents saw (rubric.build_user_prompt format)."""
    return (
        f"PROBLEM:\n{(entry.get('problem') or '')[:2000]}\n\n"
        f"MODEL CoT (reasoning to be scored):\n{entry.get('reasoning') or ''}\n\n"
        f"FINAL ANSWER (context only — do NOT score its correctness):\n{(entry.get('answer') or '')[:2000]}\n\n"
        "Return the JSON object now."
    )


def judge_one(entry: dict, max_tokens: int) -> dict:
    try:
        content, _ = client.call(rubric.SYSTEM, build_prompt(entry), max_tokens=max_tokens)
        d = client.parse_json(content)
        d["uid"] = entry.get("uid")
        d["_status"] = "judged"
        return d
    except client.JudgeError as e:
        return {"uid": entry.get("uid"), "_status": "error", "_err": str(e)[:120]}


def batch_done(file: str) -> bool:
    p = KSCORES / file
    if not p.exists():
        return False
    try:
        scores = json.loads(p.read_text())
        batch = json.loads((BATCH / file).read_text())
        return len(scores) == len(batch) and all(s.get("_status") == "judged" for s in scores)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", type=str, default=None, help="comma substring filter on batch filenames")
    args = ap.parse_args()
    KSCORES.mkdir(parents=True, exist_ok=True)

    files = sorted(p.name for p in BATCH.glob("*.json") if p.name != "manifest.json")
    if args.only:
        subs = args.only.split(",")
        files = [f for f in files if any(s in f for s in subs)]
    todo = [f for f in files if not batch_done(f)]
    log.info("Kimi calibration: %d batches total, %d to judge (conc=%d)", len(files), len(todo), args.concurrency)

    lock = threading.Lock()
    for fi, file in enumerate(todo):
        batch = json.loads((BATCH / file).read_text())
        results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(judge_one, e, args.max_tokens): i for i, e in enumerate(batch)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
        ordered = [results[i] for i in range(len(batch))]
        with lock:
            (KSCORES / file).write_text(json.dumps(ordered, ensure_ascii=False))
        ok = sum(1 for r in ordered if r.get("_status") == "judged")
        ccrs = [r.get("ccr_closure") for r in ordered if isinstance(r.get("ccr_closure"), int)]
        log.info("[%d/%d] %s: %d/%d ok, mean_ccr=%.2f",
                 fi + 1, len(todo), file, ok, len(batch),
                 (sum(ccrs) / len(ccrs)) if ccrs else float("nan"))
    log.info("=== Kimi calibration judging done ===")


if __name__ == "__main__":
    main()

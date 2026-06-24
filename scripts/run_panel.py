#!/usr/bin/env python3
"""Multi-model judge PANEL: score the same sampled traces with all 3 fusion
backends (Kimi, DeepSeek-V4-Flash, MiniMax-M3). The 3 backends run in PARALLEL
(independent endpoints), each at LOW concurrency. Resumable. Writes
outputs/panel/<backend>/<batch>.json (index-aligned to the batch file).
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
from distill_audit.judge import fusion, rubric, client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "outputs" / "panel.log")])
log = logging.getLogger("panel")
BATCH = ROOT / "outputs" / "batches"
PANEL = ROOT / "outputs" / "panel"

# Sample: matched pairs (for the headline) + 1 batch per dataset (for rankings)
SAMPLE = ([f"paired_{i:03d}.json" for i in range(10)] +
          [f"{d}_000.json" for d in ["claude46_ti", "claude47_ti", "nohurry_opus",
                                      "gemini", "kimi", "qwen", "angrygiraffe", "roman_claude"]])


def build_user(entry: dict) -> str:
    return (f"PROBLEM:\n{(entry.get('problem') or '')[:1500]}\n\n"
            f"MODEL CoT (reasoning to be scored):\n{entry.get('reasoning') or ''}\n\n"
            f"FINAL ANSWER (context only — do NOT score correctness):\n{(entry.get('answer') or '')[:1200]}\n\n"
            "Return the JSON object now.")


def judge_trace(fn, entry: dict, max_tokens: int) -> dict:
    try:
        content, usage = fn(rubric.SYSTEM, build_user(entry), max_tokens=max_tokens)
        d = client.parse_json(content)
        d["uid"] = entry.get("uid")
        d["_status"] = "judged"
        return d
    except Exception as e:  # noqa: BLE001
        return {"uid": entry.get("uid"), "_status": "error", "_err": str(e)[:120]}


def batch_done(backend: str, file: str) -> bool:
    p = PANEL / backend / file
    if not p.exists():
        return False
    try:
        s = json.loads(p.read_text())
        b = json.loads((BATCH / file).read_text())
        return len(s) == len(b) and all(x.get("_status") == "judged" for x in s)
    except Exception:
        return False


def run_backend(backend: str, files: list[str], conc: int, max_tokens: int) -> None:
    fn = fusion.BACKENDS[backend]
    (PANEL / backend).mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not batch_done(backend, f)]
    log.info("[%s] %d/%d batches to judge (conc=%d)", backend, len(todo), len(files), conc)
    for file in todo:
        batch = json.loads((BATCH / file).read_text())
        results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(judge_trace, fn, e, max_tokens): i for i, e in enumerate(batch)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        ordered = [results[i] for i in range(len(batch))]
        (PANEL / backend / file).write_text(json.dumps(ordered, ensure_ascii=False))
        ok = sum(1 for r in ordered if r.get("_status") == "judged")
        log.info("[%s] %s: %d/%d ok", backend, file, ok, len(batch))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="kimi,deepseek_flash,minimax_m3")
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args()
    backends = [b for b in args.backends.split(",") if b in fusion.BACKENDS]
    files = [f for f in SAMPLE if (BATCH / f).exists()]
    log.info("PANEL: %d backends x %d batches", len(backends), len(files))

    # 3 backends in parallel (independent endpoints); each iterates its batches at low conc
    with ThreadPoolExecutor(max_workers=len(backends)) as ex:
        futs = {ex.submit(run_backend, b, files, args.conc, args.max_tokens): b for b in backends}
        for fut in as_completed(futs):
            b = futs[fut]
            try:
                fut.result()
                log.info("[%s] DONE", b)
            except Exception as e:  # noqa: BLE001
                log.error("[%s] crashed: %s", b, e)
    log.info("=== PANEL COMPLETE ===")


if __name__ == "__main__":
    main()

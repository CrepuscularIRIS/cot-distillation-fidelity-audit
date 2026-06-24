#!/usr/bin/env python3
"""Phase 0 de-risk: measure GLM <-> DeepSeek matched-pair yield (no API).

Direction: index the SMALL side (DeepSeek, ~7.7k questions) in memory, then
stream the LARGE side (GLM, ~18 GB) once, checking membership. This is cheap on
memory and directly answers "how many of DeepSeek's questions appear in GLM?"
(= the matched-pair yield) while emitting the actual (ds_id, glm_id) manifest we
need for Phase 1/2.

Crash-resilient: matched pairs are appended to outputs/matched_pairs.jsonl as
found, and per-GLM-file progress is checkpointed, so a re-run resumes instead of
restarting from zero.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_audit.schema import normalize_problem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "check_pairing.log")],
)
log = logging.getLogger("check_pairing")

HUB = Path("/data/huggingface/hub")
PROGRESS = OUT / "pairing_progress.json"
MANIFEST = OUT / "matched_pairs.jsonl"
SUMMARY = OUT / "pairing_summary.json"


def snapshot_files(ds: str) -> list[Path]:
    base = HUB / f"datasets--{ds}" / "snapshots"
    return sorted(set(base.glob("*/*.jsonl")) | set(base.glob("*/**/*.jsonl")))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def h(text: str) -> str:
    return hashlib.sha1(normalize_problem(text).encode("utf-8")).hexdigest()


def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"completed_files": [], "glm_rows_seen": 0}


def main() -> None:
    # 1. Index DeepSeek (small): norm-hash -> {ds_id, domain}
    ds_files = snapshot_files("Jackrong--DeepSeek-V4-Distill-8000x")
    ds_index: dict[str, dict] = {}
    for p in ds_files:
        for rec in iter_jsonl(p):
            inp = rec.get("input") or ""
            if not normalize_problem(inp):
                continue
            ds_index[h(inp)] = {"ds_id": rec.get("id"), "domain": str(rec.get("domain", "?"))}
    log.info("DeepSeek indexed: %d unique-normalized questions", len(ds_index))

    # 2. Resume state
    prog = load_progress()
    completed = set(prog["completed_files"])
    matched_ds: set[str] = set()
    if MANIFEST.exists():
        for rec in iter_jsonl(MANIFEST):
            if rec.get("ds_hash"):
                matched_ds.add(rec["ds_hash"])
        log.info("resume: %d matched ds-questions already in manifest", len(matched_ds))

    glm_files = snapshot_files("Jackrong--GLM-5.1-Reasoning-1M-Cleaned")
    log.info("GLM files: %s (completed: %s)", [p.name for p in glm_files], sorted(completed))

    # 3. Stream GLM, check membership, append matches
    glm_rows = prog["glm_rows_seen"]
    with MANIFEST.open("a", encoding="utf-8") as man:
        for p in glm_files:
            if p.name in completed:
                log.info("skip (done): %s", p.name)
                continue
            log.info("streaming %s ...", p.name)
            file_rows = 0
            for rec in iter_jsonl(p):
                inp = rec.get("input") or ""
                if not normalize_problem(inp):
                    continue
                file_rows += 1
                glm_rows += 1
                key = h(inp)
                hit = ds_index.get(key)
                if hit is not None:
                    matched_ds.add(key)
                    man.write(json.dumps({
                        "ds_hash": key,
                        "ds_id": hit["ds_id"],
                        "glm_id": rec.get("id"),
                        "domain": hit["domain"],
                    }, ensure_ascii=False) + "\n")
                if glm_rows % 100000 == 0:
                    man.flush()
                    log.info("  GLM rows seen=%d | matched ds-questions=%d", glm_rows, len(matched_ds))
            man.flush()
            completed.add(p.name)
            PROGRESS.write_text(json.dumps(
                {"completed_files": sorted(completed), "glm_rows_seen": glm_rows}, indent=2))
            log.info("done %s (+%d rows) | matched ds-questions=%d", p.name, file_rows, len(matched_ds))

    # 4. Summary
    domains: Counter[str] = Counter()
    for rec in iter_jsonl(MANIFEST):
        domains[rec.get("domain", "?")] += 1
    n_ds = len(ds_index)
    summary = {
        "deepseek_questions": n_ds,
        "glm_rows_seen": glm_rows,
        "matched_ds_questions": len(matched_ds),
        "match_rate": round(len(matched_ds) / max(n_ds, 1), 4),
        "matched_pair_rows": sum(1 for _ in iter_jsonl(MANIFEST)),
        "matched_domains": domains.most_common(20),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("=" * 60)
    log.info("YIELD: %d / %d DeepSeek questions matched in GLM (%.1f%%)",
             len(matched_ds), n_ds, 100 * summary["match_rate"])
    log.info("matched domains: %s", domains.most_common(10))
    log.info("wrote outputs/pairing_summary.json + matched_pairs.jsonl")


if __name__ == "__main__":
    main()

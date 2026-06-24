#!/usr/bin/env python3
"""Phase 1 pilot: judge a sample of GLM<->DeepSeek matched pairs (paired) and
report the paired closure comparison + cheap density baseline.

Pinned single judge (Kimi) scores BOTH halves of every pair -> no judge-confound.
Writes per-trace rows to outputs/pilot_judged.jsonl (resumable: skips uids already
judged) so a crash or rate-limit loses nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_audit.adapters import iter_traces  # noqa: E402
from distill_audit.judge import client, rubric  # noqa: E402
from distill_audit.schema import Trace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "run_pilot.log")])
log = logging.getLogger("pilot")

_CRIT = re.compile(r"\b(wait|but|however|actually|reconsider|check|verify|mistake|wrong|"
                   r"recompute|let me|hmm)\b|不对|等等|然而|重新|检查|验证|错误|修正", re.IGNORECASE)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def crit_density_per_1k(text: str) -> float:
    toks = max(len(text) / 4.0, 1.0)  # ~4 chars/token proxy
    return 1000.0 * len(_CRIT.findall(text)) / toks


def load_pairs(n: int) -> list[tuple[str, str]]:
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for rec in iter_jsonl(OUT / "matched_pairs.jsonl"):
        if rec["ds_id"] in seen:
            continue
        seen.add(rec["ds_id"])
        pairs.append((rec["ds_id"], rec["glm_id"]))
        if len(pairs) >= n:
            break
    return pairs


def traces_by_uid(name: str, uids: set[str], file_filter: str | None = None) -> dict[str, Trace]:
    found: dict[str, Trace] = {}
    if not uids:
        return found
    # Fast path: use byte-offset index for GLM main lookups
    idx_path = OUT / "glm_main_uid_index.json"
    if name == "glm" and idx_path.exists():
        log.info("using UID index for GLM fast lookup (%d uids)", len(uids))
        idx = json.loads(idx_path.read_text())
        from distill_audit.adapters import ADAPTERS
        fn = ADAPTERS["output_thinking"]
        for uid in uids:
            entry = idx.get(uid)
            if not entry:
                continue
            fpath, offset, length = entry[0], entry[1], entry[2]
            with open(fpath, "rb") as fh:
                fh.seek(offset)
                line = fh.read(length)
            try:
                rec = json.loads(line)
                found[uid] = fn(rec, name)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        log.info("UID index: found %d / %d GLM traces", len(found), len(uids))
        return found
    # Fallback: stream
    for t in iter_traces(name, file_filter=file_filter):
        if t.uid in uids:
            found[t.uid] = t
            if len(found) == len(uids):
                break
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=20000)
    args = ap.parse_args()

    done = {r["uid"] for r in iter_jsonl(OUT / "pilot_judged.jsonl")}
    log.info("resume: %d traces already judged", len(done))

    pairs = load_pairs(args.pairs)
    ds_ids = {d for d, _ in pairs}
    glm_ids = {g for _, g in pairs}
    glm_fallback_ids = {d for d, g in pairs if d != g}
    all_glm_lookups = glm_ids | glm_fallback_ids
    log.info("loading traces: %d deepseek, %d glm (incl %d fallback-by-ds-id) ...",
             len(ds_ids), len(all_glm_lookups), len(glm_fallback_ids))
    ds_tr = traces_by_uid("deepseek", ds_ids)
    glm_tr = traces_by_uid("glm", all_glm_lookups, file_filter="main")
    log.info("loaded %d ds, %d glm traces", len(ds_tr), len(glm_tr))

    with (OUT / "pilot_judged.jsonl").open("a", encoding="utf-8") as fh:
        for ds_id, glm_id in pairs:
            for t in (ds_tr.get(ds_id), glm_tr.get(glm_id) or glm_tr.get(ds_id)):
                if t is None or t.uid in done:
                    continue
                try:
                    content, usage = client.call(rubric.SYSTEM, rubric.build_user_prompt(t),
                                                  max_tokens=args.max_tokens)
                    d = client.parse_json(content)
                except client.JudgeError as e:
                    log.warning("judge failed %s/%s: %s", t.dataset, t.uid, e)
                    continue
                problems = rubric.validate(d)
                row = {"pair_ds": ds_id, "pair_glm": glm_id, "dataset": t.dataset,
                       "teacher": t.teacher, "uid": t.uid, "reasoning_chars": t.reasoning_chars,
                       "crit_density_1k": round(crit_density_per_1k(t.reasoning), 2),
                       "usage": usage, "qc": problems,
                       **{k: d.get(k) for k in rubric.EXPECTED_FIELDS}}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                log.info("%-9s ccr=%s coupling=%s depth=%s ans_change=%s topo=%s density=%.1f",
                         t.dataset, d.get("ccr_closure"), d.get("monitoring_control_coupling"),
                         d.get("causal_depth_of_critique"), d.get("answer_change"),
                         d.get("reasoning_topology"), row["crit_density_1k"])

    # paired summary
    rows = list(iter_jsonl(OUT / "pilot_judged.jsonl"))
    by_team: dict[str, list[int]] = {}
    for r in rows:
        if isinstance(r.get("ccr_closure"), int):
            by_team.setdefault(r["dataset"], []).append(r["ccr_closure"])
    summary = {ds: {"n": len(v), "mean_ccr": round(sum(v) / len(v), 2)} for ds, v in by_team.items()}
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("PILOT SUMMARY (mean CCR closure by teacher): %s", summary)


if __name__ == "__main__":
    main()

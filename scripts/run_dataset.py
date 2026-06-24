#!/usr/bin/env python3
"""Per-dataset judge runner: sample N traces from any registered dataset,
run the consolidated Kimi rubric, write resumable JSONL + summary.

Usage:
    python scripts/run_dataset.py --dataset glm --sample 200 --max-tokens 16000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_audit.adapters import DATASETS, iter_traces  # noqa: E402
from distill_audit.judge import client, rubric  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT / "run_dataset.log"),
    ],
)
log = logging.getLogger("run_dataset")

MIN_REASONING_CHARS = 200

_CRIT = re.compile(
    r"\b(wait|but|however|actually|reconsider|check|verify|mistake|wrong|"
    r"recompute|let me|hmm)\b|不对|等等|然而|重新|检查|验证|错误|修正",
    re.IGNORECASE,
)


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


def sample_traces(name: str, n: int, seed: int = 42) -> list:
    """Load all traces, skip too-short, then stratified/random sample N."""
    log.info("streaming traces from %s (filtering reasoning_chars >= %d) ...",
             name, MIN_REASONING_CHARS)
    all_traces = []
    skipped_short = 0
    for t in iter_traces(name):
        if t.reasoning_chars < MIN_REASONING_CHARS:
            skipped_short += 1
            continue
        all_traces.append(t)

    log.info("loaded %d eligible traces (%d skipped: reasoning too short)",
             len(all_traces), skipped_short)

    if len(all_traces) <= n:
        log.info("dataset has <= %d eligible traces; using all %d", n, len(all_traces))
        return all_traces

    # Stratified sampling by domain if domains exist
    rng = random.Random(seed)
    domains: dict[str, list] = {}
    for t in all_traces:
        key = t.domain or "__none__"
        domains.setdefault(key, []).append(t)

    if len(domains) > 1:
        log.info("stratified sampling across %d domains", len(domains))
        sampled = []
        # Proportional allocation
        for dom, traces in domains.items():
            k = max(1, round(n * len(traces) / len(all_traces)))
            k = min(k, len(traces))
            sampled.extend(rng.sample(traces, k))
        # Trim or fill to exactly n
        if len(sampled) > n:
            sampled = rng.sample(sampled, n)
        elif len(sampled) < n:
            remaining = [t for t in all_traces if t not in set(sampled)]
            sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))
        return sampled
    else:
        return rng.sample(all_traces, n)


def build_summary(judged_path: Path) -> dict:
    """Compute aggregate statistics from the judged JSONL."""
    rows = list(iter_jsonl(judged_path))
    if not rows:
        return {"n_judged": 0, "n_skipped": 0, "n_errors": 0}

    n_judged = sum(1 for r in rows if r.get("status") == "judged")
    n_skipped = sum(1 for r in rows if r.get("status") == "skipped")
    n_errors = sum(1 for r in rows if r.get("status") == "error")

    scored = [r for r in rows if r.get("status") == "judged"]

    def safe_values(key):
        return [r[key] for r in scored if isinstance(r.get(key), (int, float))]

    def stats_for(key):
        vals = safe_values(key)
        if not vals:
            return {"mean": None, "median": None, "std": None, "n": 0}
        return {
            "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    # Topology distribution
    topo_counts = Counter(r.get("reasoning_topology") for r in scored
                          if r.get("reasoning_topology"))

    # Failure type distribution
    failure_counts = Counter(r.get("failure_type") for r in scored
                             if r.get("failure_type"))

    # Critique density
    density_vals = [r["crit_density_1k"] for r in scored
                    if isinstance(r.get("crit_density_1k"), (int, float))]

    summary = {
        "n_judged": n_judged,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "ccr_closure": stats_for("ccr_closure"),
        "monitoring_control_coupling": stats_for("monitoring_control_coupling"),
        "causal_depth_of_critique": stats_for("causal_depth_of_critique"),
        "topology_distribution": dict(topo_counts.most_common()),
        "mean_crit_density_1k": (
            round(statistics.mean(density_vals), 3) if density_vals else None
        ),
        "failure_type_distribution": dict(failure_counts.most_common()),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Kimi judge on a sampled dataset")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                    help="Logical dataset name from adapters.DATASETS")
    ap.add_argument("--sample", type=int, default=100,
                    help="Number of traces to sample (default: 100)")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="Max tokens for Kimi judge response (default: 16000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling (default: 42)")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds to wait between API calls to avoid rate limits (default: 2.0)")
    args = ap.parse_args()

    judged_path = OUT / f"{args.dataset}_judged.jsonl"
    summary_path = OUT / f"{args.dataset}_summary.json"

    # Resume: load already-judged UIDs
    done = set()
    for r in iter_jsonl(judged_path):
        uid = r.get("uid")
        if uid:
            done.add(uid)
    log.info("resume: %d traces already in %s", len(done), judged_path.name)

    # Sample traces
    traces = sample_traces(args.dataset, args.sample, seed=args.seed)
    log.info("sampled %d traces; %d already judged -> %d to judge",
             len(traces), sum(1 for t in traces if t.uid in done),
             sum(1 for t in traces if t.uid not in done))

    n_judged_this_run = 0
    n_skipped = 0
    n_errors = 0

    with judged_path.open("a", encoding="utf-8") as fh:
        for i, t in enumerate(traces, 1):
            if t.uid in done:
                continue

            # Skip traces with too-short reasoning (double-check after sampling)
            if t.reasoning_chars < MIN_REASONING_CHARS:
                row = {
                    "status": "skipped", "reason": "reasoning_too_short",
                    "dataset": t.dataset, "teacher": t.teacher, "uid": t.uid,
                    "reasoning_chars": t.reasoning_chars,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done.add(t.uid)
                n_skipped += 1
                log.info("[%d/%d] SKIP %s (reasoning_chars=%d)",
                         i, len(traces), t.uid, t.reasoning_chars)
                continue

            # Judge the trace
            try:
                user_prompt = rubric.build_user_prompt(t)
                content, usage = client.call(
                    rubric.SYSTEM, user_prompt, max_tokens=args.max_tokens,
                )
                d = client.parse_json(content)
            except client.JudgeError as e:
                row = {
                    "status": "error", "error": str(e),
                    "dataset": t.dataset, "teacher": t.teacher, "uid": t.uid,
                    "reasoning_chars": t.reasoning_chars,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done.add(t.uid)
                n_errors += 1
                log.warning("[%d/%d] ERROR %s/%s: %s",
                            i, len(traces), t.dataset, t.uid, e)
                # Extra backoff after errors (likely rate-limited)
                if args.delay > 0:
                    time.sleep(args.delay * 5)
                continue

            problems = rubric.validate(d)
            density = round(crit_density_per_1k(t.reasoning), 2)

            row = {
                "status": "judged",
                "dataset": t.dataset,
                "teacher": t.teacher,
                "uid": t.uid,
                "domain": t.domain,
                "reasoning_chars": t.reasoning_chars,
                "crit_density_1k": density,
                "usage": usage,
                "qc": problems,
                **{k: d.get(k) for k in rubric.EXPECTED_FIELDS},
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(t.uid)
            n_judged_this_run += 1

            log.info(
                "[%d/%d] %-9s uid=%-12s ccr=%s coupling=%s depth=%s "
                "topo=%-5s density=%.1f qc=%s",
                i, len(traces), t.dataset, t.uid[:12],
                d.get("ccr_closure"), d.get("monitoring_control_coupling"),
                d.get("causal_depth_of_critique"), d.get("reasoning_topology"),
                density, problems or "ok",
            )

            # Rate-limit delay between API calls
            if args.delay > 0:
                time.sleep(args.delay)

    log.info("run complete: judged=%d skipped=%d errors=%d",
             n_judged_this_run, n_skipped, n_errors)

    # Build and write summary from ALL rows in the file (including prior runs)
    summary = build_summary(judged_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("summary written to %s: %s", summary_path.name,
             json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

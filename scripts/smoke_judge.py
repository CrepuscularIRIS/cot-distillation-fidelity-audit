#!/usr/bin/env python3
"""Smoke test: run the consolidated judge on a few real traces, end to end.

Validates adapters -> rubric -> Kimi client -> JSON parse -> schema validate.
Not a measurement; just proves the pipeline works and prints what the judge emits.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_audit.adapters import iter_traces  # noqa: E402
from distill_audit.judge import client, rubric  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smoke")

CORE = ["critique_present", "ccr_closure", "monitoring_control_coupling",
        "causal_depth_of_critique", "answer_change", "verification_independence",
        "failure_type", "reasoning_topology"]


def pick(name: str, n: int, lo: int, hi: int):
    out = []
    for t in iter_traces(name, limit=4000):
        if lo <= t.reasoning_chars <= hi:
            out.append(t)
            if len(out) >= n:
                break
    return out


def main() -> None:
    traces = pick("deepseek", 2, 1500, 12000) + pick("glm", 1, 1500, 12000)
    log.info("selected %d traces", len(traces))
    for i, t in enumerate(traces):
        log.info("--- trace %d | %s/%s | reasoning=%d chars ---",
                 i, t.dataset, t.teacher, t.reasoning_chars)
        user = rubric.build_user_prompt(t)
        content, usage = client.call(rubric.SYSTEM, user, max_tokens=2000)
        try:
            d = client.parse_json(content)
        except client.JudgeError as e:
            log.error("parse failed: %s | content head: %s", e, content[:200])
            continue
        problems = rubric.validate(d)
        log.info("scores: %s", {k: d.get(k) for k in CORE})
        log.info("evidence: %s", str(d.get("evidence"))[:240])
        log.info("rationale: %s", str(d.get("rationale"))[:240])
        log.info("usage: %s | QC problems: %s", usage, problems or "clean")


if __name__ == "__main__":
    main()

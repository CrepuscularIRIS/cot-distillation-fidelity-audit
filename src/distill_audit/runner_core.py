"""Concurrent judging engine: thread-pooled Kimi judge calls, resumable,
thread-safe single-file writes. One output file per job -> no write collisions.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .adapters import iter_traces
from .judge import client, rubric
from .schema import Trace

log = logging.getLogger("runner")

_CRIT = re.compile(
    r"\b(wait|but|however|actually|reconsider|recheck|check|verify|mistake|wrong|"
    r"recompute|let me|hmm|oops|incorrect)\b|不对|等等|然而|重新|检查|验证|错误|修正",
    re.IGNORECASE,
)


def crit_density_per_1k(text: str) -> float:
    toks = max(len(text) / 4.0, 1.0)  # ~4 chars/token proxy
    return round(1000.0 * len(_CRIT.findall(text)) / toks, 2)


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def sample_traces(name: str, n: int, *, file_filter: str | None = None,
                  scan_limit: int | None = None, min_chars: int = 200,
                  seed: int = 42) -> list[Trace]:
    """Reservoir-sample n eligible traces (reasoning >= min_chars, unique uid)."""
    rng = random.Random(seed)
    pool: list[Trace] = []
    seen: set[str] = set()
    scanned = 0
    for t in iter_traces(name, file_filter=file_filter):
        scanned += 1
        if scan_limit and scanned > scan_limit:
            break
        if t.reasoning_chars < min_chars or t.uid in seen:
            continue
        seen.add(t.uid)
        i = len(seen) - 1
        if len(pool) < n:
            pool.append(t)
        else:
            j = rng.randint(0, i)
            if j < n:
                pool[j] = t
    return pool


def _judge(trace: Trace, extra: dict, max_tokens: int) -> dict:
    user = rubric.build_user_prompt(trace)
    try:
        content, usage = client.call(rubric.SYSTEM, user, max_tokens=max_tokens)
        d = client.parse_json(content)
        status, qc = "judged", rubric.validate(d)
    except client.JudgeError as e:
        d, usage, status, qc = {}, {}, "error", [str(e)[:120]]
    row = {
        **extra,
        "dataset": trace.dataset, "teacher": trace.teacher, "uid": trace.uid,
        "reasoning_chars": trace.reasoning_chars,
        "crit_density_1k": crit_density_per_1k(trace.reasoning),
        "domain": trace.domain, "status": status, "qc": qc, "usage": usage,
        **{k: d.get(k) for k in rubric.EXPECTED_FIELDS},
    }
    return row


def judge_many(items: list[tuple[Trace, dict]], out_path: Path, *,
               max_tokens: int = 16000, concurrency: int = 8) -> int:
    """Judge (trace, extra) items concurrently into out_path. Returns n new rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r.get("uid") for r in _iter_jsonl(out_path) if r.get("status") != "error"}
    todo = [(t, x) for (t, x) in items if t.uid not in done]
    log.info("%s: %d items, %d already done, %d to judge (conc=%d)",
             out_path.name, len(items), len(items) - len(todo), len(todo), concurrency)
    if not todo:
        return 0

    lock = threading.Lock()
    written = 0
    with out_path.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_judge, t, x, max_tokens): t for (t, x) in todo}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                log.warning("worker crashed: %s", e)
                continue
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                written += 1
                if written % 10 == 0 or written == len(todo):
                    log.info("  %s: %d/%d written (last: %s ccr=%s topo=%s)",
                             out_path.name, written, len(todo), row["dataset"],
                             row.get("ccr_closure"), row.get("reasoning_topology"))
    return written

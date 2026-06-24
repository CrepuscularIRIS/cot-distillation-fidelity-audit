"""Dataset adapters: map each raw record layout into the unified Trace.

Registry-based so the harness only ever sees Trace objects. Three record layouts
verified on disk (2026-06-24):
- output_thinking: GLM / DeepSeek — <think>..</think> + answer inside `output`.
- trace_inversion: Claude TraceInversion — reasoning in `inverted_reasoning`,
  answer in `output`.
- gemini: original_input / model_thoughts / model_response.
- nohurry: problem / thinking / solution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Callable

from .extract import split_thinking
from .schema import Trace

HUB = Path("/data/huggingface/hub")

AdapterFn = Callable[[dict, str], Trace]
ADAPTERS: dict[str, AdapterFn] = {}


def register(name: str) -> Callable[[AdapterFn], AdapterFn]:
    def deco(fn: AdapterFn) -> AdapterFn:
        ADAPTERS[name] = fn
        return fn
    return deco


@register("output_thinking")
def _output_thinking(rec: dict, dataset: str) -> Trace:
    reasoning, answer = split_thinking(rec.get("output") or "")
    meta = rec.get("meta") or {}
    return Trace(
        dataset=dataset, teacher=str(meta.get("teacher_model", "?")),
        problem=rec.get("input") or "", reasoning=reasoning, answer=answer,
        domain=str(rec.get("domain", "")), uid=str(rec.get("id", "")), meta=meta,
    )


@register("trace_inversion")
def _trace_inversion(rec: dict, dataset: str) -> Trace:
    inv = rec.get("inverted_reasoning") or ""
    reasoning, _ = split_thinking(inv)
    if not reasoning:
        reasoning = inv.strip()
    meta = rec.get("meta") or {}
    return Trace(
        dataset=dataset, teacher=str(meta.get("teacher_model", "claude")),
        problem=rec.get("input") or "", reasoning=reasoning, answer=rec.get("output") or "",
        domain=str(rec.get("domain", "")), uid=str(rec.get("id", "")), meta=meta,
    )


@register("gemini")
def _gemini(rec: dict, dataset: str) -> Trace:
    oi = rec.get("original_input") or {}
    # original_input is a dict with keys: domain, concept, difficulty, text
    if isinstance(oi, dict):
        problem = oi.get("text") or ""
        domain = str(oi.get("domain", ""))
        difficulty = str(oi.get("difficulty", ""))
    else:
        problem = str(oi)
        domain = ""
        difficulty = ""
    # No id/hash field in Gemini records; derive UID from problem text
    uid = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:16]
    return Trace(
        dataset=dataset, teacher="gemini-3.1-pro",
        problem=problem, reasoning=rec.get("model_thoughts") or "",
        answer=rec.get("model_response") or "", domain=domain, difficulty=difficulty,
        uid=uid,
        meta={k: v for k, v in (oi if isinstance(oi, dict) else {}).items()
              if k in ("difficulty", "domain", "concept")},
    )


@register("nohurry")
def _nohurry(rec: dict, dataset: str) -> Trace:
    return Trace(
        dataset=dataset, teacher="claude-opus-4.6",
        problem=rec.get("problem") or "", reasoning=rec.get("thinking") or "",
        answer=rec.get("solution") or "", domain=str(rec.get("category", "")),
        difficulty=str(rec.get("difficulty", "")), uid=str(rec.get("hash", "")),
    )


def _first(msgs: list, role: str) -> str:
    for m in msgs:
        if m.get("role") == role:
            c = m.get("content")
            return c if isinstance(c, str) else ""
    return ""


@register("messages_think")
def _messages_think(rec: dict, dataset: str) -> Trace:
    # angrygiraffe: multi-turn messages; first assistant turn carries <think>..</think>
    msgs = rec.get("messages") or []
    problem = _first(msgs, "user")
    asst = _first(msgs, "assistant")
    reasoning, answer = split_thinking(asst)
    uid = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:16]
    return Trace(
        dataset=dataset, teacher=str(rec.get("model", "claude")),
        problem=problem, reasoning=reasoning, answer=answer,
        domain=str(rec.get("category", "")), uid=uid,
        meta={"model": rec.get("model"), "category": rec.get("category")},
    )


@register("messages_inline")
def _messages_inline(rec: dict, dataset: str) -> Trace:
    # Roman claude-opus-4.6-10000x: messages + metadata; NO <think>. Reasoning is
    # inline in the assistant answer (direct-answer baseline).
    msgs = rec.get("messages") or []
    md = rec.get("metadata") or {}
    problem = _first(msgs, "user")
    asst = _first(msgs, "assistant")
    uid = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:16]
    return Trace(
        dataset=dataset, teacher=str(md.get("model", "claude-opus-4.6")),
        problem=problem, reasoning=asst, answer=asst,
        domain=str(md.get("category", "")), difficulty=str(md.get("difficulty", "")),
        uid=uid, meta=md,
    )


# logical name -> (hub repo dir, adapter)
DATASETS: dict[str, tuple[str, str]] = {
    "glm": ("Jackrong--GLM-5.1-Reasoning-1M-Cleaned", "output_thinking"),
    "deepseek": ("Jackrong--DeepSeek-V4-Distill-8000x", "output_thinking"),
    "claude46_ti": ("Jackrong--Claude-opus-4.6-TraceInversion-9000x", "trace_inversion"),
    "claude47_ti": ("Jackrong--Claude-opus-4.7-TraceInversion-5000x", "trace_inversion"),
    "gemini": ("Roman1111111--gemini-3.1-pro-hard-high-reasoning", "gemini"),
    "nohurry_opus": ("nohurry--Opus-4.6-Reasoning-3000x-filtered", "nohurry"),
    "kimi": ("Jackrong--Kimi-K2.5-Reasoning-1M-Cleaned", "output_thinking"),
    "qwen": ("Jackrong--Qwen3.5-reasoning-700x", "output_thinking"),
    "angrygiraffe": ("angrygiraffe--claude-opus-4.6-4.7-reasoning-8.7k", "messages_think"),
    "roman_claude": ("Roman1111111--claude-opus-4.6-10000x", "messages_inline"),
}


def snapshot_files(repo: str, glob: str = "*.jsonl") -> list[Path]:
    base = HUB / f"datasets--{repo}" / "snapshots"
    return sorted(set(base.glob(f"*/{glob}")) | set(base.glob(f"*/**/{glob}")))


def iter_traces(name: str, *, limit: int | None = None,
                file_filter: str | None = None) -> Iterator[Trace]:
    """Yield Traces for a logical dataset name (optionally one file / capped)."""
    repo, adapter = DATASETS[name]
    fn = ADAPTERS[adapter]
    n = 0
    for path in snapshot_files(repo):
        if file_filter and file_filter not in path.name:
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield fn(rec, name)
                n += 1
                if limit and n >= limit:
                    return

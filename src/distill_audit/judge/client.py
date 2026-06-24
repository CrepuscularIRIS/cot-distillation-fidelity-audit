"""Kimi-for-coding execution-judge client (OpenAI-compatible, via local proxy).

Zero-dependency (urllib) so the harness runs without an env install. The proxy
at KIMI_BASE_URL injects OAuth, so no real API key is needed client-side.

Hard model constraint (verified): kimi-for-coding only accepts temperature=1.
It is a reasoning model — the answer is in message.content, hidden reasoning in
message.reasoning_content (which we ignore for scoring but log token usage for).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

log = logging.getLogger("judge.client")

BASE = os.environ.get("KIMI_BASE_URL", "http://localhost:4242").rstrip("/")
MODEL = os.environ.get("KIMI_MODEL", "kimi-for-coding")
ENDPOINT = f"{BASE}/v1/chat/completions"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)


class JudgeError(RuntimeError):
    pass


def call(system: str, user: str, *, max_tokens: int = 1800, retries: int = 6,
         timeout: int = 240) -> tuple[str, dict]:
    """Single chat completion. Returns (content, usage). Raises JudgeError."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": 1,  # forced by the model
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": min(max_tokens // 2, 8000)},
    }
    data = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                ENDPOINT, data=data, method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer proxy"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            if "error" in d:
                raise JudgeError(str(d["error"]))
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                raise JudgeError("empty content")
            return content, d.get("usage", {})
        except Exception as e:  # noqa: BLE001 - retry on any transport/model error
            last_err = e
            is_429 = "429" in str(e)
            # Longer backoff for rate limits: 5, 10, 20, 40, 80, 160s
            # Shorter for other errors: 2, 4, 8, 16, 32, 64s
            base = 5 if is_429 else 2
            wait = base * (2 ** attempt)
            log.warning("judge call failed (attempt %d/%d): %s; retry in %ds",
                        attempt + 1, retries, e, wait)
            time.sleep(wait)
    raise JudgeError(f"judge call failed after {retries} attempts: {last_err}")


def parse_json(content: str) -> dict:
    """Extract a JSON object from judge content (tolerates ```json fences)."""
    m = _FENCE.search(content)
    candidate = m.group(1) if m else content
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    m2 = _OBJ.search(content)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    cleaned = content.strip()
    if cleaned.startswith("{"):
        cleaned = cleaned.rsplit("}", 1)
        if len(cleaned) >= 2:
            try:
                return json.loads(cleaned[0] + "}")
            except json.JSONDecodeError:
                pass
    raise JudgeError(f"no JSON object in content: {content[:200]!r}")

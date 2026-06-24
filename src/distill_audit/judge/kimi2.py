"""Two-key Kimi client: calls api.kimi.com/coding directly with KEY1+KEY2 rotated,
replicating the proxy's X-Msh headers. Built for LOW concurrency (the high-concurrency
single-key path drained quota fast). Per-key cooldown on 429 so a dead key is skipped.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import threading
import time
import urllib.request
from pathlib import Path

from .client import JudgeError, parse_json  # reuse JSON parsing  # noqa: F401

log = logging.getLogger("judge.kimi2")

BASE = "https://api.kimi.com/coding/v1/chat/completions"
_VER = "1.40.0"


def _find_env() -> Path:
    """Locate the .env holding API keys. Order: $DISTILL_AUDIT_ENV, then the first
    .env found walking up from CWD and from this file. Keeps the repo portable —
    no hardcoded home path."""
    explicit = os.environ.get("DISTILL_AUDIT_ENV")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    seen = []
    for base in [Path.cwd(), Path(__file__).resolve()]:
        for parent in [base, *base.parents]:
            cand = parent / ".env"
            if cand not in seen:
                seen.append(cand)
                if cand.exists():
                    return cand
    return Path.cwd() / ".env"  # may not exist; _load_cfg degrades gracefully


def _load_cfg() -> dict:
    cfg = {}
    env = _find_env()
    if not env.exists():
        log.warning("no .env found (set DISTILL_AUDIT_ENV); judge backends will be unauthenticated")
        return cfg
    for line in env.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    return cfg


_cfg = _load_cfg()
MODEL = _cfg.get("KIMI_MODEL", "kimi-for-coding")
_KEYS = [_cfg.get("KIMI_API_KEY1"), _cfg.get("KIMI_API_KEY2")]
_KEYS = [k for k in _KEYS if k]
_DEV = Path.home() / ".kimi" / "device_id"
_DEVICE_ID = _DEV.read_text(encoding="utf-8").strip() if _DEV.exists() else ""

# key index -> epoch until which the key is cooling (after a 429)
_cool: dict[int, float] = {}
_lock = threading.Lock()
_rr = [0]


def _headers(key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": f"KimiCLI/{_VER}",
        "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": _VER,
        "X-Msh-Device-Name": (platform.node() or socket.gethostname())[:64],
        "X-Msh-Device-Model": platform.machine(),
        "X-Msh-Os-Version": platform.version()[:64],
        "X-Msh-Device-Id": _DEVICE_ID,
    }


def _pick_key(now: float) -> int:
    """Round-robin, skipping keys still cooling from a recent 429."""
    with _lock:
        for _ in range(len(_KEYS)):
            i = _rr[0] % len(_KEYS)
            _rr[0] += 1
            if _cool.get(i, 0) <= now:
                return i
        # all cooling -> return the one cooling for the shortest time
        return min(range(len(_KEYS)), key=lambda i: _cool.get(i, 0))


def n_keys() -> int:
    return len(_KEYS)


def call(system: str, user: str, *, max_tokens: int = 16000, retries: int = 8,
         timeout: int = 240) -> tuple[str, dict]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False, "temperature": 1, "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": min(max_tokens // 2, 8000)},
    }
    data = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        now = time.time()
        ki = _pick_key(now)
        try:
            req = urllib.request.Request(BASE, data=data, method="POST", headers=_headers(_KEYS[ki]))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d:
                raise JudgeError(str(d["error"]))
            content = d["choices"][0]["message"].get("content") or ""
            if not content.strip():
                raise JudgeError("empty content")
            return content, d.get("usage", {})
        except Exception as e:  # noqa: BLE001
            last = e
            is429 = "429" in str(e) or "limit" in str(e).lower()
            if is429:
                with _lock:
                    _cool[ki] = time.time() + 90  # cool this key 90s
                wait = 6
            else:
                wait = 3 * (attempt + 1)
            log.warning("kimi2 key%d fail (try %d/%d): %s; wait %ds", ki + 1, attempt + 1, retries, str(e)[:80], wait)
            time.sleep(wait)
    raise JudgeError(f"kimi2 failed after {retries}: {last}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"keys loaded: {len(_KEYS)} | device_id set: {bool(_DEVICE_ID)} | model: {MODEL}")
    for i, k in enumerate(_KEYS):
        try:
            req = urllib.request.Request(
                BASE, data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Reply OK"}],
                                       "max_tokens": 200, "temperature": 1, "stream": False,
                                       "thinking": {"type": "enabled", "budget_tokens": 100}}).encode(),
                method="POST", headers=_headers(k))
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.loads(r.read())
            c = d["choices"][0]["message"].get("content") if "error" not in d else None
            print(f"  KEY{i+1}: {'OK ' + repr(c)[:30] if c else 'ERR ' + str(d.get('error'))[:120]} | usage {d.get('usage')}")
        except Exception as e:
            print(f"  KEY{i+1}: FAILED {e}")

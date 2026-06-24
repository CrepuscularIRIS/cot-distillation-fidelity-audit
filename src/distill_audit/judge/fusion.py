"""Multi-backend judge fusion: Kimi (2 keys, rotated) + DeepSeek-V4-Flash (OpenCode)
+ MiniMax-M3. Each backend is an independent endpoint/account, so they run in
parallel at LOW per-backend concurrency (the earlier single-key high-concurrency
path drained quota). Used as a JUDGE PANEL -> per-trace scores from each model.

Self-bias note: deepseek_flash judging the `deepseek` dataset and kimi judging the
`kimi` dataset are teacher-judging-own-student; flag/exclude downstream.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from . import kimi2
from .client import JudgeError, parse_json  # noqa: F401  (re-exported)

log = logging.getLogger("judge.fusion")
_cfg = kimi2._load_cfg()


def _post(url: str, headers: dict, payload: dict, timeout: int = 240) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _content(d: dict) -> tuple[str, dict]:
    if "error" in d:
        raise JudgeError(str(d["error"])[:160])
    c = d["choices"][0]["message"].get("content") or ""
    if not c.strip():
        raise JudgeError("empty content")
    return c, d.get("usage", {})


# ---- backend: Kimi (two keys, rotated, per-key cooldown) ----
def kimi_call(system: str, user: str, *, max_tokens: int = 16000) -> tuple[str, dict]:
    return kimi2.call(system, user, max_tokens=max_tokens)


# ---- backend: DeepSeek-V4-Flash via OpenCode zen gateway ----
_OC_URL = _cfg["OpenCode-URL"]
_OC_KEY = _cfg["OpenCode-APIKEY"]
_OC_MODEL = _cfg["OpenCode-model"]
_OC_HDR = {"Authorization": f"Bearer {_OC_KEY}", "Content-Type": "application/json",
           "User-Agent": "opencode-cli/1.0", "HTTP-Referer": "https://opencode.ai"}


def deepseek_call(system: str, user: str, *, max_tokens: int = 14000, retries: int = 5) -> tuple[str, dict]:
    payload = {"model": _OC_MODEL,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "stream": False, "max_tokens": max_tokens, "temperature": 0.3}
    last = None
    for a in range(retries):
        try:
            return _content(_post(_OC_URL, _OC_HDR, payload))
        except Exception as e:  # noqa: BLE001
            last = e
            wait = (8 if ("429" in str(e) or "limit" in str(e).lower()) else 3) * (a + 1)
            log.warning("deepseek_flash fail (try %d): %s; wait %ds", a + 1, str(e)[:80], wait)
            time.sleep(wait)
    raise JudgeError(f"deepseek_flash failed: {last}")


# ---- backend: MiniMax-M3 ----
_MM_KEY = _cfg["MINIMAX_API_KEY"]
_MM_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
_MM_HDR = {"Authorization": f"Bearer {_MM_KEY}", "Content-Type": "application/json"}


def minimax_call(system: str, user: str, *, max_tokens: int = 14000, retries: int = 5) -> tuple[str, dict]:
    payload = {"model": "MiniMax-M3",
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "max_tokens": max_tokens}
    last = None
    for a in range(retries):
        try:
            return _content(_post(_MM_URL, _MM_HDR, payload))
        except Exception as e:  # noqa: BLE001
            last = e
            wait = (8 if ("429" in str(e) or "limit" in str(e).lower()) else 3) * (a + 1)
            log.warning("minimax_m3 fail (try %d): %s; wait %ds", a + 1, str(e)[:80], wait)
            time.sleep(wait)
    raise JudgeError(f"minimax_m3 failed: {last}")


# ---- backend: Codex / GPT-5.4 via codex-switch proxy ----
_CX_URL = "http://localhost:4141/v1/chat/completions"
_CX_HDR = {"Authorization": "Bearer codex-switch-proxy", "Content-Type": "application/json"}


def codex_call(system: str, user: str, *, max_tokens: int = 14000, retries: int = 5) -> tuple[str, dict]:
    payload = {"model": "gpt-5.4",
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "stream": False, "max_tokens": max_tokens}
    last = None
    for a in range(retries):
        try:
            return _content(_post(_CX_URL, _CX_HDR, payload, timeout=300))
        except Exception as e:  # noqa: BLE001
            last = e
            wait = (8 if ("429" in str(e) or "limit" in str(e).lower()) else 3) * (a + 1)
            log.warning("codex fail (try %d): %s; wait %ds", a + 1, str(e)[:80], wait)
            time.sleep(wait)
    raise JudgeError(f"codex failed: {last}")


BACKENDS = {
    "kimi": kimi_call,
    "deepseek_flash": deepseek_call,
    "minimax_m3": minimax_call,
    "codex": codex_call,
}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from distill_audit.judge import rubric  # noqa: E402
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    batch = json.loads((Path(__file__).resolve().parents[3] / "outputs/batches/paired_001.json").read_text())
    t = batch[0]
    user = (f"PROBLEM:\n{t['problem'][:1500]}\n\nMODEL CoT:\n{t['reasoning']}\n\n"
            f"FINAL ANSWER (context only):\n{t['answer'][:1200]}\n\nReturn the JSON object now.")
    for name, fn in BACKENDS.items():
        try:
            content, usage = fn(rubric.SYSTEM, user, max_tokens=12000)
            d = parse_json(content)
            print(f"  {name:14s} OK | ccr={d.get('ccr_closure')} topo={d.get('reasoning_topology')} "
                  f"coupling={d.get('monitoring_control_coupling')} | usage_completion={usage.get('completion_tokens')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:14s} FAIL: {str(e)[:140]}")

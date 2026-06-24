#!/usr/bin/env python3
"""Diagnose how many tokens kimi-for-coding needs to finish the rubric JSON."""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_audit.adapters import iter_traces
from distill_audit.judge import rubric

t = next(t for t in iter_traces("deepseek", limit=50) if 1500 <= t.reasoning_chars <= 8000)
print(f"trace reasoning chars: {t.reasoning_chars}", flush=True)

for mt in (4000, 8000, 16000):
    payload = {"model": "kimi-for-coding", "stream": False, "temperature": 1, "max_tokens": mt,
               "messages": [{"role": "system", "content": rubric.SYSTEM},
                            {"role": "user", "content": rubric.build_user_prompt(t)}]}
    req = urllib.request.Request("http://localhost:4242/v1/chat/completions",
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer proxy"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=400).read())
    except Exception as e:
        print(f"max_tokens={mt}: ERROR {e}", flush=True)
        continue
    if "error" in d:
        print(f"max_tokens={mt}: API error {d['error']}", flush=True)
        continue
    ch = d["choices"][0]; m = ch["message"]
    content = m.get("content") or ""
    print(f"max_tokens={mt}: finish={ch.get('finish_reason')} "
          f"len(content)={len(content)} len(reasoning)={len(m.get('reasoning_content') or '')} "
          f"usage={d.get('usage')}", flush=True)
    if content.strip():
        print(f"  content head: {content[:200]}", flush=True)
        break

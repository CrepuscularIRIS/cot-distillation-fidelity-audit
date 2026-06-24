"""Reasoning/answer extraction from raw distilled outputs.

Jackrong-family datasets (GLM / DeepSeek / Claude-TraceInversion) wrap the chain
of thought in <think>..</think> or <thinking>..</thinking> inside the `output`
field, followed by the final answer. We split the two so the judge can score the
reasoning while the answer is available for answer-change checks.
"""

from __future__ import annotations

import re

_THINK = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def split_thinking(output: str) -> tuple[str, str]:
    """Return (reasoning, answer).

    If no think block is present, reasoning is empty and the whole text is
    treated as the answer (so the trace is still usable, just flagged as having
    no externalized CoT).
    """
    if not output:
        return "", ""
    m = _THINK.search(output)
    if not m:
        return "", output.strip()
    reasoning = m.group(1).strip()
    answer = output[m.end():].strip()
    return reasoning, answer

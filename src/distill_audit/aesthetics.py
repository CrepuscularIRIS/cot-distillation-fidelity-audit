"""Gemini advanced/aesthetic + first-principles metrics (mindset/Gemini/
advanced_metrics_exploration.md + first_principles_perspectives.md). Pure text/graph,
zero API.

  CE   Compression Efficiency   = logical-skeleton tokens / total tokens (MDL aesthetic)
  NRR  Noise Reduction Ratio    = defensive/filler token proportion (lower = "high-cold" confidence)
  BSD  Branching Sparsity Degree= mean out-degree of the CoT dependency graph
  ICI  Information Concentration= Gini coefficient of PageRank over the dependency graph (hub-ness / low-rank)
  (LSI logical symmetry = FUTURE WORK per the doc — needs reverse-problem generation; not computed.)

  MI-collapse (first_principles angle 1) is computed at the DATASET level from the LLM
  judgments (I(critique_present ; answer_changed)) in scripts/run_aesthetics.py.
"""

from __future__ import annotations

import math
import re

from .structural import BACKREF, BRANCH, CAUSAL, MERGE, paragraphs

_WORD = re.compile(r"\b\w+\b")
_LOGIC = re.compile(
    r"(?i)\b(if|then|therefore|thus|hence|so|because|since|assume|suppose|let|given|"
    r"implies|follows|provided|otherwise|whereas|consequently|iff|where|define)\b"
    r"|因为|所以|因此|假[设如]|令|设|由此|则|当且仅当|定义|推出",
)
_NOISE = re.compile(
    r"(?i)(as an ai|i (might|may|could) be wrong|i apologize|i'm sorry|sorry|"
    r"i hope this helps|please note|it's worth noting|just to clarify|"
    r"i'm just|to be honest|if i'm not mistaken|disclaimer|of course|certainly!)"
    r"|抱歉|对不起|需要注意|值得注意|希望(这)?对你有帮助|当然",
)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text or "")


def compute_ce(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return float("nan")
    return round(len(_LOGIC.findall(text or "")) / len(toks), 4)


def compute_nrr(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return float("nan")
    # noise proportion (per 1k tokens); low = confident/clean
    return round(1000.0 * len(_NOISE.findall(text or "")) / len(toks), 3)


def _dep_graph(paras: list[str]) -> list[list[int]]:
    """Directed edges (out-adjacency). Causal/branch/merge → prev paragraph;
    back-reference → an earlier paragraph (resolve 'step N' else default lag 3)."""
    n = len(paras)
    adj: list[list[int]] = [[] for _ in range(n)]
    for j, para in enumerate(paras):
        if j > 0 and (CAUSAL.search(para) or BRANCH.search(para) or MERGE.search(para)):
            adj[j - 1].append(j)
        if BACKREF.search(para):
            m = re.search(r"step (\d+)", para, re.IGNORECASE)
            tgt = int(m.group(1)) if m else max(0, j - 3)
            if 0 <= tgt < j:
                adj[tgt].append(j)
    return adj


def compute_bsd(text: str) -> float:
    paras = paragraphs(text)
    if len(paras) < 3:
        return float("nan")
    adj = _dep_graph(paras)
    inter = adj[:-1] if len(adj) > 1 else adj  # intermediate nodes
    degs = [len(a) for a in inter]
    return round(sum(degs) / len(degs), 3) if degs else 0.0


def _pagerank(adj: list[list[int]], damping: float = 0.85, iters: int = 40) -> list[float]:
    n = len(adj)
    if n == 0:
        return []
    pr = [1.0 / n] * n
    out_deg = [len(a) for a in adj]
    for _ in range(iters):
        new = [(1 - damping) / n] * n
        for i in range(n):
            if out_deg[i]:
                share = damping * pr[i] / out_deg[i]
                for j in adj[i]:
                    new[j] += share
            else:  # dangling: distribute uniformly
                d = damping * pr[i] / n
                for j in range(n):
                    new[j] += d
        pr = new
    return pr


def _gini(values: list[float]) -> float:
    v = sorted(x for x in values if x is not None)
    n = len(v)
    if n < 2:
        return float("nan")
    s = sum(v)
    if s == 0:
        return 0.0
    cum = sum((i + 1) * v[i] for i in range(n))
    return round((2 * cum) / (n * s) - (n + 1) / n, 4)


def compute_ici(text: str) -> float:
    paras = paragraphs(text)
    if len(paras) < 4:
        return float("nan")
    pr = _pagerank(_dep_graph(paras))
    return _gini(pr)


def all_aesthetics(text: str) -> dict:
    return {
        "ce": compute_ce(text),
        "nrr": compute_nrr(text),
        "bsd": compute_bsd(text),
        "ici": compute_ici(text),
    }

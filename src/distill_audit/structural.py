"""Pure-text structural metrics from mindset/Opus (final_decision.md +
operationalization_protocol.md). Zero API, zero GPU. Computes the exact metrics
the LLM-judge rubric only approximates, for convergent-validity cross-checking.

Metrics:
  CCS  Correction Coupling Strength  (final_decision)  -> critique<->context lexical coupling vs random
  PDD  Paragraph Dependency Depth     (final_decision)  -> mean back-reference span
  IS   Inertia Slope                  (final_decision)  -> new-concept-rate slope (J/P/drift)
  EAR  Early Assertion Rate           (operationalization) -> assertions in first 20%
  OED  Overthrow Event Density        (operationalization) -> "established->overturned" events /1k tok
  RT   topology_density, branch_merge_ratio, back_ref_depth (operationalization)
  RAE  Reasoning-Action bigram entropy + routinization     (operationalization)
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter

_PARA = re.compile(r"\n\s*\n")
_WORD = re.compile(r"\b\w+\b")
_TOK = re.compile(r"\S+")

CRITIQUE = re.compile(
    r"(?i)\b(wait|but|however|actually|reconsider|recheck|re-check|check|verify|mistake|wrong|"
    r"recompute|let me|hmm|oops|incorrect|hold on|on second thought)\b|不对|等等|然而|重新|检查|验证|错误|修正",
)
ASSERTION = re.compile(
    r"(?i)\b(the answer is|therefore|thus|clearly|obviously|it follows that|we can conclude|"
    r"this means|the result is|certainly|definitely)\b|(答案[是为]|因此|所以|显然|可以得出|结论[是为])",
)
OVERTHROW = re.compile(
    r"(?i)((earlier|previously|above|before|i (said|stated|concluded|assumed)).{0,90}"
    r"(wrong|incorrect|mistake|flawed|not right|doesn't hold|does not hold))"
    r"|((wait|hold on|actually).{0,50}(this (is|was) wrong|let me reconsider|i need to rethink))"
    r"|((之前|前面|上面|刚才).{0,50}(错了|不对|有误|不成立))",
    re.DOTALL,
)
BACKREF = re.compile(
    r"(?i)(as (mentioned|shown|noted|discussed|established|calculated) (above|earlier|before|previously)|"
    r"from (step|equation|part) (\d+)|recall(ing)? that|going back to|using the (result|value|equation) (from|we))"
    r"|如前[所面]述|根据(前面|上面|之前)|回[到顾]",
)
CAUSAL = re.compile(r"(?i)\b(therefore|hence|thus|consequently|so |as a result|this (means|implies|gives))\b|因此|所以|由此")
BRANCH = re.compile(r"(?i)\b(alternatively|on the other hand|another (approach|way|method)|case [12]|if .{0,30} then)\b|另一[方种]面|或者|如果")
MERGE = re.compile(r"(?i)\b(combining|integrating|in (summary|conclusion)|putting it all together|overall|taking (both|all))\b|综合|综上|总[结的]来")

ACTIONS = {
    "VERIFY": re.compile(r"(?i)(let me (check|verify|confirm|double.?check)|to verify|checking|验证|检查)"),
    "CRITIQUE": re.compile(r"(?i)(wait|hold on|but this|however this|this (is|seems) wrong)|等等|不对|这.{0,5}(有问题|错了)"),
    "CONCLUDE": re.compile(r"(?i)(therefore|thus|in conclusion|the (answer|result|solution) is)|因此|所以|结论|答案"),
    "EXPLORE": re.compile(r"(?i)(alternatively|another (way|approach)|what if|suppose)|另一[种方]|或者|假[设如]"),
    "DECOMPOSE": re.compile(r"(?i)(let me break|step \d|first.{0,10}(we|i) (need|should|will))|分[解步]|第[一二三四\d]步|首先"),
    "CALCULATE": re.compile(r"(?i)(calculating|computing|evaluating|substituting|plugging in)|计算|代入|求解|\$.*[=+\-*/].*\$"),
}


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA.split(text or "") if p.strip()]


def _toks(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def compute_ccs(text: str, rng: random.Random) -> float:
    paras = paragraphs(text)
    if len(paras) < 4:
        return float("nan")
    crit = [i for i, p in enumerate(paras) if CRITIQUE.search(p) and i > 0]
    if not crit:
        return float("nan")
    coupling, rand = [], []
    for ci in crit:
        ct = _toks(paras[ci])
        if not ct:
            continue
        for off in range(1, min(6, ci + 1)):
            coupling.append(_jaccard(ct, _toks(paras[ci - off])))
        for _ in range(5):
            ri = rng.randint(0, len(paras) - 1)
            if ri != ci:
                rand.append(_jaccard(ct, _toks(paras[ri])))
    if not coupling or not rand:
        return float("nan")
    mc, mr = sum(coupling) / len(coupling), sum(rand) / len(rand)
    return mc / max(mr, 0.001)


def compute_pdd(text: str) -> float:
    paras = paragraphs(text)
    depths = []
    for j, para in enumerate(paras):
        if BACKREF.search(para):
            m = re.search(r"step (\d+)", para, re.IGNORECASE)
            if m:
                d = j - int(m.group(1))
                if d > 0:
                    depths.append(d)
            else:
                depths.append(3)  # conservative default (per final_decision)
    return sum(depths) / len(depths) if depths else 0.0


def compute_is(text: str) -> float:
    paras = paragraphs(text)
    n = len(paras)
    if n < 6:
        return float("nan")
    third = n // 3
    sections = [paras[:third], paras[third:2 * third], paras[2 * third:]]
    seen: set[str] = set()
    rates = []
    for sec in sections:
        words = set(re.findall(r"\b\w{4,}\b", " ".join(sec).lower()))
        new = words - seen
        seen |= words
        rates.append(len(new) / max(len(words), 1))
    return rates[2] - rates[0]


def compute_ear(text: str) -> float:
    paras = paragraphs(text)
    n = len(paras)
    if n < 5:
        return float("nan")
    cut = max(1, int(n * 0.2))
    early = paras[:cut]
    return sum(1 for p in early if ASSERTION.search(p)) / len(early)


def compute_oed(text: str) -> float:
    n_tok = len(_TOK.findall(text or ""))
    if n_tok < 100:
        return float("nan")
    return 1000.0 * len(OVERTHROW.findall(text)) / n_tok


def compute_rt(text: str) -> dict:
    paras = paragraphs(text)
    n = len(paras)
    if n < 3:
        return {"topology_density": float("nan"), "branch_merge_ratio": float("nan"), "back_ref_depth": 0.0}
    edges = 0
    branch = merge = 0
    depths = []
    for j, para in enumerate(paras):
        if j > 0 and CAUSAL.search(para):
            edges += 1
        if BACKREF.search(para):
            edges += 1
            depths.append(3)
        if BRANCH.search(para):
            edges += 1
            branch += 1
        if MERGE.search(para):
            edges += 1
            merge += 1
    return {
        "topology_density": round(edges / n, 3),
        "branch_merge_ratio": round(branch / max(merge, 1), 3),
        "back_ref_depth": round(sum(depths) / len(depths), 2) if depths else 0.0,
    }


def _action_of(para: str) -> str:
    for name, rx in ACTIONS.items():
        if rx.search(para):
            return name
    return "OTHER"


def compute_rae(text: str) -> dict:
    paras = paragraphs(text)
    if len(paras) < 5:
        return {"bigram_entropy": float("nan"), "routinization": float("nan")}
    acts = [_action_of(p) for p in paras]
    comp = [acts[0]]
    for a in acts[1:]:
        if a != comp[-1]:
            comp.append(a)
    bg = [(comp[i], comp[i + 1]) for i in range(len(comp) - 1)]
    if len(bg) < 3:
        return {"bigram_entropy": float("nan"), "routinization": float("nan")}
    c = Counter(bg)
    tot = sum(c.values())
    ent = -sum((v / tot) * math.log2(v / tot) for v in c.values())
    max_ent = math.log2(min(tot, len(ACTIONS) ** 2)) or 1.0
    return {"bigram_entropy": round(ent, 3),
            "routinization": round(1 - ent / max_ent, 3),
            "dominant_bigram_ratio": round(sum(v for _, v in c.most_common(3)) / tot, 3)}


_FMT = {
    "math": re.compile(r"\$.+?\$|\\boxed|\\frac|\\sum|\\int|[=≠≤≥]\s*\d|\b\d+\s*[+\-*/]\s*\d+"),
    "code": re.compile(r"```|\bdef \b|\bimport \b|\breturn \b|[{};]\s*$|^\s*(for|while|if)\s*\("),
    "list": re.compile(r"^\s*(\d+[.)]|[-*•]|\bstep \d)", re.IGNORECASE),
    "table": re.compile(r"\|.+\|"),
}


def _fmt_of(para: str) -> str:
    for name, rx in _FMT.items():
        if rx.search(para):
            return name
    return "prose"


def compute_fsd(text: str) -> float:
    """H3 format-switch density: representation-format switches per paragraph."""
    paras = paragraphs(text)
    if len(paras) < 3:
        return float("nan")
    fmts = [_fmt_of(p) for p in paras]
    switches = sum(1 for i in range(1, len(fmts)) if fmts[i] != fmts[i - 1])
    return round(switches / (len(fmts) - 1), 3)


def compute_ci(ear: float | None, oed: float | None, inertia_slope: float | None) -> float | None:
    """Cognitive Inertia Index (operationalization_protocol): EAR/(OED+eps) x slope.
    More positive = higher inertia (early assertion, no overthrow, front-loaded concepts)."""
    if ear is None or oed is None or inertia_slope is None:
        return None
    return round((ear / (oed + 0.01)) * inertia_slope, 3)


def all_metrics(text: str, seed: int = 42) -> dict:
    rng = random.Random(seed)
    rt = compute_rt(text)
    rae = compute_rae(text)
    _nz = lambda x: round(x, 3) if not math.isnan(x) else None  # noqa: E731
    ear = _nz(compute_ear(text))
    oed = _nz(compute_oed(text))
    is_ = _nz(compute_is(text))
    return {
        "ccs": _nz(compute_ccs(text, rng)),
        "pdd": round(compute_pdd(text), 3),
        "inertia_slope": is_,
        "ear": ear,
        "oed": oed,
        "ci_index": compute_ci(ear, oed, is_),
        "fsd": _nz(compute_fsd(text)),
        "rt_topology_density": rt["topology_density"],
        "rt_branch_merge_ratio": rt["branch_merge_ratio"],
        "rt_back_ref_depth": rt["back_ref_depth"],
        "rae_bigram_entropy": rae.get("bigram_entropy"),
        "rae_routinization": rae.get("routinization"),
    }

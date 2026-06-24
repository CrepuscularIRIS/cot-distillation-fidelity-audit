"""Consolidated per-trace audit rubric (Opus-primary core + Gemini garnish).

One judge pass scores every kept dimension at once, anchored to verbatim CoT
quotes. The judge scores the REASONING ONLY and must ignore whether the final
answer is correct. If there is no genuine self-critique, the closure scores are
0 — fabricated critique must not be rewarded.
"""

from __future__ import annotations

from ..schema import Trace

SYSTEM = """You are a rigorous reasoning-trace auditor. You read a model's chain of \
thought (CoT) and score its STRUCTURE, not whether the final answer is correct. \
Score only from the CoT text. If a dimension is not present, score it 0 / "none" — \
never invent critique that is not in the text. Output ONE JSON object and nothing else.

Scoring rubric (the CORE three are most important):

CORE — critique-correction closure:
- critique_present (0/1): is there any genuine self-critique / doubt / error-finding?
- monitoring_control_coupling (0-3): 0 = a critique marker (wait/however/let me check) \
with NO change to the reasoning path; 1 = local tweak that keeps the same conclusion; \
2 = introduces a new method/branch to test the error and changes a conclusion; \
3 = full loop: finds error, attributes cause, fixes it, and re-checks downstream.
- causal_depth_of_critique (0-2): ablation test — if you deleted the critique sentence, \
0 = text still flows (decorative), 1 = a number/term would change, 2 = the whole chain \
would diverge (structural pivot).
- answer_change ("none"|"intermediate"|"final"): did a critique actually change an \
intermediate conclusion or the final answer?
- verification_independence ("none"|"restatement"|"independent"): is any verification a \
mere restatement of the answer, or an independent check (re-derive, substitute back, \
boundary/unit check, cross-method)?
- ccr_closure (0-4): overall closure strength. 0 none; 1 surface marker only; 2 finds a \
problem but only a local patch, answer unchanged; 3 locate+cause+fix but weak verify; \
4 full locate -> cause -> fix -> answer-change/independent-verify loop.
- failure_type: one of [surface_marker_only, redundant_check, local_patch_only, \
no_answer_change, false_correction, overlong_but_shallow, format_only] or null if closure is strong.

TOPOLOGY:
- reasoning_topology: one of [chain, tree, graph, loop, drift].
- uncertainty_stage: subset list of [early, mid, late] (where doubt is expressed); [] if none.
- early_assertion (0/1): is a firm conclusion asserted within roughly the first fifth?
- overthrow_present (0/1): is a previously established intermediate conclusion overturned?

GARNISH — executive function:
- planning_depth (0-3): 0 none, 1 trivial 'first/second', 2 modular phases, 3 phases with exit conditions.
- plan_execution_consistency (0-3): 0 abandons plan, 1 drops/scrambles steps, 2 follows plan, 3 follows + status-checks.

GARNISH — open reasoning:
- divergent_score (0-3): genuine alternative-path generation (not templated headers).
- creative_destruction (0/1): abandons a working-but-inelegant approach for a better paradigm.
- templated_divergence (0/1): divergence is a fixed template/ritual rather than problem-driven.

META (required):
- evidence: 1-3 SHORT verbatim quotes from the CoT supporting the closure scores.
- rationale: <= 2 sentences justifying ccr_closure.
- judge_confidence ("low"|"medium"|"high").
"""

EXPECTED_FIELDS = {
    "critique_present", "monitoring_control_coupling", "causal_depth_of_critique",
    "answer_change", "verification_independence", "ccr_closure", "failure_type",
    "reasoning_topology", "uncertainty_stage", "early_assertion", "overthrow_present",
    "planning_depth", "plan_execution_consistency", "divergent_score",
    "creative_destruction", "templated_divergence", "evidence", "rationale",
    "judge_confidence",
}

# Pilot truncation budget; Phase 2 replaces this with map-reduce for ultra-long traces.
MAX_REASONING_CHARS = 24000


def build_user_prompt(trace: Trace, max_reasoning_chars: int = MAX_REASONING_CHARS) -> str:
    reasoning = trace.reasoning
    note = ""
    if len(reasoning) > max_reasoning_chars:
        head = reasoning[: max_reasoning_chars // 2]
        tail = reasoning[-max_reasoning_chars // 2:]
        reasoning = f"{head}\n\n[... {len(trace.reasoning) - max_reasoning_chars} chars elided ...]\n\n{tail}"
        note = ("\n[NOTE: the CoT was truncated head+tail to fit; score closure on the "
                "visible structure and lower judge_confidence accordingly.]")
    answer = trace.answer[:2000]
    return (
        f"PROBLEM:\n{trace.problem[:2000]}\n\n"
        f"MODEL CoT (reasoning to be scored):\n{reasoning}{note}\n\n"
        f"FINAL ANSWER (context only — do NOT score its correctness):\n{answer}\n\n"
        "Return the JSON object now."
    )


def validate(d: dict) -> list[str]:
    """Return a list of QC problems (empty = clean)."""
    problems = [f"missing:{f}" for f in EXPECTED_FIELDS if f not in d]
    ccr = d.get("ccr_closure")
    if not isinstance(ccr, int) or not (0 <= ccr <= 4):
        problems.append(f"bad ccr_closure:{ccr!r}")
    if d.get("reasoning_topology") not in {"chain", "tree", "graph", "loop", "drift", None}:
        problems.append(f"bad topology:{d.get('reasoning_topology')!r}")
    return problems

# Plan — Distillation Reasoning-Fidelity Audit (Track A)

> Status: DRAFT v1 · Date: 2026-06-24 · Owner: Claude (lead analyst) + Codex (hook-assist) + Kimi-for-coding (execution judge)
> Source thinking: `mindset/` (Opus = primary, Gemini = garnish). Survey paper = Track B (later, cites this).

## 0. One-line thesis (descriptive / correlational — no causal claim)

Public reasoning-distillation datasets inherit the **vocabulary** of self-critique while losing its
**causal topology**: critique markers proliferate, but their coupling to error-localization →
correction → answer-change is weak and teacher-dependent. We build a low-cost, reproducible
**LLM-as-judge audit** that separates genuine critique-correction closure from ritual critique,
validated on a natural experiment (same question, different teacher).

## 1. Grounded reality (verified 2026-06-24)

**Datasets cached at `/data/huggingface/hub/` (complete unless noted):**
- Opus/Claude: angrygiraffe-8.7k, Jackrong TraceInversion 4.6 / 4.7, nohurry Opus-4.6-filtered, Roman opus-4.6-10000x
- Gemini: Roman gemini-3.1-pro-hard-high-reasoning (schema: `original_input/model_thoughts/model_response`)
- GLM-5.1-Reasoning-1M-Cleaned (30G), DeepSeek-V4-Distill-8000x (136M) — **the natural experiment**
- Kimi-K2.5 (downloading), OpenThoughts-114k (downloading), Qwen reasoning (pending)
- Open baselines present: NuminaMath-CoT / 1.5
- **Finetuned model cached: `Jackrong/Qwopus3.6-27B-v1-preview`** → enables optional model-behavior validation (Phase 4)

**APIs (`.env`, secrets by name only):**
- Kimi-for-coding: OpenAI-compatible via `KIMI_BASE_URL=http://localhost:4242` (proxy `~/claw/codex-switch/kimi_proxy.py`), model `kimi-for-coding`, keys `KIMI_API_KEY1/2`. **Proxy must be started (OAuth) before use.**
- Codex/GPT-5.x: `OPENAI_BASE_URL=http://localhost:4141`; also assists via Claude Code hooks.
- `Huggingface_Token` for downloading datasets + finetuned weights.

**Operating model (Approach C, lead = Claude):**
- **Claude (me)** = lead analyst: writes/monitors/reviews code, designs rubric, acts as calibration judge, drives reconvergence. Replaces the human-annotator role.
- **Codex** = assists via hooks (review / second opinion).
- **Kimi-for-coding** = pinned execution judge, called from a Python file, scoring every entry uniformly. Python only plumbs (load/sample/aggregate/plot); all judgments are LLM-emitted.

## 2. Research questions

- **RQ1 (paradox):** does critique-marker *density* dissociate from critique-correction *closure*?
- **RQ2 (teacher fingerprint):** holding the question fixed (GLM↔DeepSeek pairs), do teachers differ in closure structure?
- **RQ3 (method/construct validity):** does the cheap LLM rubric agree with a high-reasoning calibration judge (Claude/Codex)? precision/recall on a held set.
- **RQ4 (curation contrast, secondary):** auto-distilled (GLM/DeepSeek) vs curated (nohurry-filtered, Claude-TraceInversion) vs half-explicit (Gemini) — incl. Gemini "templated divergence".

## 3. Executable hypothesis triage (Opus primary, Gemini garnish)

| Idea | Source | Role | Rubric fields |
|---|---|---|---|
| Critique→correction closure, causal depth (ablation), uncertainty topology | Opus H1/H2 + Gemini MC | **CORE** | monitoring_control_coupling 0-3, causal_depth 0-2, CCR 0-4, answer_change, verification_independence, failure_type |
| Reasoning topology + correction-loop (chain/tree/graph/loop/drift) | Opus H1 | **CORE** | reasoning_topology, uncertainty_stage |
| Cognitive inertia (early assertion, overthrow, where uncertainty sits) | Opus H2 | CORE (merged) | early_assertion, overthrow_density |
| Executive function (planning, plan-exec consistency) | Gemini EF | garnish | planning_depth 0-3, plan_execution_consistency 0-3 |
| Open reasoning (divergent/convergent, creative destruction, templated divergence) | Gemini Open | garnish | divergent 0-3, creative_destruction 0/1, templated_divergence 0/1 |
| Routinization entropy (H5), format friction (H3), ToM, SC | Opus/Gemini | optional/parked | — |
| Incompressible kernel (H4, cross-scale) | Opus H4 | **parked → Phase 4 / Track B** | needs model pairs (Qwopus3.6 partially) |
| Value anchoring (VAS) | — | **dropped** (data cleaned of refusals) | — |

## 4. Unified schema + adapters (registry pattern)

`Trace{ dataset, teacher, problem, reasoning, answer, domain, difficulty, meta, uid }`
- Jackrong-shared (GLM/DeepSeek/Claude-TraceInversion): `problem=input`, `reasoning=<thinking>`, `answer=text after </thinking>`, `teacher=meta.teacher_model`.
- Gemini: `problem=original_input`, `reasoning=model_thoughts`, `answer=model_response`.
- nohurry: `problem/thinking/solution` fields.

## 5. Measurement protocol

- **One consolidated rubric**, scored in a single judge pass per trace, every score backed by a verbatim CoT quote. Hard rule: score from CoT only; ignore final-answer correctness; no critique node → scores 0 (no hallucinated critique).
- **Pinned judge** (Kimi-for-coding, fixed version) scores *both halves* of every matched pair → no judge-confound.
  - VERIFIED constraints: model forces `temperature=1` (cannot pin 0); it is a reasoning model (answer in `content`, hidden CoT in `reasoning_content`), so judge calls need a large `max_tokens` budget or `content` returns empty. Reproducibility comes from judge self-consistency + calibration, not temp-0.
- **Calibration leg (RQ3):** Claude + Codex re-score a ~150-300 held subset; report Kimi-vs-strong agreement (κ / correlation). Claude (me) is the lead calibration judge.
- **Ultra-long traces** (Qwen/GLM up to ~88k chars): map-reduce — chunk → per-chunk action+critique+answer-state timeline → second pass scores closure/EF/topology over the timeline. Threshold documented; capped-window fallback if cost balloons.
- **Cheap structural baseline** (per-1k-token critique-marker density) computed in Python ONLY to demonstrate the paradox (density vs LLM closure). Keywords route/sample; never score.

## 6. Phased execution

- **Phase 0 — de-risk (no API):** unify schema; **measure GLM↔DeepSeek matched-pair yield** (the spine precondition); report distribution by domain/length. Fallback if yield is low: stratified domain/difficulty matching instead of exact-question pairing.
- **Phase 1 — pilot (needs Kimi proxy up):** consolidated rubric on ~100-200 traces incl. matched pairs; Claude/Codex calibrate; iterate rubric until κ acceptable.
- **Phase 2 — scale:** Kimi judges the full analysis sample (matched pairs + stratified draws across all datasets); aggregate stats (Wilcoxon paired, Cliff's δ, bootstrap CI, McNemar on topology); headline density-vs-closure scatter.
- **Phase 3 — reconverge:** Claude reads aggregated results, marks which Opus/Gemini hypotheses survived, redesigns the failed ones, re-runs. Loop until stable. Produce the "new convergence".
- **Phase 4 — optional model-behavior validation:** probe `Qwopus3.6-27B` — do data-fingerprints predict the finetuned model's behavior? (GPU-gated; stretch.)

## 7. Decision rule (what makes the thesis stand)

Thesis stands if (1) density⊥closure dissociation is significant with non-trivial effect size, (2) the
Kimi judge is calibration-validated, (3) teacher differences in closure survive length/domain controls.
Any failing → honest reporting; the validated-protocol contribution (RQ3) survives regardless.

## 8. Top risks

1. ~~Low matched-pair yield~~ **RESOLVED 2026-06-24**: Phase 0 measured **100% yield — 7708/7708 DeepSeek questions match GLM (8873 pair rows, all in GLM `main`)**. Spine confirmed; manifest at `distill_audit/outputs/matched_pairs.jsonl`.
2. **Judge-confound** → pinned single judge per comparison.
3. **"No errors"** = pipeline correctness (extraction/pairing/no silent truncation) + measured judge noise (self-consistency + calibration), not an infallible judge.
4. **Cleaning bias** (refusals removed) → all "fidelity" numbers reported as conditional on cleaning.
5. **Cost / ultra-long traces** → map-reduce + sample-size caps, logged.

## 9. Project layout (`distill_audit/`)

`src/distill_audit/`: `schema.py`, `extract.py`, `adapters.py` (registry), `pairing.py`,
`judge/{client,rubric,schema}.py`, `metrics/`. `scripts/`: `check_pairing.py`, `run_pilot.py`.
`tests/` (pytest). `outputs/` (results, conditional-on-cleaning notes). Python via `uv`; logging not print.

# Roadmap — next steps (deferred)

The dataset audit (Track A) is complete: 440 traces across 10 datasets, all judged, with
the natural experiment as the robust core (see [`README`](README.md) and
[`outputs/`](outputs/)). The two phases below validate the audit's central claim **on the
models themselves** rather than on their training text. Both are planned future work — the
methodology and target models are fixed; only the GPU runs remain.

> Central claim under test: **a model's self-correction closure is inherited from the
> closure of its distillation training data.** If true, this audit becomes a *predictive*
> training-data filter, not just a descriptive one.

---

## Round 6 — LLM behavioral-fidelity validation

Do models *trained on* a dataset reproduce that dataset's measured closure when they
reason? We generate fresh `<think>` traces from trained models on the **same matched
problems** used in the natural experiment, judge them with the **same CCR rubric**, and
compare each model's behavioral closure to its training data's closure.

### A. Distillation-inheritance (a *controlled* lift of the natural experiment)

Jackrong publishes the distillation students with a shared base, which gives a clean control:

| Model | Base | Trained on | Training-set closure (this audit) |
|---|---|---|---|
| `Jackrong/Qwen3.5-9B-GLM5.1-Distill-v1` | Qwen3.5-9B | GLM-5.1 distillation | CCR **1.67** (high) |
| `Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash` | Qwen3.5-9B | DeepSeek-V4 distillation | CCR **0.60** (low) |

**Same base, same scale — only the teacher's data differs.** Prediction: the GLM-student
reasons with measurably higher closure than the DeepSeek-student, reproducing the data-level
gap at the behavioral level. A null result (equal closure) would show closure is *not*
inherited from training text — equally informative.

### B. Cross-scale (the H4 "incompressible kernel")

| Model | Scale | Family |
|---|---|---|
| `Jackrong/Qwopus3.5-9B-v3` | 9B | Qwopus |
| `Jackrong/Qwopus3.6-27B-v1-preview` | 27B | Qwopus |

Does closure rise, fall, or stay flat with model size on a fixed training recipe? `Qwopus3.6`
was SFT'd primarily on a low-closure Claude-distillation mix (8B-style-filtered), so the
prediction is **low-moderate closure that does not jump to the high-closure end**, regardless
of scale.

### Method (tooling already in this repo)
1. `scripts/gen_model_traces.py --model-dir <m> --tag <t> --out outputs/h4_<t>_traces.jsonl`
   — generate `<think>` traces on the matched problems (transformers, `qwen3_5` arch, no remote code).
2. `scripts/h4_prep_and_struct.py --traces … --tag <t>` — zero-API structural read + judge-batch prep.
3. Judge the batches with the same Opus CCR rubric (`src/distill_audit/judge/rubric.py`).
4. Compare: GLM-student vs DeepSeek-student (inheritance), Qwopus 9B vs 27B (scale), all vs
   the dataset baselines in `outputs/report_cross_dataset.json`.

All target models are public/ungated `Qwen3_5ForConditionalGeneration` checkpoints.

---

## Round 7 — benchmark scoring of the models (跑分)

After the closure-behavior check, score the same models on standard reasoning benchmarks
(e.g. GSM8K / MATH / a reasoning suite) **and** on our CCR closure metric, then test whether
**measured closure correlates with benchmark accuracy.** This asks the practical question:
does training on higher-closure data buy better reasoning, or only better-looking reasoning?

Deferred — to be run later.

---

## Already parked (with justification)
- **LSI** (logical symmetry) — needs reverse-problem generation.
- **SC / ToM** cognitive lenses — the corpus is solitary problem-solving; no social/nested-belief content.
- **Value-anchoring (VAS)** — refusal/safety content was cleaned from these datasets upstream.

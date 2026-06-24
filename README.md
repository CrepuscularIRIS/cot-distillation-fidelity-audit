# CoT Distillation Fidelity Audit

**Does reasoning distillation transmit the *act* of self-correction, or only its *vocabulary*?**

An LLM-as-judge audit of **critique–correction closure** in 10 public chain-of-thought
distillation datasets (440 traces). The finding, in one line:

> **Distillation copies the *sound* of a mind correcting itself far more easily than the
> *act*. The correction loop has to be captured live, not reconstructed — and you can
> audit each trace for it, cheaply, before you train on it.**

---

## The headline result

A **natural experiment**: `DeepSeek-V4-Distill` and `GLM-5.1` answer the *same 60
questions*. Same task, length-matched, both emit `<think>`. We score each trace's
**Critique-Correction Closure (CCR, 0–4)** with an LLM judge.

| | GLM-5.1 | DeepSeek-V4 |
|---|---:|---:|
| **CCR (closure)** | **1.67** | 0.60 |
| critique-word density | 7.34 | **8.66** |
| self-overthrow rate | **31.7%** | 11.7% |

GLM shows more self-revision (**38 of 60 pairs vs 4**, Wilcoxon z = −4.98, p ≈ 0, Cliff's δ =
0.51 *large*) **while using fewer critique words.** The vocabulary of self-correction points
the *wrong way*; the measured act co-varies with teacher identity. (This is observational — it
fixes the question, not the teacher; training pipeline and per-trace length aren't separated.)
It holds for **every judge that scored the pairs — 4 judges, including DeepSeek's own teacher**
(DeepSeek-Flash agreed on all 5 it scored).

---

## What's in this repo

```
mindset/     The research design — the hypotheses being tested (Opus + Gemini docs)
src/         The audit toolkit (LLM judge clients, rubric, structural & aesthetic metrics)
scripts/     The pipeline (pairing, judging, aggregation, statistics)
outputs/     All results: reports (.md) + machine-readable scores (.json/.jsonl)
plan/        The execution plan
```

### Read the results in this order
1. **[`outputs/SYNTHESIS.md`](outputs/SYNTHESIS.md)** — the complete story, every design doc threaded in.
2. **[`outputs/HYPOTHESES_VS_RESULTS.md`](outputs/HYPOTHESES_VS_RESULTS.md)** — each hypothesis, one-by-one, with exact numbers and a verdict.
3. **[`outputs/REPORT_distill_fidelity.md`](outputs/REPORT_distill_fidelity.md)** — the full 9-section report.
4. **[`outputs/CONCLUSIONS.md`](outputs/CONCLUSIONS.md)** — the short version.
5. **[`outputs/COMPLETENESS.md`](outputs/COMPLETENESS.md)** — every metric mapped to its implementation.

---

## The corpus (10 datasets · 440 traces)

| Dataset | Teacher | Distillation method | CCR |
|---|---|---|---:|
| qwen | Qwen3.5-27B | native-capture | **2.08** |
| glm | GLM-5.1 | native-capture | **1.67** |
| kimi | KIMI-K2.5 | native-capture | **1.30** |
| claude47_ti | claude-opus-4.7 | trace-inversion (reconstruction) | 0.68 |
| deepseek | DeepSeek-V4-Distill | native-capture (short) | 0.60 |
| gemini | gemini-3.1-pro | native-capture | 0.60 |
| claude46_ti | claude-opus-4.6 | trace-inversion (reconstruction) | 0.43 |
| nohurry_opus | claude-opus-4.6 | human-filtered "high quality" | 0.18 |
| angrygiraffe | claude-opus-4.6 | synthetic / roleplay | 0.05 |
| roman_claude | claude-opus-4.6 | filtered / direct (no `<think>`) | 0.03 |

Grouped by source family: **Jackrong** (glm, deepseek, kimi, qwen, gemini, claude46_ti,
claude47_ti, angrygiraffe, nohurry_opus), **Roman** (roman_claude), **Opus** distillation
(nohurry_opus). By method, native-capture ranks above reconstruction/synthetic above
filtered/direct — **but ~⅔ of that gap is trace length** (native ~11k vs reconstruction ~1.8k
chars), and the one *short* native set (DeepSeek) ranks among the reconstructions, so method and
length are not separable at the dataset level.

---

## Methodology — "LLM judges, Python plumbs"

Every score is an LLM reading one trace against a fixed rubric. Python only loads, samples,
calls the API, and aggregates — **no regex ever assigns a judgment.**

- **Judge fleet:** Opus 4.8 (primary, all 440) validated by a 5-model panel — Kimi,
  DeepSeek-V4-Flash, MiniMax-M3, Codex/GPT-5.4. Self-bias flagged where a teacher judges its
  own student.
- **Reliability:** Opus self-consistency 95.6% within-1; pairwise QWK 0.36–0.86 (magnitudes
  differ). Every judge ranks GLM > DeepSeek, but that is the only pair all judges scored — so the
  panel ρ=1.0 is a 2-point agreement; the full 10-dataset ranking is Opus-only.
- **Free cross-check:** zero-API structural metrics from `mindset/Opus` (CCS, PDD, IS, RT, RAE)
  and aesthetics from `mindset/Gemini` (CE, NRR, BSD, ICI) + an information-theoretic
  MI-collapse. RAE bigram-entropy reproduces the dataset ranking at Spearman **0.927**.

### Honest caveat (the length confound)
~⅔ of the *cross-dataset* gap is a trace-length / observability effect (Pearson 0.65;
native 11k vs reconstruction 1.8k chars). Reconstruction yields short, polished text in which
self-correction is **unobservable**. We *hypothesize* that what the text doesn't show, a
text-only student can't learn (unobservable ⇒ unlearnable) — **untested here, and exactly what
Round 6 (H4) probes.** The length-matched natural experiment is the robust core.

---

## Reproduce

Reading the results needs nothing — they're committed under `outputs/`. To re-run the
pipeline you need the source datasets (see [`DATASETS.md`](DATASETS.md)) and API access for
the judges:

```bash
pip install -e .
cp .env.example .env      # fill in judge API keys
python scripts/run_structural.py     # zero-API: structural metrics
python scripts/run_aesthetics.py     # zero-API: aesthetic + MI metrics
python scripts/run_pilot.py          # LLM judge on the GLM↔DeepSeek pairs
```

The audit toolkit (`src/distill_audit/`) is API-agnostic; point `DISTILL_AUDIT_ENV` at your
`.env` or drop one in the repo root.

---

## Status

- ✅ **Track A (this release): complete.** Jackrong, Roman, and Opus dataset families fully
  analyzed — 440 traces, all judged, reports + machine-readable scores in `outputs/`.
- 🗺️ **Next (deferred): model-behavioral validation** — see **[`ROADMAP.md`](ROADMAP.md)**.
  - *Round 6:* generate `<think>` from the trained **distillation students** and test whether a
    model's closure is inherited from its training data — a *controlled* lift of the natural
    experiment (`Qwen3.5-9B-GLM5.1-Distill` vs `Qwen3.5-9B-DeepSeek-V4-Flash`, same base/scale)
    plus a cross-scale leg (Qwopus 9B vs 27B). Tooling is in `scripts/gen_model_traces.py` +
    `scripts/h4_prep_and_struct.py`.
  - *Round 7:* benchmark-score those models and test whether measured closure correlates with
    reasoning accuracy.
- ⏸ Parked with justification: LSI (needs reverse-problem generation), SC/ToM lenses
  (data is solitary problem-solving), value-anchoring (refusals cleaned from data).

## License & citation

Code and reports: **MIT** (see [`LICENSE`](LICENSE)). Source datasets retain their original
licenses ([`DATASETS.md`](DATASETS.md)). If you use this, please cite the repo.

```bibtex
@misc{cot_distill_fidelity_audit,
  title  = {CoT Distillation Fidelity Audit: Critique-Correction Closure in Reasoning Distillation},
  year   = {2026},
  note   = {LLM-as-judge audit of 10 distillation datasets},
  howpublished = {\url{https://github.com/CrepuscularIRIS/cot-distillation-fidelity-audit}}
}
```

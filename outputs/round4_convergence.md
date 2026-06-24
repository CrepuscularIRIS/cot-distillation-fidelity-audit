> ⚠️ **SUPERSEDED IN PART by `REPORT_distill_fidelity.md` (v2)** after a Codex adversarial review. See the "Response to adversarial review" section at the bottom — several claims here were corrected (notably: the cross-dataset gap is ~⅔ a trace-length/observability effect; "method determines topology" was overclaimed; total n=440 not 435). The natural-experiment result stands.

# Round 4 — Convergence Analysis

> Opus 4.8 judge over 440 traces across 10 distillation datasets (Jackrong, Roman, Opus families).
> Judge self-consistency: 94.3% within-1 CCR, mean abs diff 0.37. Opus-vs-Kimi calibration: Opus is the stricter judge (1.2 vs 1.8, 77% within-1).

## 1. The data overturned the prior — and gave a sharper thesis

**Original prior (from `mindset`):** Claude/Opus = the closure "gold standard"; open models = "ritual critique."

**What the data shows (CCR closure, 0–4):**

| dataset | n | ccr | coupling | depth | density/1k | loop% | overthrow% | distill method |
|---|---|---|---|---|---|---|---|---|
| qwen (Qwen3.5) | 40 | **2.08** | 1.15 | 0.95 | 9.2 | 17.5 | 25 | native-capture |
| glm (GLM-5.1) | 60 | **1.67** | 0.95 | 0.77 | 7.3 | 5.0 | 32 | native-capture |
| kimi (K2.5) | 40 | **1.30** | 0.75 | 0.62 | 11.7 | 12.5 | 15 | native-capture |
| claude47_ti | 40 | 0.68 | 0.23 | 0.20 | 3.9 | 0 | 5 | reconstruction |
| deepseek (V4) | 60 | 0.60 | 0.35 | 0.32 | 8.7 | 0 | 12 | native-capture |
| gemini (3.1) | 40 | 0.60 | 0.33 | 0.33 | 3.5 | 2.5 | 12 | native-capture |
| claude46_ti | 40 | 0.42 | 0.05 | 0.03 | 4.3 | 0 | 0 | reconstruction |
| nohurry_opus | 40 | **0.17** | 0.00 | 0.00 | 7.9 | 0 | 0 | filtered |
| angrygiraffe | 40 | **0.05** | 0.03 | 0.03 | 5.5 | 0 | 0 | synthetic |
| roman_claude | 40 | **0.03** | 0.00 | 0.00 | 0.5 | 0 | 0 | direct-answer |

The Claude/Opus-*derived* sets occupy the **bottom**, not the top. But this is not "Claude reasons worse" — none of these are native Claude thinking. They are **reconstructions** (TraceInversion: a 4B model regenerates a plausible CoT), **synthetic** CoT (angrygiraffe), **surface-filtered** (nohurry), or **direct answers** (roman). The reframed thesis is stronger and more original than the prior:

> **Critique-correction closure is a property of the distillation *method*, not the teacher's brand. Native-capture distillation (recording the teacher's actual `<think>` stream) preserves closure topology. Reconstruction / synthesis / surface-filtering keep the *vocabulary* of critique while destroying its *topology*.**

By method group (unweighted dataset means): **native-capture 1.25 vs reconstruction/synthetic 0.38 vs filtered/direct 0.10** — a ~3.3× / ~13× gap. (Method and trace length are entangled — see §2 — so read this as descriptive, not a clean method effect.)

## 2. The natural experiment (the clean, confound-free core)

GLM and DeepSeek-V4 answer the **same 60 questions** (DeepSeek-V4-Distill prompts come from GLM-5.1), both via native-capture, both judged by the same Opus instance:

- **GLM CCR 1.67 vs DeepSeek 0.60** — Cliff's δ = **0.513 (large)**, Wilcoxon z = −4.98, **p ≈ 0**.
- DeepSeek has *higher* critique density (8.66 vs 7.34) yet lower closure, and **0% loop topology vs GLM 5%**.
- DeepSeek's dominant failure mode: `surface_marker_only` (34/60) — ritual "wait/however" with no coupled correction.

This holds the question and method fixed — isolating those two confounds, though not the teacher's full training pipeline or trace length: on identical questions, GLM traces show genuinely coupled self-correction far more often than DeepSeek's.

## 3. Hypothesis survival (Opus H1–H5, Gemini lenses)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H2 — cognitive inertia** (Opus) | ✅ **STRONGLY CONFIRMED** | `overthrow%` = 0 for *every* reconstruction/synthetic/direct set vs 12–32% for native-capture. Reconstruction = total inertia (never overturns a conclusion). Cleanest result in the study. |
| **H1 — reasoning topology** (Opus) | ✅ Confirmed | loop% spans 0→17.5%; native-capture preserves loop/graph topology, reconstruction collapses to pure chains. |
| **Gemini MC — monitoring↔control coupling** | ✅ Validated as the core instrument | `coupling` tracks ccr (r≈1); uncertainty timing splits native (late/mid-reasoning doubt, 48–70% late) from reconstruction (≈0% any uncertainty). |
| **Gemini Open — exploration** | ◑ Nuanced | qwen has highest closure *and* 50% templated_divergence — its exploration is partly ritual but still resolves. angrygiraffe has high divergent_score (1.18) but 0% creative_destruction/overthrow — synthetic *breadth without correction*. |
| **H5 — routinization** (Opus) | ◑ Partial | templated_divergence detectable (qwen 50%, glm 40%) but does not cleanly separate good/bad. |
| **Gemini EF — executive function** | ◑ Supported, weakly discriminating | planning_depth is similar across sets; plan-execution gaps appear but are not the dominant signal here. |
| **H3 — format friction** (Opus) | ⏸ Not testable | data is overwhelmingly NL+math; insufficient format switching. |
| **H4 — incompressible kernel** (Opus) | ⏸ Not testable on data | needs same-data/different-scale model checkpoints. → the `Qwopus3.6-27B` model-behavior probe (future). |
| Value anchoring / SC / ToM | ❌ Dropped | data lacks refusals / social / multi-agent content (confirmed). |

## 4. The "paradox," stated honestly (self-attack survived)

Self-attack: *"Is critique density really decoupled from closure?"* Precisely measured:
- Cross-dataset Spearman(density, ccr) = **+0.52**; pooled trace-level Pearson = **+0.31**.
- So density is **weakly positively** correlated with closure — NOT anti-correlated, NOT independent.

The honest claim is therefore **not** "density is unrelated to closure" but: **density is a weak (~10% of variance), unreliable proxy that fails worst exactly where it matters — on curated/reconstructed data.** The poster child is `nohurry_opus`, a *human-filtered "high-quality"* Opus set: density 7.9 (high) but ccr 0.17 (near-zero). The curation selected for surface critique vocabulary, not coupled correction. The natural experiment is where the dissociation is cleanest (same question, DeepSeek higher density + lower closure).

## 5. Threats to validity (what could still be wrong)

1. **Method/teacher confound.** All Claude sets are reconstructions; no native Claude `<think>` capture exists in this corpus. So "native-capture preserves closure" cannot be fully separated from "Qwen/GLM/Kimi are strong reasoners." The natural experiment (GLM vs DeepSeek, both native) shows teacher matters *too*. Honest framing: in this corpus, method dominates, teacher modulates.
2. **Judge strictness, not ground truth.** Opus is the instrument; it is internally consistent (94% within-1) but stricter than Kimi. Absolute CCR values are judge-relative; the *rankings and gaps* are the robust finding.
3. **Sampling.** n=40/dataset (60 for paired). Adequate for medians and large effects; not for fine distinctions among the mid-pack (deepseek≈gemini).
4. **Cleaning bias.** GLM/Kimi pipelines removed refusals/unparseable outputs; closure numbers are conditional on that cleaning.
5. **Truncation.** Ultra-long traces (qwen) were head+tail truncated to ~9k chars for judging; mid-trace corrections could be undercounted (would *understate* qwen, which still leads).

## 6. Implications for training (the practical payoff)

- **Prefer native-capture distillation** (record the teacher's real reasoning stream) over CoT reconstruction/synthesis if you want self-correction to transfer.
- **Do not use critique-word density as a data-quality filter** — `nohurry_opus` shows this selects the wrong traces. Filter on coupled closure (CCR ≥ 2) instead.
- **Teacher choice matters within native-capture** — GLM/Qwen-style teachers transmit more coupled correction than DeepSeek-style.
- **Next step (H4):** validate against the cached `Jackrong/Qwopus3.6-27B-v1-preview` — does a model finetuned on low-closure data itself produce low-closure reasoning?

---

## Response to adversarial review (Codex, high-effort)

The Round-4 review gate ran an independent Codex statistical/methodology review. Findings and resolutions:

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | CRIT | Sample-size inconsistency (435 vs JSON 440; roman 35 vs 40) | **Fixed.** Final n=440 (roman 40); aggregator re-run; report v2 uses 440. |
| 2 | CRIT | `overthrow%` cited but absent from cited JSON | **Fixed.** `overthrow_pct` now emitted in `report_cross_dataset.json` and `report_paired.json` (GLM 31.7% vs DS 11.7%). |
| 3 | HIGH | Wilcoxon: n=60 vs n_nonzero=42; "p≈0" sloppy | **Fixed.** Reported as 42 non-zero pairs, p < 1e-6 (bound). |
| 4 | HIGH | Cliff's δ ignores pairing | **Fixed.** Added matched-pairs rank-biserial (0.567) + win/loss/tie (38/4/18) + median paired diff as the primary paired effect; δ kept as unpaired secondary. |
| 5 | HIGH | "confound-free natural experiment" overstated | **Fixed.** Reworded: controls question+method+length; teacher and pipeline still vary. |
| 6 | HIGH | "METHOD determines topology" too strong (DeepSeek/Gemini native but low) | **Accepted & reframed.** Native-capture is necessary-not-sufficient; teacher modulates strongly; the natural experiment is the load-bearing claim. |
| 7 | HIGH | Spearman 0.52 (n=10) weak / not significant | **Accepted.** Now framed as directional only, not significant. |
| 8 | MED | coupling≈CCR is tautological (rubric component) | **Accepted.** Demoted from "validation" to descriptive. |
| 9 | HIGH | Single-judge validity not established | **Accepted.** Report v2 §4 states construct validity NOT established; added QWK=0.35 (only "fair" cross-judge); human-κ leg was not run. |
| 10 | MED | Opus-vs-Kimi needs weighted kappa | **Fixed.** Quadratic-weighted κ added (0.35). |
| 11 | HIGH | **Strongest missed alternative: length/observability** | **Accepted — major revision.** Pearson(log len, CCR)=0.654; native 11k vs recon 1.8k chars; length-matched gap shrinks 3.2×→2.3×. ~⅔ of the cross-dataset gap is observability. Report v2 §2 rewritten around this. |
| 12 | MED | density-fails claim = selected counterexamples | **Accepted.** Framed as counterexamples (nohurry, paired) + modest correlations, not a general law. |

**Net effect of the review:** the broad "native-capture preserves closure topology" thesis was pulled back to "most of the cross-dataset gap is trace-length/observability; a ~2.3× residual method effect survives length-matching." The **natural-experiment result (GLM vs DeepSeek, same question/method/length) was untouched and is now the primary finding.** This is the convergence working as intended — the prior was falsified, the first replacement was over-strong, and the adversarial pass produced the defensible version.

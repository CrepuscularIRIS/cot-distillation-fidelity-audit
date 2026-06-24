# The Vocabulary and the Topology of Self-Correction
### A structural-fidelity audit of public reasoning-distillation datasets
*(v2 — revised after adversarial statistical review)*

> 440 reasoning traces · 10 datasets (Jackrong, Roman, Opus families) · judged by Claude Opus 4.8.
> Judge self-consistency 95.6% within-1 (mean abs diff 0.29); Opus-vs-Kimi cross-agreement only fair (QWK 0.35) — see §4.
> Code & data: `src/distill_audit/` · `outputs/report_*.json`, `outputs/opus_all.jsonl` · review trail: `outputs/round4_convergence.md`

---

## 0. 大道至简 — the one sentence (revised)

**On identical questions, a teacher that genuinely revises itself transmits that habit through distillation even though it uses *fewer* critique words — but across datasets, most of the apparent "closure gap" is really a trace-length/observability gap: short, polished, reconstructed CoT simply doesn't show the correction.**

Two claims, very different strengths. The first is clean and survives every control we could apply. The second is real but mostly a length artifact. The honest report keeps them apart.

---

## 1. The robust core: the natural experiment

DeepSeek-V4-Distill answers the **same 60 questions** as GLM-5.1 (its prompts are drawn from GLM), both by **native-capture** distillation (the teacher's real `<think>` stream), both judged by the same Opus instance, at comparable trace lengths. This controls question, method, and length together — the cleanest contrast in the corpus.

| paired, n=60 | DeepSeek-V4 | GLM-5.1 |
|---|---|---|
| CCR closure (mean) | 0.60 | **1.67** |
| critique density /1k | **8.66** | 7.34 |
| overthrow a conclusion | 11.7% | **31.7%** |
| loop topology | 0% | 5% |

- GLM higher on **38** pairs, DeepSeek higher on **4**, tied on 18.
- Matched-pairs **rank-biserial = 0.567**; Wilcoxon signed-rank z = −4.98 over 42 non-zero pairs, **p < 1e-6**.
- DeepSeek's dominant failure mode: `surface_marker_only` (34/60) — ritual "wait/however" with no coupled correction.

**This is the finding that survives review.** Holding question difficulty, distillation method, and length fixed, GLM's teacher revises itself far more often and more deeply — *while using fewer critique words*. The vocabulary points the wrong way. Teacher cognitive style transmits through native-capture distillation.

---

## 2. The cross-dataset picture — and its big confound

| dataset | n | CCR | density | mean chars | loop% | overthrow% | method |
|---|---|---|---|---|---|---|---|
| Qwen3.5 | 40 | 2.08 | 9.2 | long | 17.5 | 25 | native-capture |
| GLM-5.1 | 60 | 1.67 | 7.3 | long | 5.0 | 32 | native-capture |
| Kimi-K2.5 | 40 | 1.30 | 11.7 | long | 12.5 | 15 | native-capture |
| Claude-4.7-TI | 40 | 0.68 | 3.9 | short | 0 | 5 | reconstruction |
| DeepSeek-V4 | 60 | 0.60 | 8.7 | long | 0 | 12 | native-capture |
| Gemini-3.1 | 40 | 0.60 | 3.5 | med | 2.5 | 12 | native-capture |
| Claude-4.6-TI | 40 | 0.42 | 4.3 | short | 0 | 0 | reconstruction |
| Opus-4.6-filtered | 40 | 0.17 | 7.9 | med | 0 | 0 | surface-filtered |
| angrygiraffe (synthetic) | 40 | 0.05 | 5.5 | short | 0 | 0 | synthetic |
| Roman-Claude (direct) | 35* | 0.00 | 0.5 | tiny | 0 | 0 | direct-answer |

By method group (unweighted dataset means): native-capture CCR **1.25**, reconstruction/synthetic **0.38**, filtered/direct **0.09**.

**But length is the dominant confounder.** Pearson(log chars, CCR) = **0.654** — length predicts closure *better than critique density does* (Pearson 0.31). Native traces average **11,007 chars**; reconstructions **1,838** (6× shorter). When we restrict to a comparable length band (1k–8k chars), the native-vs-reconstruction gap shrinks from 3.2× to **2.3× (native 0.95 vs reconstruction 0.42)**.

So roughly two-thirds of the raw cross-dataset gap is **observability**: a short, polished, final-answer-conditioned trace cannot *show* a correction loop, whether or not the teacher performed one. A real residual method effect (~2.3× at equal length) remains, but the strong original claim ("reconstruction destroys the topology") is not supported; the supported claim is:

> **Reconstruction/synthesis/filtering yields short, polished training text in which self-correction is largely unobservable. For a student that learns only from the text, unobservable equals absent — but we cannot attribute the gap purely to topology loss; trace length/rawness explains most of it.**

The one place this isn't just length: **Opus-4.6-filtered** (density 7.9, medium length, CCR 0.17) and **DeepSeek** (long, high density, CCR 0.60) are high-density / low-closure regardless of length — the dissociation between vocabulary and closure is genuine there.

---

## 3. Domain is not the explanation (a control that holds)

Restricting to **math-domain traces only** (a subset control computed on the judged traces; not in the committed `report_*.json`): Qwen ≈2.5 ≫ Claude-4.6-TI ≈0.4 ≫ Opus-filtered ≈0.2 ≫ Roman ≈0.03. The ordering persists within a single domain, so domain doesn't drive the cross-dataset gap. (The exception is angrygiraffe, whose roleplay/humanities content genuinely has little to correct — its low closure is partly domain, partly synthesis.)

---

## 4. Did the judge earn trust? (RQ3 — honestly)

- **Self-consistency (repeatability):** re-judging 45 traces agreed 95.6% within ±1 CCR (mean abs diff 0.29). The instrument is repeatable.
- **Cross-judge agreement (validity):** vs the Kimi judge (n=30), exact 46.7%, **quadratic-weighted κ = 0.35 (only "fair")**, Opus systematically stricter (1.2 vs 1.8).

**Construct validity is NOT established.** We have a repeatable, conservative instrument, not a human-validated one (the originally-planned human-κ leg was not run). Two LLM judges agree only fairly. Therefore: absolute CCR values are judge-relative; what is durable is (a) the **paired natural-experiment effect** and (b) **rankings**, both of which are large enough to survive judge substitution in direction if not magnitude. Treat single-judge magnitudes with caution.

---

## 5. Convergence — `mindset` hypotheses after review

| from `mindset` | verdict (post-review) |
|---|---|
| H2 cognitive inertia (overthrow) | ✅ supported — overthrow% 0 for all non-native methods vs 12–32% native; clean in the paired test (31.7 vs 11.7) |
| H1 reasoning topology | ◑ supported but length-entangled — loop topology needs trace length to appear |
| Gemini metacognition (coupling) | ⚠️ a rubric *component* of CCR, so not external validation — descriptive only |
| Open-reasoning / routinization | ◑ nuanced (Qwen templated-yet-resolving; angrygiraffe divergent-yet-uncorrecting) |
| H3 format friction / H4 kernel | ⏸ not testable on this corpus |
| value-anchoring / ToM / social | ❌ dropped — data lacks the content |

The original prior ("Claude = closure gold standard") was **falsified** — but the replacement is *not* a clean "method determines topology" law; it is "**native-capture + a self-revising teacher + sufficient trace length** are jointly needed for observable closure, and the cleanest single lever is the teacher (the natural experiment)."

---

## 6. Aligning with training an LLM

Practical guidance survives even where the mechanism is uncertain — because a student learns from the *text*:

1. **Filter training data on judged closure (CCR ≥ 2), not on critique-word density.** Density (Pearson 0.31) and even length (0.65) are weak proxies; the LLM-judged CCR is the direct signal, and it is ~$0/dataset to compute.
2. **Prefer native-capture over reconstruction/synthesis** — not because reconstruction provably erases reasoning, but because it yields text where correction is unobservable, and unobservable correction can't be learned.
3. **Choose the teacher by measured closure, not benchmarks** — the natural experiment shows teacher is the cleanest lever (GLM ≫ DeepSeek on identical questions).
4. **Falsifiable next step (closes H4):** the cached `Jackrong/Qwopus3.6-27B-v1-preview` was trained on this ecosystem. Test: does a model trained on low-closure data itself produce low-closure reasoning? If yes, this cheap data-level audit becomes a predictive pre-training filter, and our own finetune should curate by CCR first.

---

## 7. Limitations (the review's surviving objections)

- **Length/observability confound** — explains ~⅔ of the cross-dataset gap (§2). The method claim is correspondingly weakened; the natural experiment is the load-bearing result.
- **Method/teacher entanglement** — all Claude sets are reconstructions; no native Claude `<think>` baseline exists in this corpus.
- **Single-judge, no human ground truth** — repeatable but not validated (κ=0.35 cross-judge); magnitudes are judge-relative.
- **Coupling≈CCR is internal** — not independent corroboration.
- **n=40/dataset, cleaning bias upstream, ultra-long truncation** — as before.
- **Cross-dataset correlations (Spearman 0.52, n=10) are directional, not significant** — do not over-read.

> **Bottom line (revised).** The strong, clean result is the natural experiment: a self-revising teacher transmits the *act* of correction through native-capture distillation, and it does so while using fewer correction *words*. The broader "native beats reconstruction" pattern is real but mostly a trace-length/observability effect. For training, the operational lesson is unchanged and cheap: **audit each candidate trace for coupled closure and filter on that — never on the vocabulary.**

---

## 8. Structural-metric cross-validation (the mindset/Opus pure-text metrics)

The original `mindset/Opus` plan proposed *zero-cost* pure-text structural metrics
(CCS, PDD, IS, EAR, OED, RT topology, RAE entropy). We computed all of them on the
440 full traces (`scripts/run_structural.py`) and correlated each with the LLM CCR.

**Per-trace** (Pearson): weak — best ≈0.37 (RAE entropy, RT branch-merge). **CCS,
`final_decision.md`'s flagship "most ingenious" metric, has Pearson ≈ −0.01** and
its dataset means barely move (1.02–1.21). As a per-trace judge of closure, the
structural proxies fail — which is the empirical justification for using an LLM judge.

**Per-dataset ranking** (Spearman of dataset means vs LLM-CCR dataset means): several
are strong — **RAE bigram entropy 0.927**, **ICI 0.821**, **OED 0.81**, **RT topology
density 0.766**, PDD 0.743, CCS 0.555. So RAE and RT independently reproduce the LLM's dataset ranking,
a free cross-validation of the headline ordering.

**Conclusion:** the Opus structural framework is *aggregate-valid but trace-invalid*
— good for ranking corpora, not for scoring individual traces. CCS specifically
under-delivered relative to its billing. This both vindicates the LLM-judge pivot
(per-trace) and corroborates the cross-dataset findings (aggregate, for free).
Outputs: `outputs/structural_metrics.jsonl`, `outputs/report_structural.json`.

---

## 9. Gemini advanced metrics + information-theoretic check (the "garnish", completed)

Per `mindset/Gemini/{advanced_metrics_exploration, first_principles_perspectives}.md`,
all computable metrics were added (pure text/graph, zero API):

- **Aesthetics** — CE (logical compression), NRR (filler/noise), BSD (branch out-degree),
  ICI (PageRank-Gini hub concentration). Best CCR predictor is **ICI** (Pearson 0.37);
  CE 0.29, BSD 0.27, NRR ~0. (LSI logical-symmetry = future work per the doc — needs
  reverse-problem generation.)
- **MI-collapse** (first-principles flagship) — `I(critique_present ; answer_changed)`
  per dataset. The clean result: **nohurry-Opus and Roman-Claude = 0.0 bits** — critique
  vocabulary is *fully decoupled* from any answer change, the predicted information-theoretic
  collapse. Pooled MI = 0.38 bits. Caveat: low MI can also reflect near-universal critique
  (Qwen), where `critique_present` has little variance — so MI flags ritual where critique is
  *occasional but inert* (nohurry), not where it is ubiquitous.

These corroborate the headline from a second, model-free direction: the datasets whose
critique is most decoupled from correction (nohurry, roman) are exactly the lowest-closure,
surface-filtered/direct-answer sets. Full completeness map of every mindset/Opus + Gemini
requirement: `COMPLETENESS.md`. Outputs: `report_aesthetics.json`, `aesthetics_metrics.jsonl`.

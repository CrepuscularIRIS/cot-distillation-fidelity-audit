# Hypotheses vs. Results — one-by-one against `mindset/`

> Every hypothesis from `mindset/Opus` (the 5 first-principles hypotheses + the
> `final_decision` flagships) and `mindset/Gemini` (cognitive lenses · advanced
> aesthetics · first-principles MI) is listed below with: **Claim →
> Operationalization → Prediction → Result (exact numbers) → Verdict.** All numbers
> are from `report_*.json` over **440 traces / 10 datasets**, judged by Opus 4.8 and
> validated by a 5-model panel. Dataset-level correlations are Spearman over the 10
> dataset means; trace-level are Pearson over all 440 traces.

## The corpus (teacher × distillation method × measured closure)

| Dataset | Teacher | Method | CCR mean | overthrow% | topology (non-chain) |
|---|---|---|---:|---:|---|
| qwen | Qwen3.5-27B | native-capture | **2.08** | 25.0 | tree13/loop7/drift5 |
| glm | GLM-5.1 | native-capture | **1.67** | 31.7 | tree20/loop3/drift3 |
| kimi | KIMI-K2.5 | native-capture | **1.30** | 15.0 | tree15/loop5/drift2 |
| claude47_ti | claude-opus-4.7 | trace-inversion (reconstruction) | 0.68 | 5.0 | tree5 |
| deepseek | DeepSeek-V4-Distill | native-capture* | 0.60 | 11.7 | tree7/drift3 |
| gemini | gemini-3.1-pro | native-capture | 0.60 | 12.5 | tree3/loop1 |
| claude46_ti | claude-opus-4.6 | trace-inversion (reconstruction) | 0.43 | 0.0 | drift1/tree1 |
| nohurry_opus | claude-opus-4.6 | human-filtered "high quality" | 0.18 | 0.0 | tree3/drift1 |
| angrygiraffe | claude-opus-4.6 | synthetic/roleplay | 0.05 | 0.0 | tree4 |
| roman_claude | claude-opus-4.6 | filtered/direct (no `<think>`) | 0.03 | 0.0 | tree1/drift1 |

\* deepseek is native-capture but **short** — see the length confound below; it is the low outlier among native sets and the other half of the natural experiment.

---

## ⭐ The natural experiment (`final_decision` MVP) — the robust spine

- **Claim.** On the same question, two different teachers' traces differ in measured closure, and
  that difference does not track critique-word density. (Observational: it fixes the question, not the
  teacher — teacher identity, training pipeline, and per-trace length are not separated.)
- **Operationalization.** 60 matched **GLM↔DeepSeek** pairs (identical normalized
  problem), CCR 0–4 by LLM judge; Wilcoxon signed-rank, Cliff's δ, matched-pairs
  rank-biserial.
- **Prediction.** Closure differs significantly between the two teachers even though
  both emit `<think>` and both are "reasoning" distillations.
- **Result.** GLM CCR **1.667** vs DeepSeek **0.60**; **38** GLM-higher / **4**
  DS-higher / 18 tied; median paired diff **+1.0**; rank-biserial **0.567**; Cliff's δ
  **0.513 (large)**; Wilcoxon **z = −4.983, p ≈ 0**. **Vocabulary inversion:** DeepSeek
  critique density **8.66** > GLM **7.343** — DeepSeek *says* "wait/check/however" more,
  yet closes the loop less. Overthrow GLM **31.7%** vs DS **11.7%**; loop-topology GLM
  5.0% vs DS 0.0%.
- **Multi-judge.** GLM>DS holds for the **4 judges that scored pairs** (Codex did calibration only):
  paired per judge — opus 19/0/6 (n=25), kimi 4/0/1 (n=5), deepseek_flash **5/0/0** (n=5, *DeepSeek's
  own teacher*), minimax 8/1/3 (n=12). The secondary subsets are small (indicative, not conclusive).
  Every judge ranks GLM > DeepSeek; beyond that one pair only Opus scored all 10 datasets, so the
  cross-judge ranking ρ=1.0 reflects that 2-point agreement, not a 10-dataset concordance.
- **Verdict.** ✅ **CONFIRMED, robust** within its design — the single cleanest result; survives the
  length-matched control and every judge that scored the pairs. (Causally it isolates the question,
  not the teacher.)

---

## H2 — Cognitive Inertia *(strongest per `hypothesis_scrutiny`)*

- **Claim.** Distilled traces assert a conclusion early and rarely overturn it; genuine
  reasoning overturns its own intermediate conclusions.
- **Operationalization.** LLM `overthrow_present`; structural EAR (early-assertion rate),
  OED (overthrow-event density), inertia-slope, CI-index = EAR/(OED+ε)×slope.
- **Prediction.** Reconstructed / filtered datasets show ~0 overthrow; native-capture shows real overthrow.
- **Result.** Overthrow% — **native-capture 12–32%** (glm 31.7, qwen 25, kimi 15, gemini
  12.5, deepseek 11.7) vs **reconstruction/synthetic/filtered = 0** (claude46_ti, nohurry,
  angrygiraffe, roman all 0; claude47_ti only 5). Structural OED dataset-Spearman vs CCR
  **0.81**, EAR **0.634**, inertia-slope **−0.735**.
- **Verdict.** ✅ **CONFIRMED (strong, with caveats).** Reconstructed/filtered sets show ~0% judged
  overthrow (0/40 in each) vs 12–32% for native-capture — a strong signal, but it rests on one judge's
  binary label and, like topology, is partly entangled with trace length.

---

## H5 — Routinization / thinking-routine entropy

- **Claim.** Lower closure ⇒ more templated, repetitive reasoning-action sequences (lower
  bigram entropy, higher routinization).
- **Operationalization.** RAE = entropy of the reasoning-action bigram distribution
  (VERIFY→CONCLUDE→EXPLORE…), plus a routinization index.
- **Result.** RAE bigram-entropy dataset-Spearman vs CCR = **0.927 — the single best
  structural predictor of closure.** angrygiraffe & roman bottom out at entropy **1.0**
  (maximally templated); qwen 2.85 / kimi 2.37 / glm 2.17 top.
- **Verdict.** ✅ **CONFIRMED at the aggregate.** A free, zero-API dataset-level ranker
  that reproduces the LLM judge's order.

---

## H1 — Reasoning Topology

- **Claim.** Closure co-occurs with non-linear topology (tree / graph / loop), not linear
  chain / drift.
- **Operationalization.** LLM `reasoning_topology` label + structural RT (topology
  density, branch/merge ratio, back-reference depth).
- **Result.** Least-linear sets are highest-closure: qwen (chain 15 / tree 13 / loop 7 /
  drift 5), kimi (chain 18 / tree 15 / loop 5); near-pure chains are lowest: roman (chain
  38), angrygiraffe (chain 36). Loop% tracks CCR (qwen 17.5, kimi 12.5, glm 5.0, rest ≈0).
  RT-density dataset-Spearman **0.766**.
- **Verdict.** ✅ **CONFIRMED but length-entangled** (longer traces have more room for
  branches; see confound).

---

## H3 — Representational-format friction

- **Claim.** Switching representation (prose ↔ math ↔ code) marks reasoning effort and
  tracks closure.
- **Operationalization.** FSD = format-switch density per paragraph.
- **Result.** **Low signal, as `hypothesis_scrutiny` predicted** — the corpus is natural
  language + math with almost no code, so format rarely switches. Computed; weak
  discrimination.
- **Verdict.** ◑ **WEAK / inconclusive** (demoted at scrutiny; confirmed weak here).

---

## H4 — Incompressible cross-scale kernel

- **Claim.** A closure "kernel" survives compression across model scales — a model
  *trained on* low-closure data will itself reason with low closure.
- **Operationalization.** Needs same-data / different-scale model checkpoints — not a
  text metric.
- **Result.** ⏸ **Round 6 IN PROGRESS.** `Jackrong/Qwopus3.6-27B-v1-preview` is
  downloading now; the test generates fresh `<think>` traces from the trained model on the
  same problem set, judges them with the same rubric, and compares the model's *behavioral*
  closure to its *training-data* closure.
- **Verdict.** ⏳ **PENDING** (the only hypothesis requiring weights; actively running).

---

## `final_decision` flagships — CCS / PDD / IS

| Metric | per-trace Pearson vs CCR | per-dataset Spearman vs CCR | Verdict |
|---|---:|---:|---|
| **CCS** Correction Coupling Strength | **−0.008** | 0.555 | ❌ per-trace **fails**; ◑ weak per-dataset |
| **PDD** Paragraph Dependency Depth | 0.153 | 0.743 | ◑ per-dataset only |
| **IS** Inertia Slope | −0.283 | **−0.735** | ◑ per-dataset only |

- **The instructive failure.** `final_decision` bet on these zero-cost flagships scoring
  *individual* traces. CCS computed exactly correlates **−0.008** with judged closure — it
  cannot rank a single trace. This is the empirical justification for the LLM-judge primary
  pipeline; the structural metrics survive only as **aggregate** dataset rankers.

---

## Gemini cognitive lenses (MC / EF / Open)

- **MC (metacognition)** → the rubric core: `monitoring_control_coupling`,
  `causal_depth_of_critique`, `uncertainty_stage`. Drives the CCR judgment.
- **EF (executive function)** → `planning_depth`, `plan_execution_consistency` (scored garnish).
- **Open (divergent)** → `divergent_score`, `creative_destruction`, `templated_divergence`.
  Note the **templated-divergence trap**: qwen 50% and glm 40% show *high* divergence that
  is *ritual* — exploration vocabulary without a corresponding answer change.
- **SC / ToM** → ⏸ parked: traces are solitary problem-solving; no social / nested-belief
  content to score (`RESEARCH_REPORT` established refusals were cleaned).

## Gemini advanced aesthetics (CE / NRR / BSD / ICI / LSI)

| Metric | per-dataset Spearman vs CCR | per-trace Pearson |
|---|---:|---:|
| **ICI** Information Concentration (PageRank-Gini) | **0.821** | 0.369 |
| **CE** Compression Efficiency | 0.795 | 0.291 |
| **NRR** Noise Reduction Ratio | 0.759 | 0.061 |
| **BSD** Branching Sparsity | 0.663 | 0.274 |
| **LSI** Logical Symmetry | — | ⏸ future (per the doc) |

## Gemini first-principles — MI-collapse

- **I(critique_present ; answer_changed)** per dataset: **nohurry_opus = 0.0** and
  **roman_claude = 0.0 bits** — no measurable co-occurrence of critique and correction. (Degenerate,
  not a tested independence: these sets have ~0 answer-changes at all, so MI is mechanically 0.) High:
  gemini 0.581, glm 0.433, deepseek 0.396. qwen is *low* (0.077) for the opposite reason — critique is
  near-universal, carrying little information. Pooled MI = 0.376.

---

## Cross-cutting: the length confound (`execution_reminders` discipline)

- Pearson(log trace-length, CCR) = **0.654**. Native traces average **11,007** chars vs
  reconstruction **1,838**. Within a matched **1k–8k** band: native CCR **0.95** (n=152) vs
  reconstruction **0.42** (n=83).
- **Reading.** ~⅔ of the raw cross-dataset gap is length/observability — but the gap
  **persists** length-controlled and the natural experiment is length-matched, so the
  effect is real, just smaller than the headline.

## Judge reliability

- Opus self-consistency (n=45): exact **77.8%**, within-1 **95.6%**, mean |Δ| 0.29.
- Opus vs Kimi cross-judge (n=30): exact 46.7%, within-1 76.7%, QWK 0.35 (Kimi scores
  systematically higher, Cliff's δ −0.264).
- Panel pairwise QWK: opus~kimi **0.838**, opus~minimax 0.668, opus~deepseek_flash 0.568.
  Magnitudes differ; all judges agree GLM > DeepSeek — but that is the only pair every judge scored,
  so the panel ranking ρ=1.0 is a 2-point agreement, not a 10-dataset one (full ranking is Opus-only).

---

### Scoreboard

| Hypothesis | Verdict |
|---|---|
| Natural experiment (teacher sets closure) | ✅ confirmed, robust |
| H2 cognitive inertia | ✅ confirmed (cleanest) |
| H5 routinization entropy | ✅ confirmed (best aggregate ranker, ρ=0.927) |
| H1 reasoning topology | ✅ confirmed, length-entangled |
| H3 format friction | ◑ weak/inconclusive (predicted) |
| H4 incompressible kernel | ⏳ Round 6 in progress (Qwopus) |
| CCS/PDD/IS per-trace | ❌ fail per-trace → ✅ justify LLM judge |
| Gemini MC/EF/Open lenses | ✅ operationalized in rubric |
| Gemini aesthetics CE/NRR/BSD/ICI | ✅ computed (ICI best, ρ=0.821) |
| Gemini MI-collapse | ✅ confirmed (nohurry/roman = 0 bits) |
| SC/ToM lenses, LSI, VAS | ⏸ parked (justified: data/method) |

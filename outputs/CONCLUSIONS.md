# 结论报告 — Distillation Reasoning-Fidelity Audit
### Conclusions Report

> 10 datasets · 440 traces · primary judge Claude Opus 4.8 · validated by a 5-model judge fleet (Kimi, DeepSeek-V4-Flash, MiniMax-M3, Codex GPT-5.4) + the mindset/Opus structural metrics.
> Full report: `REPORT_distill_fidelity.md`. All numbers: `outputs/report_*.json`, `opus_all.jsonl`, `structural_metrics.jsonl`.

---

## 1. The question

Public reasoning-distillation datasets are full of self-critique vocabulary
("wait", "let me check", "however"). Does that vocabulary correspond to a real
**critique → correction → verification** loop, or is it ritual? We scored every
trace 0–4 on closure (CCR), entry-by-entry, with an LLM judge.

## 2. The robust conclusion (survives every control and every judge)

**On identical questions, the teacher that genuinely revises itself transmits that
habit through distillation — even though it uses *fewer* critique words.**

The natural experiment: DeepSeek-V4-Distill answers GLM-5.1's exact 60 questions
(same questions, same native-capture method, comparable length).

- **GLM CCR 1.67 vs DeepSeek 0.60.** GLM higher on 38/60 pairs, DeepSeek on 4, 18 tied.
- Matched-pairs rank-biserial 0.567; Wilcoxon **p < 1e-6**.
- DeepSeek has **higher** critique density (8.66 vs 7.34) yet lower closure, and 0% loop topology vs GLM's; its dominant failure is `surface_marker_only` (34/60).
- **Validated by the 4 judges with paired scores** (Opus, Kimi, DeepSeek-Flash, MiniMax-M3; Codex was used for calibration but scored no pairs): in the subset each scored (n = 25 / 5 / 5 / 12 pairs), GLM is rated higher in the large majority — **DeepSeek-Flash, DeepSeek's own teacher, agreed on all 5 it scored.** The secondary subsets are small (indicative, not conclusive). Across the only two datasets all judges scored (GLM, DeepSeek) every judge ranks GLM above DeepSeek — a 2-point rank correlation, so the ρ=1.0 reflects that single shared ordering, not a 10-dataset agreement.

**The vocabulary of self-correction points the wrong way. The act of self-correction
is the real signal, and it co-varies with teacher identity** (teacher, training
pipeline, and per-trace length are not separated by this observational comparison).

## 3. The honest cross-dataset conclusion (after adversarial review)

Across all 10 datasets the closure ordering is qwen 2.08 > glm 1.67 > kimi 1.30 >
claude47-TI 0.68 > deepseek 0.60 ≈ gemini 0.60 > claude46-TI 0.42 > opus-filtered
0.18 > angrygiraffe 0.05 > roman 0.03. By distillation method (unweighted dataset
means): native-capture 1.25, reconstruction/synthetic 0.38, filtered/direct 0.10 —
but see the length caveat below: the one *short* native set (DeepSeek) ranks with the
reconstructions, so method and length are not separable at the dataset level.

But **~⅔ of that gap is a trace-length / observability effect, not topology loss.**
Closure correlates with length at Pearson 0.65 (stronger than with critique density,
0.31); native traces are 11k chars vs reconstructions' 1.8k. Length-matched, the gap
shrinks from 3.2× to 2.3×. The defensible claim:

> Reconstruction / synthesis / surface-filtering produce short, polished training
> text in which self-correction is largely **unobservable**. For a student that learns
> only from the text, unobservable = unlearnable — but we cannot claim the teacher's
> reasoning was destroyed; mostly it was never written down.

Genuine vocabulary-vs-closure dissociations remain where length can't explain them:
**opus-filtered** (density 7.9, CCR 0.17 — a human "high-quality" filter that selected
critique words, not closure) and **DeepSeek** (long, dense, low closure).

## 4. Methodology conclusions (what we learned about measuring this)

- **Per-trace, an LLM judge is necessary.** The mindset/Opus pure-text structural
  metrics, computed exactly: CCS (`final_decision.md`'s flagship) has Pearson **−0.01**
  with closure; the best (RAE entropy, RT density) reach only ~0.37. They cannot score
  an individual trace.
- **At the dataset level, the structural metrics work for free.** RAE bigram entropy
  (Spearman 0.927) and RT topology density (0.766) reproduce the LLM dataset ranking —
  a zero-API corroboration of the headline.
- **Single-judge validity is not established, but it doesn't need to be.** Opus is
  repeatable (95.6% within-1) but cross-judge κ is only fair (0.35). The 5-model fleet
  resolves this: judges differ on exact magnitudes (QWK 0.36–0.86) but agree on order
  and on the headline. **Rankings and the paired effect are durable; absolute scores
  are judge-relative.**

## 5. Practical conclusion — for training an LLM

1. **Filter training data on judged closure (CCR ≥ 2), never on critique-word density.**
   Density and length are weak proxies; opus-filtered proves density-filtering selects
   the wrong traces. The CCR judge is ~$0/dataset.
2. **Prefer native-capture distillation** (record the teacher's real `<think>` stream)
   over reconstruction/synthesis — the latter yields text where correction is unobservable.
3. **Pick the teacher by measured closure, not benchmarks** — teacher is the cleanest lever.
4. **Falsifiable next step (Round 6):** `Jackrong/Qwopus3.6-27B-v1-preview` is cached.
   Test whether a model trained on low-closure data itself produces low-closure reasoning.
   If yes, this cheap data audit becomes a predictive pre-training filter.

## 6. Coverage & limitations

- **All 10 datasets analyzed** (Jackrong, Roman, Opus families); all mindset/Opus
  structural metrics computed; Gemini cognitive lenses = next (garnish).
- Limits: single primary judge (mitigated by the 5-model fleet); method/teacher
  entanglement (all Claude sets are reconstructions; no native Claude baseline);
  n=40/dataset; upstream cleaning bias; ultra-long truncation for the LLM judge.

---

> **One line.** Distillation copies the *sound* of a mind correcting itself far more
> easily than the *act*; audit each trace for the act — cheaply — before you train on it.

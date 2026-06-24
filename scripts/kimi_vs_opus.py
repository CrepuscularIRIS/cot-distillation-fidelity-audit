#!/usr/bin/env python3
"""Opus-vs-Kimi cross-judge calibration over the SAME traces (index-joined).

Answers Codex's RQ3 objection: not just point agreement (QWK), but whether the
DATASET RANKINGS and the native-vs-reconstruction gap survive judge substitution.

Reads outputs/batches/<f>, outputs/opus_scores/<f>, outputs/kimi_scores/<f>.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BATCH, OPUS, KIMI = OUT / "batches", OUT / "opus_scores", OUT / "kimi_scores"


def load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def qwk(pairs, K=5):
    O = [[0] * K for _ in range(K)]
    for a, b in pairs:
        if 0 <= a < K and 0 <= b < K:
            O[a][b] += 1
    n = len(pairs)
    row = [sum(O[i]) for i in range(K)]
    col = [sum(O[i][j] for i in range(K)) for j in range(K)]
    W = [[((i - j) ** 2) / ((K - 1) ** 2) for j in range(K)] for i in range(K)]
    num = sum(W[i][j] * O[i][j] for i in range(K) for j in range(K))
    den = sum(W[i][j] * row[i] * col[j] / n for i in range(K) for j in range(K))
    return round(1 - num / den, 3) if den else None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 3) if n > 1 else None


def main():
    pairs = []                       # (opus_ccr, kimi_ccr)
    by_ds = defaultdict(lambda: {"opus": [], "kimi": []})
    for bf in sorted(BATCH.glob("*.json")):
        if bf.name == "manifest.json":
            continue
        batch, op, km = load(bf), load(OPUS / bf.name), load(KIMI / bf.name)
        if not batch or not op or not km:
            continue
        if not (len(batch) == len(op) == len(km)):
            continue
        for i in range(len(batch)):
            o, k = op[i].get("ccr_closure"), km[i].get("ccr_closure")
            if isinstance(o, int) and isinstance(k, int):
                pairs.append((o, k))
                ds = batch[i].get("dataset")
                by_ds[ds]["opus"].append(o)
                by_ds[ds]["kimi"].append(k)

    if not pairs:
        print("No overlapping Opus+Kimi judgments yet.")
        return

    n = len(pairs)
    exact = sum(1 for a, b in pairs if a == b)
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1)
    opus_mean = sum(a for a, b in pairs) / n
    kimi_mean = sum(b for a, b in pairs) / n
    # bias-by-score: for each Opus level, mean Kimi
    bias = {}
    for lvl in range(5):
        ks = [b for a, b in pairs if a == lvl]
        if ks:
            bias[lvl] = round(sum(ks) / len(ks), 2)

    ds_names = sorted(by_ds)
    ds_opus_means = [sum(by_ds[d]["opus"]) / len(by_ds[d]["opus"]) for d in ds_names]
    ds_kimi_means = [sum(by_ds[d]["kimi"]) / len(by_ds[d]["kimi"]) for d in ds_names]
    rank_corr = spearman(ds_opus_means, ds_kimi_means)

    report = {
        "n_overlap": n,
        "exact_agree_pct": round(100 * exact / n, 1),
        "within1_agree_pct": round(100 * within1 / n, 1),
        "quadratic_weighted_kappa": qwk(pairs),
        "opus_mean": round(opus_mean, 2),
        "kimi_mean": round(kimi_mean, 2),
        "opus_minus_kimi_bias": round(opus_mean - kimi_mean, 2),
        "kimi_mean_by_opus_score": bias,
        "dataset_rank_spearman_opus_vs_kimi": rank_corr,
        "per_dataset": {d: {"n": len(by_ds[d]["opus"]),
                            "opus_mean": round(sum(by_ds[d]["opus"]) / len(by_ds[d]["opus"]), 2),
                            "kimi_mean": round(sum(by_ds[d]["kimi"]) / len(by_ds[d]["kimi"]), 2)}
                        for d in ds_names},
    }
    (OUT / "report_kimi_vs_opus.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n=== OPUS vs KIMI cross-judge calibration (n={n} overlapping traces) ===")
    print(f"  exact {report['exact_agree_pct']}% | within-1 {report['within1_agree_pct']}% | QWK {report['quadratic_weighted_kappa']}")
    print(f"  Opus mean {report['opus_mean']} vs Kimi mean {report['kimi_mean']} (Opus-Kimi bias {report['opus_minus_kimi_bias']})")
    print(f"  Kimi mean by Opus score: {bias}")
    print(f"  DATASET-RANK Spearman(Opus, Kimi) = {rank_corr}  <- do rankings survive judge swap?")
    print(f"\n  {'dataset':14s} {'n':>3s} {'opus':>5s} {'kimi':>5s}")
    for d in ["deepseek", "glm", "claude46_ti", "claude47_ti", "nohurry_opus", "gemini", "kimi", "qwen", "angrygiraffe", "roman_claude"]:
        if d in report["per_dataset"]:
            r = report["per_dataset"][d]
            print(f"  {d:14s} {r['n']:>3d} {r['opus_mean']:>5.2f} {r['kimi_mean']:>5.2f}")


if __name__ == "__main__":
    main()

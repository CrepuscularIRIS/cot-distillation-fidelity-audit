#!/usr/bin/env python3
"""4-judge panel analysis: Opus (full) + Kimi + DeepSeek-V4-Flash + MiniMax-M3
(on the sampled batches). Tests whether the headline (GLM>DeepSeek) and the
dataset rankings survive a judge swap, plus inter-judge agreement and self-bias.
Index-joins batch + opus_scores + panel/<backend> per file.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BATCH, OPUS, PANEL = OUT / "batches", OUT / "opus_scores", OUT / "panel"
JUDGES = ["opus", "kimi", "deepseek_flash", "minimax_m3", "codex"]
SAMPLE = ([f"paired_{i:03d}.json" for i in range(10)] +
          [f"{d}_000.json" for d in ["claude46_ti", "claude47_ti", "nohurry_opus",
                                      "gemini", "kimi", "qwen", "angrygiraffe", "roman_claude"]])


def load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def ccr_list(scores):
    return scores if scores else None


def spearman(x, y):
    if len(x) < 2:
        return None
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for pos, i in enumerate(o):
            r[i] = pos + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x); d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 3)


def qwk(pairs, K=5):
    if len(pairs) < 3:
        return None
    O = [[0] * K for _ in range(K)]
    for a, b in pairs:
        if 0 <= a < K and 0 <= b < K:
            O[a][b] += 1
    n = len(pairs); row = [sum(O[i]) for i in range(K)]; col = [sum(O[i][j] for i in range(K)) for j in range(K)]
    W = [[((i - j) ** 2) / ((K - 1) ** 2) for j in range(K)] for i in range(K)]
    num = sum(W[i][j] * O[i][j] for i in range(K) for j in range(K))
    den = sum(W[i][j] * row[i] * col[j] / n for i in range(K) for j in range(K))
    return round(1 - num / den, 3) if den else None


def main():
    # rows[(file,i)] = {"dataset","pair_ds", judge->ccr}
    rows = {}
    for f in SAMPLE:
        batch = load(BATCH / f)
        if not batch:
            continue
        sources = {"opus": load(OPUS / f)}
        for b in ["kimi", "deepseek_flash", "minimax_m3", "codex"]:
            sources[b] = load(PANEL / b / f)
        for i, m in enumerate(batch):
            rec = {"dataset": m.get("dataset"), "pair_ds": m.get("pair_ds"), "j": {}}
            for jname, sc in sources.items():
                if sc and i < len(sc) and isinstance(sc[i].get("ccr_closure"), int):
                    rec["j"][jname] = sc[i]["ccr_closure"]
            rows[(f, i)] = rec

    have = {j: sum(1 for r in rows.values() if j in r["j"]) for j in JUDGES}
    print("=== judges available (n scored on sample) ===")
    for j in JUDGES:
        print(f"  {j:14s}: {have[j]}")

    # 1) per-judge dataset means
    ds_means = defaultdict(dict)  # dataset -> judge -> mean
    by_ds_judge = defaultdict(lambda: defaultdict(list))
    for r in rows.values():
        for j, v in r["j"].items():
            by_ds_judge[r["dataset"]][j].append(v)
    for ds, jm in by_ds_judge.items():
        for j, vs in jm.items():
            ds_means[ds][j] = round(sum(vs) / len(vs), 2)

    datasets = [d for d in ["qwen", "glm", "kimi", "claude47_ti", "deepseek", "gemini",
                            "claude46_ti", "nohurry_opus", "angrygiraffe", "roman_claude"] if d in ds_means]
    print("\n=== per-judge dataset CCR means (sample) ===")
    print(f"  {'dataset':14s} " + " ".join(f"{j[:8]:>8s}" for j in JUDGES))
    for ds in datasets:
        print(f"  {ds:14s} " + " ".join(f"{ds_means[ds].get(j,'-'):>8}" for j in JUDGES))

    # 2) ranking agreement: spearman of dataset-mean vectors between judges
    print("\n=== dataset-ranking agreement (Spearman of per-dataset means) ===")
    rank_corr = {}
    for a, b in combinations(JUDGES, 2):
        xs = [ds_means[d][a] for d in datasets if a in ds_means[d] and b in ds_means[d]]
        ys = [ds_means[d][b] for d in datasets if a in ds_means[d] and b in ds_means[d]]
        s = spearman(xs, ys)
        rank_corr[f"{a}~{b}"] = s
        print(f"  {a:14s} ~ {b:14s}: {s}")

    # 3) headline paired GLM vs DeepSeek, per judge
    print("\n=== headline: GLM vs DeepSeek (sampled pairs), per judge ===")
    pair_glm = defaultdict(dict)  # pair_ds -> judge -> {ds:.., glm:..}
    for r in rows.values():
        if r["pair_ds"] and r["dataset"] in ("deepseek", "glm"):
            for j, v in r["j"].items():
                pair_glm[r["pair_ds"]].setdefault(j, {})[r["dataset"]] = v
    paired_per_judge = {}
    for j in JUDGES:
        diffs = [(p[j]["glm"] - p[j]["deepseek"]) for p in pair_glm.values()
                 if j in p and "glm" in p[j] and "deepseek" in p[j]]
        if diffs:
            wins = sum(1 for d in diffs if d > 0); losses = sum(1 for d in diffs if d < 0)
            paired_per_judge[j] = {"n": len(diffs), "glm_higher": wins, "ds_higher": losses,
                                   "tied": len(diffs) - wins - losses,
                                   "mean_diff": round(sum(diffs) / len(diffs), 2)}
            print(f"  {j:14s}: n={len(diffs)} GLM>{wins} DS>{losses} tie={len(diffs)-wins-losses} mean_diff={paired_per_judge[j]['mean_diff']:+.2f}")

    # 4) pairwise QWK (point agreement)
    print("\n=== pairwise QWK (point agreement on shared traces) ===")
    pair_qwk = {}
    for a, b in combinations(JUDGES, 2):
        prs = [(r["j"][a], r["j"][b]) for r in rows.values() if a in r["j"] and b in r["j"]]
        pair_qwk[f"{a}~{b}"] = {"n": len(prs), "qwk": qwk(prs)}
        print(f"  {a:14s} ~ {b:14s}: n={len(prs)} qwk={qwk(prs)}")

    # 5) self-bias: deepseek_flash on deepseek rows; kimi on kimi rows
    print("\n=== self-bias check (teacher judging own student) ===")
    for jname, dsname in [("deepseek_flash", "deepseek"), ("kimi", "kimi")]:
        own = [r["j"][jname] for r in rows.values() if r["dataset"] == dsname and jname in r["j"]]
        others = [v for r in rows.values() if r["dataset"] == dsname
                  for jn, v in r["j"].items() if jn != jname]
        if own and others:
            print(f"  {jname} on '{dsname}' rows: self_mean={sum(own)/len(own):.2f} (n={len(own)}) "
                  f"vs others_mean={sum(others)/len(others):.2f} -> bias {sum(own)/len(own)-sum(others)/len(others):+.2f}")

    report = {"judges_n": have, "dataset_means": dict(ds_means), "ranking_spearman": rank_corr,
              "paired_per_judge": paired_per_judge, "pairwise_qwk": pair_qwk}
    (OUT / "report_panel.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nwrote outputs/report_panel.json")


if __name__ == "__main__":
    main()

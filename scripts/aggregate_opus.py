#!/usr/bin/env python3
"""Aggregate the Opus-4.8 judge workflow outputs into the full analysis.

Consumes:
  outputs/batches/<file>.json          (trace metadata: dataset, teacher, reasoning, pair tags)
  outputs/opus_scores/<file>.json      (Opus judgments per uid)
  outputs/opus_scores_verify/<file>    (independent re-judge, self-consistency)
  outputs/*_judged.jsonl               (prior Kimi judgments, for calibration)

Produces:
  outputs/opus_all.jsonl               (every scored trace, full metadata + scores)
  outputs/report_cross_dataset.json    (per-dataset summary table)
  outputs/report_paired.json           (GLM vs DeepSeek: Wilcoxon + Cliff's delta)
  outputs/report_verify.json           (Opus self-consistency)
  outputs/report_calibration.json      (Kimi vs Opus agreement)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BATCH = OUT / "batches"
SCORES = OUT / "opus_scores"
VERIFY = OUT / "opus_scores_verify"

_CRIT = re.compile(
    r"\b(wait|but|however|actually|reconsider|recheck|check|verify|mistake|wrong|"
    r"recompute|let me|hmm|oops|incorrect)\b|不对|等等|然而|重新|检查|验证|错误|修正",
    re.IGNORECASE,
)


def density(text: str) -> float:
    return round(1000.0 * len(_CRIT.findall(text or "")) / max(len(text or "") / 4.0, 1.0), 2)


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def cliffs_delta(x, y):
    if not x or not y:
        return 0.0
    more = sum(1 for a in x for b in y if a > b)
    less = sum(1 for a in x for b in y if a < b)
    return round((more - less) / (len(x) * len(y)), 3)


def interp_delta(d):
    a = abs(d)
    return "large" if a > 0.474 else "medium" if a > 0.33 else "small" if a > 0.147 else "negligible"


def wilcoxon(diffs):
    nz = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0]
    if len(nz) < 6:
        return {"n_nonzero": len(nz), "note": "too few non-zero pairs"}
    nz.sort()
    ranks = list(range(1, len(nz) + 1))
    wp = sum(r for r, (_, s) in zip(ranks, nz) if s > 0)
    wm = sum(r for r, (_, s) in zip(ranks, nz) if s < 0)
    n = len(nz)
    mean_w, std_w = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (min(wp, wm) - mean_w) / std_w if std_w else 0
    # two-sided normal approx p
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"n_nonzero": n, "W_plus": wp, "W_minus": wm, "z": round(z, 3),
            "p_approx": round(p, 5), "direction": "GLM>DS" if wp > wm else "DS>GLM"}


def mean(v):
    return round(sum(v) / len(v), 3) if v else None


def load_all_rows():
    manifest = load_json(BATCH / "manifest.json") or []
    rows = []
    missing = []
    for b in manifest:
        batch = load_json(BATCH / b["file"]) or []
        scores = load_json(SCORES / b["file"])
        if scores is None:
            missing.append(b["file"])
            continue
        # Index-join (positional): paired ds/glm traces share a uid, so a uid
        # dict collapses them. Batch order == score order (verified), so join by
        # position. Fall back to uid-join only if counts/order disagree.
        if len(batch) == len(scores) and all(
                batch[i].get("uid") == scores[i].get("uid") for i in range(len(batch))):
            paired_iter = zip(batch, scores)
        else:
            by_uid = {t["uid"]: t for t in batch}
            paired_iter = [(by_uid.get(s.get("uid")), s) for s in scores]
        for m, s in paired_iter:
            if not m:
                continue
            rows.append({
                "job": b["job"], "dataset": m.get("dataset"), "teacher": m.get("teacher"),
                "uid": s.get("uid"), "domain": m.get("domain"),
                "reasoning_chars": m.get("reasoning_chars"),
                "crit_density_1k": density(m.get("reasoning")),
                "pair_ds": m.get("pair_ds"), "pair_glm": m.get("pair_glm"),
                **{k: s.get(k) for k in (
                    "critique_present", "monitoring_control_coupling", "causal_depth_of_critique",
                    "answer_change", "verification_independence", "ccr_closure", "failure_type",
                    "reasoning_topology", "uncertainty_stage", "early_assertion", "overthrow_present",
                    "planning_depth", "plan_execution_consistency", "divergent_score",
                    "creative_destruction", "templated_divergence", "evidence", "rationale")},
            })
    return rows, missing


def summarize(rows):
    by_ds = defaultdict(list)
    for r in rows:
        if isinstance(r.get("ccr_closure"), int):
            by_ds[r["dataset"]].append(r)
    out = {}
    for ds, rs in sorted(by_ds.items()):
        ccr = [r["ccr_closure"] for r in rs]
        topo = Counter(r.get("reasoning_topology") for r in rs)
        fail = Counter(r.get("failure_type") for r in rs if r.get("failure_type"))
        out[ds] = {
            "n": len(rs),
            "ccr_mean": mean(ccr),
            "ccr_median": sorted(ccr)[len(ccr) // 2],
            "coupling_mean": mean([r["monitoring_control_coupling"] for r in rs if isinstance(r.get("monitoring_control_coupling"), int)]),
            "depth_mean": mean([r["causal_depth_of_critique"] for r in rs if isinstance(r.get("causal_depth_of_critique"), int)]),
            "density_mean": mean([r["crit_density_1k"] for r in rs]),
            "loop_pct": round(100 * topo.get("loop", 0) / len(rs), 1),
            "overthrow_pct": round(100 * sum(1 for r in rs if r.get("overthrow_present") == 1) / len(rs), 1),
            "early_assertion_pct": round(100 * sum(1 for r in rs if r.get("early_assertion") == 1) / len(rs), 1),
            "topology": dict(topo.most_common()),
            "answer_change_final_pct": round(100 * sum(1 for r in rs if r.get("answer_change") == "final") / len(rs), 1),
            "templated_divergence_pct": round(100 * sum(1 for r in rs if r.get("templated_divergence") == 1) / len(rs), 1),
            "failure_types": dict(fail.most_common(5)),
            "teacher": rs[0].get("teacher"),
        }
    return out


def paired_analysis(rows):
    pairs = defaultdict(dict)
    for r in rows:
        if r.get("pair_ds") and isinstance(r.get("ccr_closure"), int):
            pairs[r["pair_ds"]][r["dataset"]] = r
    valid = [(p["deepseek"], p["glm"]) for p in pairs.values() if "deepseek" in p and "glm" in p]
    if not valid:
        return {"n_pairs": 0}
    ds_ccr = [d["ccr_closure"] for d, g in valid]
    glm_ccr = [g["ccr_closure"] for d, g in valid]
    diffs = [g - d for d, g in zip(ds_ccr, glm_ccr)]
    n = len(valid)
    wins = sum(1 for x in diffs if x > 0)   # GLM > DS
    losses = sum(1 for x in diffs if x < 0)  # DS > GLM
    ties = sum(1 for x in diffs if x == 0)
    sorted_d = sorted(diffs)
    median_diff = sorted_d[n // 2] if n % 2 else (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2
    return {
        "n_pairs": n,
        "deepseek_ccr_mean": mean(ds_ccr), "glm_ccr_mean": mean(glm_ccr),
        "median_paired_diff_glm_minus_ds": median_diff,
        "pairs_glm_higher": wins, "pairs_ds_higher": losses, "pairs_tied": ties,
        "matched_pairs_rank_biserial": round((wins - losses) / n, 3),  # paired effect size
        "cliffs_delta_glm_vs_ds_unpaired": cliffs_delta(glm_ccr, ds_ccr),
        "delta_interp": interp_delta(cliffs_delta(glm_ccr, ds_ccr)),
        "wilcoxon": wilcoxon(diffs),
        "deepseek_density_mean": mean([d["crit_density_1k"] for d, g in valid]),
        "glm_density_mean": mean([g["crit_density_1k"] for d, g in valid]),
        "deepseek_loop_pct": round(100 * sum(1 for d, g in valid if d.get("reasoning_topology") == "loop") / len(valid), 1),
        "glm_loop_pct": round(100 * sum(1 for d, g in valid if g.get("reasoning_topology") == "loop") / len(valid), 1),
        "deepseek_overthrow_pct": round(100 * sum(1 for d, g in valid if d.get("overthrow_present") == 1) / n, 1),
        "glm_overthrow_pct": round(100 * sum(1 for d, g in valid if g.get("overthrow_present") == 1) / n, 1),
    }


def verify_consistency():
    manifest = load_json(BATCH / "manifest.json") or []
    cmp = []
    for b in manifest:
        vp = VERIFY / b["file"]
        sp = SCORES / b["file"]
        v, s = load_json(vp), load_json(sp)
        if not v or not s:
            continue
        sm = {x["uid"]: x.get("ccr_closure") for x in s}
        for x in v:
            a, c = x.get("ccr_closure"), sm.get(x.get("uid"))
            if isinstance(a, int) and isinstance(c, int):
                cmp.append((a, c))
    if not cmp:
        return {"n": 0}
    exact = sum(1 for a, c in cmp if a == c)
    within1 = sum(1 for a, c in cmp if abs(a - c) <= 1)
    return {"n": len(cmp), "exact_agree_pct": round(100 * exact / len(cmp), 1),
            "within1_agree_pct": round(100 * within1 / len(cmp), 1),
            "mean_abs_diff": round(sum(abs(a - c) for a, c in cmp) / len(cmp), 2)}


def calibration(rows):
    # Kimi scores from prior *_judged.jsonl, keyed by uid
    kimi = {}
    for f in OUT.glob("*_judged.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d.get("ccr_closure"), int):
                kimi[d.get("uid")] = d["ccr_closure"]
    cmp = [(r["ccr_closure"], kimi[r["uid"]]) for r in rows
           if isinstance(r.get("ccr_closure"), int) and r.get("uid") in kimi]
    if not cmp:
        return {"n": 0, "note": "no overlapping uids between Opus and Kimi judgments"}
    opus = [o for o, k in cmp]
    kim = [k for o, k in cmp]
    exact = sum(1 for o, k in cmp if o == k)
    within1 = sum(1 for o, k in cmp if abs(o - k) <= 1)
    # quadratic weighted kappa on 0-4
    K = 5
    O = [[0] * K for _ in range(K)]
    for o, k in cmp:
        if 0 <= o < K and 0 <= k < K:
            O[o][k] += 1
    ntot = len(cmp)
    row = [sum(O[i]) for i in range(K)]
    col = [sum(O[i][j] for i in range(K)) for j in range(K)]
    W = [[((i - j) ** 2) / ((K - 1) ** 2) for j in range(K)] for i in range(K)]
    num = sum(W[i][j] * O[i][j] for i in range(K) for j in range(K))
    den = sum(W[i][j] * row[i] * col[j] / ntot for i in range(K) for j in range(K))
    qwk = round(1 - num / den, 3) if den else None
    return {"n": len(cmp), "opus_mean": mean(opus), "kimi_mean": mean(kim),
            "exact_agree_pct": round(100 * exact / len(cmp), 1),
            "within1_agree_pct": round(100 * within1 / len(cmp), 1),
            "quadratic_weighted_kappa": qwk,
            "cliffs_delta_opus_vs_kimi": cliffs_delta(opus, kim)}


def main():
    rows, missing = load_all_rows()
    log.info("loaded %d scored rows (%d batch files missing scores)", len(rows), len(missing))
    if missing:
        log.warning("missing score files: %s", missing[:20])
    (OUT / "opus_all.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    cross = summarize(rows)
    paired = paired_analysis(rows)
    verify = verify_consistency()
    calib = calibration(rows)

    # Length / observability check (Codex #11): is closure just a proxy for trace length?
    NATIVE = {"glm", "deepseek", "kimi", "qwen", "gemini"}
    RECON = {"claude46_ti", "claude47_ti", "angrygiraffe"}
    valid = [r for r in rows if isinstance(r.get("ccr_closure"), int) and r.get("reasoning_chars")]
    lens = [math.log10(max(r["reasoning_chars"], 1)) for r in valid]
    ccrs = [r["ccr_closure"] for r in valid]
    ml, mc2 = sum(lens) / len(lens), sum(ccrs) / len(ccrs)
    cov = sum((lens[i] - ml) * (ccrs[i] - mc2) for i in range(len(lens)))
    sl = math.sqrt(sum((x - ml) ** 2 for x in lens)); sc2 = math.sqrt(sum((x - mc2) ** 2 for x in ccrs))
    pear_len = round(cov / (sl * sc2), 3) if sl * sc2 else 0
    # within a comparable length band (1000-8000 chars), native vs reconstruction
    band = [r for r in valid if 1000 <= r["reasoning_chars"] <= 8000]
    nat_band = [r["ccr_closure"] for r in band if r["dataset"] in NATIVE]
    rec_band = [r["ccr_closure"] for r in band if r["dataset"] in RECON]
    length_report = {
        "pearson_log_len_vs_ccr": pear_len,
        "mean_chars_native": round(sum(r["reasoning_chars"] for r in valid if r["dataset"] in NATIVE) / max(sum(1 for r in valid if r["dataset"] in NATIVE), 1)),
        "mean_chars_reconstruction": round(sum(r["reasoning_chars"] for r in valid if r["dataset"] in RECON) / max(sum(1 for r in valid if r["dataset"] in RECON), 1)),
        "length_band_1k_8k": {
            "native_n": len(nat_band), "native_ccr": round(sum(nat_band) / max(len(nat_band), 1), 2),
            "recon_n": len(rec_band), "recon_ccr": round(sum(rec_band) / max(len(rec_band), 1), 2),
        },
    }
    (OUT / "report_length.json").write_text(json.dumps(length_report, indent=2))

    (OUT / "report_cross_dataset.json").write_text(json.dumps(cross, indent=2, ensure_ascii=False))
    (OUT / "report_paired.json").write_text(json.dumps(paired, indent=2, ensure_ascii=False))
    (OUT / "report_verify.json").write_text(json.dumps(verify, indent=2, ensure_ascii=False))
    (OUT / "report_calibration.json").write_text(json.dumps(calib, indent=2, ensure_ascii=False))

    print("\n=== CROSS-DATASET (Opus 4.8 judge) ===")
    print(f"{'dataset':14s} {'n':>3s} {'ccr':>5s} {'coupl':>5s} {'depth':>5s} {'dens':>5s} {'loop%':>5s} {'final%':>6s}")
    order = ["deepseek", "glm", "claude46_ti", "claude47_ti", "nohurry_opus", "gemini", "kimi", "qwen", "angrygiraffe", "roman_claude"]
    for ds in order:
        s = cross.get(ds)
        if not s:
            continue
        print(f"{ds:14s} {s['n']:>3d} {s['ccr_mean']:>5.2f} {s['coupling_mean']:>5.2f} "
              f"{s['depth_mean']:>5.2f} {s['density_mean']:>5.1f} {s['loop_pct']:>5.1f} {s['answer_change_final_pct']:>6.1f}")
    print(f"\n=== PAIRED (GLM vs DeepSeek, n={paired.get('n_pairs')}) ===")
    print(json.dumps(paired, indent=2))
    print(f"\n=== VERIFY (Opus self-consistency) ===\n{json.dumps(verify, indent=2)}")
    print(f"\n=== CALIBRATION (Opus vs Kimi) ===\n{json.dumps(calib, indent=2)}")
    print(f"\n=== LENGTH / OBSERVABILITY ===\n{json.dumps(length_report, indent=2)}")
    print(f"\nTOTAL scored rows: {len(rows)}")
    log.info("wrote report_*.json + opus_all.jsonl")


if __name__ == "__main__":
    main()

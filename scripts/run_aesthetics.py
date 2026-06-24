#!/usr/bin/env python3
"""Gemini advanced-metrics + first-principles completion:
 - aesthetics CE/NRR/BSD/ICI on full traces (advanced_metrics_exploration.md)
 - MI-collapse per dataset: I(critique_present ; answer_changed) (first_principles angle 1)
Cross-checks against the LLM CCR. Zero API.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from distill_audit import aesthetics  # noqa: E402
from distill_audit.runner_core import sample_traces  # noqa: E402
import run_pilot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aesthetics")
OUT = ROOT / "outputs"
DATASET_JOBS = [("claude46_ti", 40, None, None, 200), ("claude47_ti", 40, None, None, 200),
                ("nohurry_opus", 40, None, None, 200), ("gemini", 40, None, None, 200),
                ("kimi", 40, None, 60000, 200), ("qwen", 40, None, None, 200),
                ("angrygiraffe", 40, "full_train.jsonl", None, 200), ("roman_claude", 40, None, None, 60)]
ORDER = ["qwen", "glm", "kimi", "claude47_ti", "deepseek", "gemini", "claude46_ti",
         "nohurry_opus", "angrygiraffe", "roman_claude"]


def avg(vs):
    vs = [v for v in vs if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vs) / len(vs), 3) if vs else None


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float))
           and not (isinstance(x, float) and math.isnan(x))]
    if len(pts) < 4:
        return None
    xs2 = [p[0] for p in pts]; ys2 = [p[1] for p in pts]
    mx = sum(xs2) / len(xs2); my = sum(ys2) / len(ys2)
    cov = sum((xs2[i] - mx) * (ys2[i] - my) for i in range(len(xs2)))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs2)); sy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    return round(cov / (sx * sy), 3) if sx * sy else None


def mutual_info(xy: list[tuple[int, int]]) -> float | None:
    """I(X;Y) in bits for binary X,Y."""
    n = len(xy)
    if n < 8:
        return None
    from collections import Counter
    joint = Counter(xy); px = Counter(x for x, _ in xy); py = Counter(y for _, y in xy)
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        denom = (px[x] / n) * (py[y] / n)
        if pxy > 0 and denom > 0:
            mi += pxy * math.log2(pxy / denom)
    return round(mi, 4)


def main():
    # --- MI-collapse from LLM judgments ---
    by_ds_mi = defaultdict(list)
    ccr_by = {}
    for line in (OUT / "opus_all.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cp = r.get("critique_present")
        ac = r.get("answer_change")
        if cp in (0, 1) and ac in ("none", "intermediate", "final"):
            by_ds_mi[r["dataset"]].append((cp, 0 if ac == "none" else 1))
        if isinstance(r.get("ccr_closure"), int):
            ccr_by[(r["dataset"], r["uid"])] = r["ccr_closure"]

    # --- aesthetics on full traces ---
    full = []
    pairs = run_pilot.load_pairs(60)
    ds_ids = {d for d, _ in pairs}; glm_lk = {g for _, g in pairs} | {d for d, g in pairs if d != g}
    ds_tr = run_pilot.traces_by_uid("deepseek", ds_ids)
    glm_tr = run_pilot.traces_by_uid("glm", glm_lk, file_filter="main")
    for ds_id, glm_id in pairs:
        if ds_id in ds_tr:
            full.append(("deepseek", ds_id, ds_tr[ds_id].reasoning))
        gt = glm_tr.get(glm_id) or glm_tr.get(ds_id)
        if gt:
            full.append(("glm", gt.uid, gt.reasoning))
    for name, n, ff, scan, minc in DATASET_JOBS:
        for t in sample_traces(name, n, file_filter=ff, scan_limit=scan, min_chars=minc):
            full.append((name, t.uid, t.reasoning))
    log.info("computing aesthetics on %d full traces", len(full))

    rows = []
    with (OUT / "aesthetics_metrics.jsonl").open("w") as fh:
        for dataset, uid, reasoning in full:
            m = aesthetics.all_aesthetics(reasoning)
            m.update({"dataset": dataset, "uid": uid, "llm_ccr": ccr_by.get((dataset, uid))})
            rows.append(m)
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    print("\n=== GEMINI ADVANCED/AESTHETIC + MI-collapse per dataset ===")
    print(f"  {'dataset':14s} {'llm_ccr':>7s} {'CE':>6s} {'NRR':>6s} {'BSD':>6s} {'ICI':>6s} {'MI(crit;ans)':>13s}")
    summary = {}
    for ds in ORDER:
        if ds not in by_ds:
            continue
        rs = by_ds[ds]
        mi = mutual_info(by_ds_mi.get(ds, []))
        row = {"n": len(rs), "llm_ccr": avg([r["llm_ccr"] for r in rs]),
               "CE": avg([r["ce"] for r in rs]), "NRR": avg([r["nrr"] for r in rs]),
               "BSD": avg([r["bsd"] for r in rs]), "ICI": avg([r["ici"] for r in rs]),
               "MI_critique_answerchange": mi}
        summary[ds] = row
        print(f"  {ds:14s} {str(row['llm_ccr']):>7s} {str(row['CE']):>6s} {str(row['NRR']):>6s} "
              f"{str(row['BSD']):>6s} {str(row['ICI']):>6s} {str(mi):>13s}")

    print("\n=== convergent validity: Pearson(aesthetic, LLM CCR) trace-level ===")
    corr = {}
    for k in ["ce", "nrr", "bsd", "ici"]:
        c = pearson([r.get(k) for r in rows], [r.get("llm_ccr") for r in rows])
        corr[k] = c
        print(f"  {k:5s}: {c}")

    print("\n=== MI-collapse interpretation ===")
    print("  MI(critique_present ; answer_changed) — LOW MI = critique present but answer unchanged = ritual.")
    overall = mutual_info([p for v in by_ds_mi.values() for p in v])
    print(f"  pooled MI = {overall} bits")

    (OUT / "report_aesthetics.json").write_text(json.dumps(
        {"per_dataset": summary, "convergent_validity_pearson": corr,
         "pooled_MI_critique_answerchange": overall,
         "note_LSI": "Logical Symmetry Index = FUTURE WORK (needs reverse-problem generation), per advanced_metrics_exploration.md"},
        indent=2, ensure_ascii=False))
    log.info("wrote aesthetics_metrics.jsonl + report_aesthetics.json")


if __name__ == "__main__":
    main()

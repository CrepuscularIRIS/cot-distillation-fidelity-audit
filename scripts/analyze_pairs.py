#!/usr/bin/env python3
"""Round 1 analysis: paired comparison of GLM vs DeepSeek on matched questions.

Reads pilot_judged.jsonl, groups by pair, runs Wilcoxon signed-rank test on CCR,
computes Cliff's delta effect size, and reports the density-vs-closure paradox.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def cliffs_delta(x: list[float], y: list[float]) -> float:
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def wilcoxon_signed_rank(diffs: list[float]) -> dict:
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0]
    if len(nonzero) < 5:
        return {"n": len(nonzero), "note": "too few non-zero pairs for Wilcoxon"}
    nonzero.sort(key=lambda x: x[0])
    ranks = list(range(1, len(nonzero) + 1))
    w_plus = sum(r for r, (_, s) in zip(ranks, nonzero) if s > 0)
    w_minus = sum(r for r, (_, s) in zip(ranks, nonzero) if s < 0)
    n = len(nonzero)
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24) if n > 0 else 1
    z = (min(w_plus, w_minus) - mean_w) / std_w if std_w > 0 else 0
    return {"n": n, "W+": w_plus, "W-": w_minus, "z": round(z, 3),
            "direction": "GLM > DS" if w_plus > w_minus else "DS > GLM"}


def main() -> None:
    rows = list(iter_jsonl(OUT / "pilot_judged.jsonl"))
    log.info("loaded %d judged traces", len(rows))

    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        ds_key = r.get("pair_ds")
        if not ds_key:
            continue
        by_pair[ds_key][r["dataset"]] = r

    valid_pairs = [(p["deepseek"], p["glm"]) for p in by_pair.values()
                   if "deepseek" in p and "glm" in p
                   and isinstance(p["deepseek"].get("ccr_closure"), int)
                   and isinstance(p["glm"].get("ccr_closure"), int)]
    log.info("valid matched pairs: %d", len(valid_pairs))

    if not valid_pairs:
        log.warning("no valid pairs to analyze")
        return

    ds_ccr = [p[0]["ccr_closure"] for p in valid_pairs]
    glm_ccr = [p[1]["ccr_closure"] for p in valid_pairs]
    diffs = [g - d for d, g in zip(ds_ccr, glm_ccr)]

    ds_density = [p[0].get("crit_density_1k", 0) for p in valid_pairs]
    glm_density = [p[1].get("crit_density_1k", 0) for p in valid_pairs]

    ds_topo = Counter(p[0].get("reasoning_topology", "?") for p in valid_pairs)
    glm_topo = Counter(p[1].get("reasoning_topology", "?") for p in valid_pairs)

    ds_coupling = [p[0].get("monitoring_control_coupling", 0) for p in valid_pairs if isinstance(p[0].get("monitoring_control_coupling"), int)]
    glm_coupling = [p[1].get("monitoring_control_coupling", 0) for p in valid_pairs if isinstance(p[1].get("monitoring_control_coupling"), int)]

    wilcox = wilcoxon_signed_rank(diffs)
    delta = cliffs_delta(glm_ccr, ds_ccr)

    result = {
        "n_pairs": len(valid_pairs),
        "ccr": {
            "deepseek_mean": round(sum(ds_ccr) / len(ds_ccr), 2),
            "glm_mean": round(sum(glm_ccr) / len(glm_ccr), 2),
            "deepseek_median": sorted(ds_ccr)[len(ds_ccr) // 2],
            "glm_median": sorted(glm_ccr)[len(glm_ccr) // 2],
            "cliffs_delta": round(delta, 3),
            "delta_interpretation": "large" if abs(delta) > 0.474 else "medium" if abs(delta) > 0.33 else "small" if abs(delta) > 0.147 else "negligible",
            "wilcoxon": wilcox,
        },
        "density_1k": {
            "deepseek_mean": round(sum(ds_density) / max(len(ds_density), 1), 2),
            "glm_mean": round(sum(glm_density) / max(len(glm_density), 1), 2),
        },
        "coupling": {
            "deepseek_mean": round(sum(ds_coupling) / max(len(ds_coupling), 1), 2),
            "glm_mean": round(sum(glm_coupling) / max(len(glm_coupling), 1), 2),
        },
        "topology": {
            "deepseek": dict(ds_topo.most_common()),
            "glm": dict(glm_topo.most_common()),
        },
        "paradox_check": {
            "ds_higher_density_lower_closure": (
                sum(ds_density) / max(len(ds_density), 1) > sum(glm_density) / max(len(glm_density), 1)
                and sum(ds_ccr) / len(ds_ccr) < sum(glm_ccr) / len(glm_ccr)
            ),
        },
    }

    (OUT / "round1_paired_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("=" * 60)
    log.info("ROUND 1 PAIRED ANALYSIS (%d pairs)", len(valid_pairs))
    log.info("CCR: DS mean=%.2f | GLM mean=%.2f", result["ccr"]["deepseek_mean"], result["ccr"]["glm_mean"])
    log.info("Cliff's delta=%.3f (%s)", delta, result["ccr"]["delta_interpretation"])
    log.info("Wilcoxon: %s", wilcox)
    log.info("Density/1k: DS=%.1f | GLM=%.1f", result["density_1k"]["deepseek_mean"], result["density_1k"]["glm_mean"])
    log.info("Coupling: DS=%.2f | GLM=%.2f", result["coupling"]["deepseek_mean"], result["coupling"]["glm_mean"])
    log.info("Topology DS: %s", dict(ds_topo.most_common()))
    log.info("Topology GLM: %s", dict(glm_topo.most_common()))
    log.info("Paradox (DS higher density + lower closure): %s", result["paradox_check"]["ds_higher_density_lower_closure"])
    log.info("wrote outputs/round1_paired_analysis.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit of the 96-work training pool and of the original 5/10/30/96 subsets.

Uses training-split information only. Writes results/work_stats.csv and results/subset_audit.{csv,md}.
Ambiguity indicators come from the classical label builder's beat-level lattice (`beat_label`,
the coarsest lattice of that beat's own score attacks) next to the measure-level label:
  4x label but the beat's own attacks need <= 2x  ("4x written, 2x articulated")
  6x label but the beat's own attacks need <= 3x  ("6x written, 3x articulated")
  6x label where the beat itself mixes duplet and triplet positions (beat_label 6)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from neural_subdivision_hmm.data_scaling import common

HERE = Path(__file__).resolve().parent

INTERVALS = HERE.parents[1] / "subdivision_hmm/results/intervals.jsonl"
OUT = HERE / "results"


def beat_level_info(perf_ids):
    info = defaultdict(dict)
    for line in INTERVALS.open():
        if '"split": "train"' not in line:
            continue
        x = json.loads(line)
        if x["performance_id"] in perf_ids:
            info[x["performance_id"]][x["beat_index"]] = (x["beat_label"], x["quality_ok"],
                                                          x["perf_end"] - x["perf_start"])
    return info


def main():
    OUT.mkdir(exist_ok=True)
    splits, norm, pool, val = common.data()
    info = beat_level_info({q.performance_id for q in pool})
    order = common.original_nested_order(pool)
    rows = []
    for q in pool:
        s = common.work_stats(q)
        bi = info[q.performance_id]
        bl = np.array([bi.get(int(b), (0, None, np.nan))[0] for b in q.beat_index])
        dur = np.array([bi.get(int(b), (0, None, np.nan))[2] for b in q.beat_index], float)
        lab = q.labels
        s.update({
            "performance_id": q.performance_id,
            "alignment_quality_ok": bool(next(iter(bi.values()))[1]) if bi else None,
            "beat_seconds_median": float(np.nanmedian(dur)), "beat_seconds_cv": float(np.nanstd(dur) / np.nanmean(dur)),
            "tempo_bpm_median": float(60 / np.nanmedian(dur)),
            "n_4_written_2_articulated": int(((lab == 4) & np.isin(bl, (1, 2))).sum()),
            "n_6_written_3_articulated": int(((lab == 6) & np.isin(bl, (1, 3))).sum()),
            "n_6_beat_mixes_2_and_3": int(((lab == 6) & (bl == 6)).sum()),
            "original_rank": order.index(q.work_id),
        })
        rows.append(s)
    rows.sort(key=lambda r: r["original_rank"])
    with (OUT / "work_stats.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def agg(sub, name):
        n_ex = sum(r[f"n_{c}"] for r in sub for c in common.RATIOS)
        d = {"subset": name, "works": len(sub), "intervals": sum(r["intervals"] for r in sub),
             "composers": len({r["composer"] for r in sub}),
             **{f"prop_{c}": sum(r[f"n_{c}"] for r in sub) / n_ex for c in common.RATIOS},
             "exact_frac": n_ex / sum(r["intervals"] for r in sub),
             "family_only_frac": sum(r["n_family_only"] for r in sub) / sum(r["intervals"] for r in sub),
             "unknown_or_oov_frac": sum(r["n_unknown"] + r["n_oov"] for r in sub) / sum(r["intervals"] for r in sub),
             "hc_exact_frac": sum(r["hc_exact_frac"] * sum(r[f"n_{c}"] for c in common.RATIOS) for r in sub
                                  if np.isfinite(r["hc_exact_frac"])) / n_ex,
             "alignment_ok_works_frac": float(np.mean([bool(r["alignment_quality_ok"]) for r in sub])),
             "tempo_bpm_median_of_works": float(np.median([r["tempo_bpm_median"] for r in sub])),
             "tempo_bpm_iqr": float(np.subtract(*np.percentile([r["tempo_bpm_median"] for r in sub], [75, 25]))),
             "beat_cv_median": float(np.median([r["beat_seconds_cv"] for r in sub])),
             "intervals_per_work_median": float(np.median([r["intervals"] for r in sub])),
             "intervals_per_work_max_share": max(r["intervals"] for r in sub) / sum(r["intervals"] for r in sub),
             "4w2a_frac_of_4": sum(r["n_4_written_2_articulated"] for r in sub) / max(sum(r["n_4"] for r in sub), 1),
             "6w3a_frac_of_6": sum(r["n_6_written_3_articulated"] for r in sub) / max(sum(r["n_6"] for r in sub), 1),
             "6mixed_frac_of_6": sum(r["n_6_beat_mixes_2_and_3"] for r in sub) / max(sum(r["n_6"] for r in sub), 1),
             "feature_rms_mean": float(np.mean([r["feature_rms"] for r in sub]))}
        return d

    subsets = [agg(rows[:5], "original first 5 (fully exact)"), agg(rows[:10], "original 10"),
               agg(rows[:30], "original 30"), agg(rows[30:], "added 66 (ranks 31-96)"), agg(rows, "all 96")]
    with (OUT / "subset_audit.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(subsets[0])); w.writeheader(); w.writerows(subsets)
    comp = defaultdict(lambda: [0, 0])
    for r in rows:
        comp[r["composer"]][0 if r["original_rank"] < 30 else 1] += 1
    keys = ["works", "intervals", "composers", "prop_1", "prop_2", "prop_3", "prop_4", "prop_6", "exact_frac",
            "hc_exact_frac", "alignment_ok_works_frac", "tempo_bpm_median_of_works", "tempo_bpm_iqr", "beat_cv_median",
            "intervals_per_work_median", "intervals_per_work_max_share", "4w2a_frac_of_4", "6w3a_frac_of_6", "6mixed_frac_of_6"]
    md = ["| Quantity | " + " | ".join(s["subset"] for s in subsets) + " |", "|---|" + "---:|" * len(subsets)]
    for k in keys:
        md.append(f"| {k} | " + " | ".join(f"{s[k]:.3f}" if isinstance(s[k], float) else str(s[k]) for s in subsets) + " |")
    md += ["", "| Composer | works in original 30 | works in added 66 |", "|---|---:|---:|"]
    md += [f"| {c} | {a} | {b} |" for c, (a, b) in sorted(comp.items())]
    (OUT / "subset_audit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

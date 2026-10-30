#!/usr/bin/env python3
"""Dataset audit + reference-label extraction. Writes results/intervals.jsonl, results/split.csv,
results/audit_*.csv and results/audit.md."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from labels import performance_intervals, load_annotations  # noqa: E402

from config import INVENTORY as INV  # noqa: E402
OUT = HERE / "results"


def main():
    inv = list(csv.DictReader(INV.open()))
    ann = load_annotations()
    eligible = [r for r in inv if r["score_performance_aligned_flag"] == "True" and r["archive_audio_present"] == "True"]
    test_works = {r["work_id"] for r in inv if r["model_split"] == "baseline_val"}
    split = {}
    for w in sorted({r["work_id"] for r in eligible}):
        if w in test_works:
            split[w] = "test"
        else:
            split[w] = "val" if int(hashlib.sha1(w.encode()).hexdigest(), 16) % 5 == 0 else "train"
    rows = []
    for k, r in enumerate(eligible):
        q = (float(r["score_perf_note_match_pct"] or 0) >= 90 and float(r["score_perf_note_residual_p95_ms"] or 1e9) <= 60)
        for x in performance_intervals(r["performance_id"], ann[r["performance_id"]], q):
            x.update(work_id=r["work_id"], split=split[r["work_id"]], composer=r["work_id"].split("::")[0],
                     feature_stem=r["feature_stem"], explore_this_split=r["model_split"],
                     xml_tuplet_ratios=r["xml_tuplet_ratios"])
            rows.append(x)
        if (k + 1) % 100 == 0:
            print(f"labelled {k + 1}/{len(eligible)}", flush=True)
    OUT.mkdir(exist_ok=True)
    with (OUT / "intervals.jsonl").open("w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    with (OUT / "split.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["work_id", "split"]); w.writerows(sorted(split.items()))

    # coverage tables
    cov = []
    for s in ("train", "val", "test"):
        rs = [x for x in rows if x["split"] == s]
        works = {x["work_id"] for x in rs}
        perfs = {x["performance_id"] for x in rs}
        c, ch = Counter(x["label"] for x in rs), Counter(x["label"] for x in rs if x["high_confidence"])
        for lab in sorted(set(c)):
            cov.append({"split": s, "label": lab if lab else "unknown", "intervals": c[lab], "high_conf_intervals": ch[lab],
                        "works_with_label": len({x["work_id"] for x in rs if x["label"] == lab}),
                        "split_works": len(works), "split_performances": len(perfs)})
    with (OUT / "audit_coverage.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cov[0])); w.writeheader(); w.writerows(cov)
    unk = [x for x in rows if x["label"] == 0]
    agree = Counter((x["label"], x["beat_label"]) for x in rows if x["label"])
    lines = ["# Dataset audit", "",
             f"* ASAP performances: {len(inv)}; aligned with exact MAESTRO audio: {len(eligible)}; works: {len(split)}.",
             f"* Split by work: train {sum(v=='train' for v in split.values())}, val {sum(v=='val' for v in split.values())}, "
             f"test {sum(v=='test' for v in split.values())} works (test = works of Explore This baseline_val ASAP).",
             f"* Labelled beat intervals: {len(rows)}; unknown (off-lattice attacks or <2 beats): {len(unk)} "
             f"({100*len(unk)/len(rows):.1f}%); of off-lattice attacks in unknown measures, "
             f"{sum(x['near_beat_offlattice_attacks'] for x in unk)}/{sum(x['offlattice_attacks'] for x in unk)} lie within 0.1 beat of a beat (grace/arpeggio-like).",
             f"* Performances with any xml tuplet: {len({x['performance_id'] for x in rows if x['xml_tuplet_ratios'] not in ('', '{}')})}.",
             f"* Beat-level vs measure-level label agreement (beat label = lattice of that beat's own attacks): "
             f"{100*sum(v for (a, b), v in agree.items() if a == b)/sum(agree.values()):.1f}% of labelled intervals.",
             "", "| split | label | intervals | high-confidence | works with label |", "|---|---|---:|---:|---:|"]
    lines += [f"| {r['split']} | {r['label']} | {r['intervals']} | {r['high_conf_intervals']} | {r['works_with_label']} |" for r in cov]
    (OUT / "audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

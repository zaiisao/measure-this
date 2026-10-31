#!/usr/bin/env python3
"""Reproduce the training-regime supervision and sample-composition audit."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parent
RESULTS = HERE / "results"
sys.path.insert(0, str(PROJECT))

from data import K, RATIOS, allowed_states, load_sequences, sequence_summary  # noqa: E402
from train import choose_sequences  # noqa: E402


def constraint_mask(label: int, mode: str) -> np.ndarray:
    if mode == "unlabeled":
        return np.ones(K, dtype=bool)
    return allowed_states(int(label), mode)


def label_bucket(label: int) -> str:
    if label in RATIOS:
        return "exact_ratio"
    if label in (8, 12):
        return "family_only_oov"
    if label == 0:
        return "unknown_zero"
    return "other_oov"


def finite_summary(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sd": float(x.std()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, .10)),
        "p90": float(np.quantile(x, .90)),
    }


def summarize_regime(name, seqs, mode, norm):
    labels = np.concatenate([q.labels for q in seqs]) if seqs else np.zeros(0, dtype=int)
    observed = np.concatenate([q.observed for q in seqs]) if seqs else np.zeros(0, dtype=bool)
    high = np.concatenate([q.high_confidence for q in seqs]) if seqs else np.zeros(0, dtype=bool)
    masks = np.stack([constraint_mask(int(y), mode) for y in labels]) if len(labels) else np.zeros((0, K), bool)
    allowed_n = masks.sum(1) if len(labels) else np.zeros(0, dtype=int)
    categories = Counter(label_bucket(int(y)) for y in labels)
    exact = np.isin(labels, RATIOS)
    ratio_counts = {str(r): int(np.sum(labels == r)) for r in RATIOS}
    exact_conf = {
        str(r): {
            "n": int(np.sum(labels == r)),
            "high_confidence": int(np.sum((labels == r) & high)),
            "low_confidence": int(np.sum((labels == r) & ~high)),
        }
        for r in RATIOS
    }
    label_conf = {
        str(int(label)): {
            "n": int(np.sum(labels == label)),
            "high_confidence": int(np.sum((labels == label) & high)),
            "low_confidence": int(np.sum((labels == label) & ~high)),
        }
        for label in sorted(set(labels.tolist()))
    }
    # One observation-level acoustic summary per included interval. The feature vector is
    # already the frozen per-performance standardized mel50 representation.
    feats = np.concatenate([q.y for q in seqs]) if seqs else np.zeros((0, 48))
    dur_z = np.concatenate([q.log_duration for q in seqs]) if seqs else np.zeros(0)
    log_duration = dur_z * norm["log_duration_std"] + norm["log_duration_mean"]
    seconds = np.exp(log_duration)
    rms = np.sqrt(np.mean(feats ** 2, axis=1)) if len(feats) else np.zeros(0)
    mean_abs = np.mean(np.abs(feats), axis=1) if len(feats) else np.zeros(0)
    exact_rms = rms[exact & observed]
    exact_abs = mean_abs[exact & observed]
    all_rms = rms[observed]
    all_abs = mean_abs[observed]
    constrained = allowed_n < K
    return {
        "name": name,
        "constraint_mode": mode,
        "sequence_summary": sequence_summary(seqs),
        "work_count": int(len({q.work_id for q in seqs})),
        "composer_count": int(len({q.work_id.split("::", 1)[0] for q in seqs})),
        "composers": sorted({q.work_id.split("::", 1)[0] for q in seqs}),
        "label_buckets": {k: int(categories.get(k, 0)) for k in
                          ("exact_ratio", "family_only_oov", "unknown_zero", "other_oov")},
        "exact_ratio_counts": ratio_counts,
        "exact_confidence_by_ratio": exact_conf,
        "label_confidence_by_value": label_conf,
        "exact_ratio_high_confidence_n": int(np.sum(exact & high)),
        "exact_ratio_low_confidence_n": int(np.sum(exact & ~high)),
        "constraint_rows": int(constrained.sum()),
        "constraint_fraction_of_intervals": float(constrained.mean()) if len(labels) else 0.0,
        "singleton_constraint_rows": int(np.sum(allowed_n == 1)),
        "family_constraint_rows_two_states": int(np.sum(allowed_n == 2)),
        "unconstrained_rows": int(np.sum(allowed_n == K)),
        "observed_acoustic_rows": int(observed.sum()),
        "missing_acoustic_rows": int((~observed).sum()),
        "duration_seconds_all": finite_summary(seconds[observed]),
        "duration_seconds_exact_labels": finite_summary(seconds[exact & observed]),
        "envelope_rms_all": finite_summary(all_rms),
        "envelope_mean_abs_all": finite_summary(all_abs),
        "envelope_rms_exact_labels": finite_summary(exact_rms),
        "envelope_mean_abs_exact_labels": finite_summary(exact_abs),
        "log_duration_z_all": finite_summary(dur_z[observed]),
    }


def per_work_rows(regimes, norm):
    rows = []
    for name, seqs, mode in regimes:
        grouped = defaultdict(list)
        for q in seqs:
            grouped[q.work_id].append(q)
        for work, qs in sorted(grouped.items()):
            labels = np.concatenate([q.labels for q in qs])
            high = np.concatenate([q.high_confidence for q in qs])
            obs = np.concatenate([q.observed for q in qs])
            masks = np.stack([constraint_mask(int(y), mode) for y in labels])
            allowed_n = masks.sum(1)
            exact = np.isin(labels, RATIOS)
            feats = np.concatenate([q.y for q in qs])
            d_z = np.concatenate([q.log_duration for q in qs])
            seconds = np.exp(d_z * norm["log_duration_std"] + norm["log_duration_mean"])
            rms = np.sqrt(np.mean(feats ** 2, axis=1))
            mean_abs = np.mean(np.abs(feats), axis=1)
            composer = work.split("::", 1)[0]
            row = {
                "regime": name,
                "constraint_mode": mode,
                "work_id": work,
                "composer": composer,
                "sequences": len(qs),
                "intervals": len(labels),
                "observed_audio_intervals": int(obs.sum()),
                "constrained_intervals": int(np.sum(allowed_n < K)),
                "singleton_constraints": int(np.sum(allowed_n == 1)),
                "two_state_family_constraints": int(np.sum(allowed_n == 2)),
                "exact_ratio_intervals": int(exact.sum()),
                "exact_ratio_high_confidence": int(np.sum(exact & high)),
                "exact_ratio_low_confidence": int(np.sum(exact & ~high)),
                "high_confidence_all": int(high.sum()),
                "duration_seconds_mean": float(seconds[obs].mean()),
                "duration_seconds_sd": float(seconds[obs].std()),
                "envelope_rms_mean": float(rms[obs].mean()),
                "envelope_mean_abs_mean": float(mean_abs[obs].mean()),
            }
            for r in RATIOS:
                row[f"label_{r}"] = int(np.sum(labels == r))
            for label in (0, 8, 12, 16, 24):
                row[f"label_{label}"] = int(np.sum(labels == label))
            rows.append(row)
    return rows


def composer_rows(per_work):
    df = pd.DataFrame(per_work)
    if df.empty:
        return []
    mean_cols = ("duration_seconds_mean", "envelope_rms_mean", "envelope_mean_abs_mean")
    numeric = [c for c in df.columns if c not in {"regime", "constraint_mode", "work_id", "composer"}
               and c not in mean_cols and c != "duration_seconds_sd"]
    out = df.groupby(["regime", "constraint_mode", "composer"], as_index=False)[numeric].sum()
    # Duration/acoustic measures are means, not additive counts. Compute interval-weighted means.
    for col in mean_cols:
        weighted = (df.assign(_v=df[col] * df["observed_audio_intervals"])
                    .groupby(["regime", "constraint_mode", "composer"], as_index=False)["_v"].sum())
        counts = (df.groupby(["regime", "constraint_mode", "composer"], as_index=False)["observed_audio_intervals"].sum())
        out = out.drop(columns=[col], errors="ignore").merge(weighted, on=["regime", "constraint_mode", "composer"])
        out = out.merge(counts, on=["regime", "constraint_mode", "composer"], suffixes=("", "_count"))
        out[col] = out.pop("_v") / out["observed_audio_intervals_count"].clip(lower=1)
        out = out.drop(columns=["observed_audio_intervals_count"])
    # Pool within-work and between-work duration variance (per-work SD uses ddof=0).
    out["duration_seconds_sd"] = 0.0
    for key, group in df.groupby(["regime", "constraint_mode", "composer"]):
        n = group["observed_audio_intervals"].to_numpy(dtype=float)
        means = group["duration_seconds_mean"].to_numpy(dtype=float)
        sds = group["duration_seconds_sd"].to_numpy(dtype=float)
        pooled_mean = float(np.sum(n * means) / max(np.sum(n), 1))
        pooled_var = float(np.sum(n * (sds ** 2 + (means - pooled_mean) ** 2)) / max(np.sum(n), 1))
        mask = ((out["regime"] == key[0]) & (out["constraint_mode"] == key[1]) &
                (out["composer"] == key[2]))
        out.loc[mask, "duration_seconds_sd"] = np.sqrt(max(pooled_var, 0.0))
    return out.to_dict(orient="records")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    splits, norm = load_sequences()
    all_train = splits["train"]
    definitions = []
    for mode in ("exact", "supervised_only", "unsupervised", "mixed", "partial_family"):
        seqs, c_mode = choose_sequences(all_train, mode)
        definitions.append((mode, seqs, c_mode))
    regimes = [(name, seqs, c_mode) for name, seqs, c_mode in definitions]
    summaries = [summarize_regime(name, seqs, c_mode, norm) for name, seqs, c_mode in regimes]
    per_work = per_work_rows(regimes, norm)
    payload = {
        "source": "frozen neural_subdivision_hmm.data.load_sequences() and train.choose_sequences()",
        "split": "original training works only; no validation/test labels used",
        "ratios": list(RATIOS),
        "normalization": norm,
        "all_train_sequences_before_mode_filter": sequence_summary(all_train),
        "regimes": summaries,
        "strict_vs_broad_comparison": {
            "strict_exact_works": sorted({q.work_id for q in definitions[0][1]}),
            "strict_exact_composers": sorted({q.work_id.split("::", 1)[0] for q in definitions[0][1]}),
            "broad_supervised_works": len({q.work_id for q in definitions[1][1]}),
            "broad_supervised_composers": sorted({q.work_id.split("::", 1)[0] for q in definitions[1][1]}),
            "mixed_added_sequences": len(definitions[3][1]) - len(definitions[1][1]),
            "mixed_added_intervals": sum(q.n for q in definitions[3][1]) - sum(q.n for q in definitions[1][1]),
        },
    }
    (RESULTS / "supervision_audit.json").write_text(json.dumps(
        payload, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    pd.DataFrame(per_work).to_csv(RESULTS / "per_work_supervision.csv", index=False)
    pd.DataFrame(composer_rows(per_work)).to_csv(RESULTS / "per_composer_supervision.csv", index=False)
    exact_conf = []
    all_conf = []
    for regime in summaries:
        for ratio, stats in regime["exact_confidence_by_ratio"].items():
            exact_conf.append({"regime": regime["name"], "ratio": int(ratio), **stats})
        for label, stats in regime["label_confidence_by_value"].items():
            all_conf.append({"regime": regime["name"], "label": int(label), **stats})
    pd.DataFrame(exact_conf).to_csv(RESULTS / "exact_label_confidence.csv", index=False)
    pd.DataFrame(all_conf).to_csv(RESULTS / "label_confidence_all.csv", index=False)

    strict, broad, mixed, partial = (summaries[i] for i in (0, 1, 3, 4))
    note = f"""# Training supervision audit

This audit rebuilds the original train split through the frozen data adapter and reproduces each regime's filtering in `train.choose_sequences()`. No model was retrained and no held-out labels were used.

## Why the sequence counts differ

`supervised_exact` first drops every sequence without mel50 and then requires every interval label to be one of `{', '.join(map(str, RATIOS))}`. Only {strict['sequence_summary']['sequences']} audio-bearing sequences ({strict['sequence_summary']['intervals']:,} intervals) meet both tests. `supervised_only` instead keeps any audio-bearing sequence with at least one in-vocabulary label, yielding {broad['sequence_summary']['sequences']} sequences ({broad['sequence_summary']['intervals']:,} intervals). This broad set contains the same five fully exact sequences plus works with partial exact labels.

The strict subset covers {', '.join(strict['composers'])} only, versus {broad['composer_count']} composers in the broad regime. The strict labels are class-skewed: {strict['exact_ratio_counts']}. The broad set has {broad['label_buckets']['exact_ratio']:,} exact-ratio labels; only {broad['exact_ratio_high_confidence_n']:,} are marked high-confidence, compared with {strict['exact_ratio_high_confidence_n']:,}/{strict['label_buckets']['exact_ratio']:,} in the strict subset. The high-confidence flag is not consulted by `choose_sequences()` or `allowed_states()`, so lower-confidence exact-ratio labels become hard singleton constraints in the broad supervised loss.

The label builder defines `high_confidence` as both good score/performance alignment and repeated diagnostic attacks for the measure label (for label 1, repeated beat attacks). Thus this is a measurable annotation-quality confound, not a claim that all other labels are wrong.

## Constraints and observations

In `supervised_only`, exact ratios contribute {broad['constraint_rows']:,}/{broad['sequence_summary']['intervals']:,} informative singleton constraints ({100*broad['constraint_fraction_of_intervals']:.1f}%). The remaining {broad['unconstrained_rows']:,} rows have labels 0, 8, 12, 16, or 24 and are unconstrained under exact mode. Labels 8 and 12 can carry family information, but this regime does not use it. In `partial_family`, labels 8/12 are mapped to `{2,4}` / `{3,6}`, increasing constrained rows to {partial['constraint_rows']:,}/{partial['sequence_summary']['intervals']:,}; labels 0/16/24 remain unrestricted. `mixed` adds {payload['strict_vs_broad_comparison']['mixed_added_sequences']} audio-bearing sequences ({payload['strict_vs_broad_comparison']['mixed_added_intervals']} intervals), none of which has an exact-ratio label.

Audio-absent sequences are dropped because they provide no acoustic gradient; no selected sequence contains missing feature rows. Across the strict versus broad observations, mel50 envelope RMS is {strict['envelope_rms_all']['mean']:.3f} vs {broad['envelope_rms_all']['mean']:.3f}, mean absolute envelope {strict['envelope_mean_abs_all']['mean']:.3f} vs {broad['envelope_mean_abs_all']['mean']:.3f}. The strict subset has narrower raw beat-duration spread ({strict['duration_seconds_all']['sd']:.3f}s vs {broad['duration_seconds_all']['sd']:.3f}s). This suggests a tempo-range and annotation-composition difference, while these global mel50 scale summaries are close.

## Artifacts

- `supervision_audit.json`: per-regime counts, constraints, class counts, confidence, acoustic summaries, and composer/work coverage.
- `per_work_supervision.csv`: one row per regime and work, including label and confidence counts.
- `per_composer_supervision.csv`: aggregated work coverage and interval counts by composer.
- `exact_label_confidence.csv`: high-/low-confidence counts by ratio and regime.
- `label_confidence_all.csv`: high-/low-confidence counts for every reference label value and regime.
"""
    (RESULTS / "README.md").write_text(note)
    print(RESULTS)
    for s in summaries:
        print(s["name"], s["sequence_summary"]["sequences"], s["sequence_summary"]["intervals"],
              s["constraint_rows"], s["exact_ratio_high_confidence_n"], s["exact_ratio_low_confidence_n"])


if __name__ == "__main__":
    main()

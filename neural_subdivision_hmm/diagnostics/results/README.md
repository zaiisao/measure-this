# Training supervision audit

This audit rebuilds the original train split through the frozen data adapter and reproduces each regime's filtering in `train.choose_sequences()`. No model was retrained and no held-out labels were used.

## Why the sequence counts differ

`supervised_exact` first drops every sequence without mel50 and then requires every interval label to be one of `1, 2, 3, 4, 6`. Only 5 audio-bearing sequences (1,293 intervals) meet both tests. `supervised_only` instead keeps any audio-bearing sequence with at least one in-vocabulary label, yielding 96 sequences (41,792 intervals). This broad set contains the same five fully exact sequences plus works with partial exact labels.

The strict subset covers Bach, Schubert only, versus 11 composers in the broad regime. The strict labels are class-skewed: {'1': 90, '2': 174, '3': 192, '4': 756, '6': 81}. The broad set has 35,316 exact-ratio labels; only 17,150 are marked high-confidence, compared with 1,279/1,293 in the strict subset. The high-confidence flag is not consulted by `choose_sequences()` or `allowed_states()`, so lower-confidence exact-ratio labels become hard singleton constraints in the broad supervised loss.

The label builder defines `high_confidence` as both good score/performance alignment and repeated diagnostic attacks for the measure label (for label 1, repeated beat attacks). Thus this is a measurable annotation-quality confound, not a claim that all other labels are wrong.

## Constraints and observations

In `supervised_only`, exact ratios contribute 35,316/41,792 informative singleton constraints (84.5%). The remaining 6,476 rows have labels 0, 8, 12, 16, or 24 and are unconstrained under exact mode. Labels 8 and 12 can carry family information, but this regime does not use it. In `partial_family`, labels 8/12 are mapped to `(2, 4)` / `(3, 6)`, increasing constrained rows to 39,103/42,002; labels 0/16/24 remain unrestricted. `mixed` adds 2 audio-bearing sequences (210 intervals), none of which has an exact-ratio label.

Audio-absent sequences are dropped because they provide no acoustic gradient; no selected sequence contains missing feature rows. Across the strict versus broad observations, mel50 envelope RMS is 2.118 vs 2.164, mean absolute envelope 1.245 vs 1.254. The strict subset has narrower raw beat-duration spread (0.197s vs 0.405s). This suggests a tempo-range and annotation-composition difference, while these global mel50 scale summaries are close.

## Artifacts

- `supervision_audit.json`: per-regime counts, constraints, class counts, confidence, acoustic summaries, and composer/work coverage.
- `per_work_supervision.csv`: one row per regime and work, including label and confidence counts.
- `per_composer_supervision.csv`: aggregated work coverage and interval counts by composer.
- `exact_label_confidence.csv`: high-/low-confidence counts by ratio and regime.
- `label_confidence_all.csv`: high-/low-confidence counts for every reference label value and regime.

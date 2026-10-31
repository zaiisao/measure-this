# NHMM training-data scaling and composition study

Isolated in `neural_subdivision_hmm/data_scaling/`. The NHMM architecture, five states, fixed lattices, normalized Gaussian emission, exact HMM inference, λ = 10 objective, mel50 input, split, evaluator and every earlier checkpoint/report are unchanged and reused. 141 training runs, 13.9 CPU-hours (2 threads per run).

## Answer

**The 30 → 96 "decline" is an optimization-schedule artifact, not evidence that more works hurt.**

The original protocol (≤10 epochs, early stopping, checkpoint by validation objective) gives each subset size a different number of optimizer updates:
- 30 works got about 80 updates.
- 96 works got about 240.

Validation balanced accuracy is strongly non-monotone in updates ([learning curves](figures/learning_curves_B2.png)):
- About 0.75 after roughly 40 updates.
- A dip to about 0.6 at 200–300 updates, while the acoustic density is fitted.
- Recovery to about 0.70–0.74 by 500–1,000 updates.

The 96-work model was therefore scored at the bottom of the dip, and the 30-work model before it. At a matched budget of 1,200 updates, 96 works scores the same on test as the best 30-work set and better on validation than random 30-work sets.

| Test BAcc (decoder chosen on validation; checkpoint = original policy) | ≤10 epochs (original) | 1,200 updates (matched) |
|---|---:|---:|
| Random 30-work subsets (5 draws, 9 runs) | .734 ± .035 | .759 ± .043 |
| Original 30-work subset (3 seeds) | .728 ± .012 | .769 ± .017 |
| All 96 works (3 seeds) | **.634 ± .036** | **.768 ± .022** |
| *Validation BAcc, same rows* | .686 / .711 / .651 | .693 / .734 / .714 |

The reproduction from scratch of the original pair gave .74 vs .64. Saved values were .76 vs .65. Validation curves agree to within 0.5 nats; CPU floating-point nondeterminism alone shifts test BAcc by about 0.01–0.02.

## 1. Audit of the original experiment

- **Selection:** the original 5/10/30/96 sets are strictly nested. They start with the 5 fully exact sequences, then add the other training sequences in a fixed permutation (`rng=20261031`). Selection used no validation or test information. The train pool has one sequence per work: 96 works with audio and in-vocabulary labels out of 116 training works.
- **Composition** ([subset_audit.md](results/subset_audit.md), per-work [work_stats.csv](results/work_stats.csv)). Compared with the 66 added works, the original 30 are:
  - more ternary: 3× + 6× = 37 % of exact labels vs 26 %;
  - lower in 4×: 24 % vs 34 %;
  - faster: median 114 vs 90 BPM;
  - a little longer per work: median 380 vs 284 intervals.

  They are **not** cleaner: 46 % vs 50 % high-confidence labels, and 47 % vs 53 % of works with good score/performance alignment.
- **Compute:** at ≤10 epochs × 4 sequences per batch, update counts scale with the number of works. All saved 30- and 96-work runs stopped at the epoch cap with the validation objective still decreasing, so early stopping never triggered.

## 2. Experiment A — randomized subsets (Stage 1)

Design:
- **Independent:** 5 uniform draws per size.
- **Nested:** 3 random orders, plus the original order.
- **Sizes:** n ∈ {10, 20, 30, 50, 70, 96}.
- **Seeds:** extra initialization seeds for two random 30-work draws, the original 30, and 96.

Exact work IDs and seeds are in [stage1_subsets.json](results/stage1_subsets.json) and in each `runs/*/training.json`. Full table: [stage1_scaling.csv](results/stage1_scaling.csv). Figure: [scaling_val_objective.png](figures/scaling_val_objective.png).

- **The decline reproduces under the original ≤10-epoch protocol for every selection protocol.** Independent subsets fall from .73 at n = 20–30 to .67 at n = 70 and .63 at n = 96, and nested orders behave the same. It is therefore not a single-subset effect, but it is a protocol effect (§3).
- **Under matched updates the curve rises, then plateaus.** Test BAcc for independent subsets:
  - n = 10: .63
  - n = 20: .72
  - n = 30: .76
  - n = 50: .77
  - n = 70: .74
  - n = 96: .77

  Validation BAcc follows the same pattern: .58 → .66 → .69 → .70 → .70 → .71.
- **Subset choice vs initialization at n = 30.** Between-subset SD is .044–.058 on validation and .029–.052 on test. Between-seed SD on the same subset is .010–.017 on validation and .016–.025 on test ([variance table](results/stage1_variance_n30.csv)). Which 30 works are drawn matters as much as or more than initialization. Three seeds on one subset therefore cannot establish a 30-work effect.

## 3. Experiment B — matched epochs vs matched updates

- **B1 (original ≤10 epochs) vs B2 (1,200 updates, constant LR, 4 sequences per batch, recorded epoch-wise permutation schedule, evaluation every 40 updates):** B2 removes the 30 vs 96 discrepancy.
- **The U-shaped validation curve is common to all sizes.** Small sets then overfit: acoustic and state NLL rise after about 500–800 updates, and n = 10 diverges. The original checkpoint rule (best validation objective) protects the larger sets but tracks validation BAcc only loosely.
- **Checkpoint policy matters as much as data size.** Choosing the checkpoint by validation BAcc selects very early checkpoints, at about 35–270 updates:

  | Model | Val BAcc | Test BAcc |
  |---|---:|---:|
  | 96 works, B1 | .766 | .785 |
  | 96 works, B2 | .761 | .810 |
  | Original 30, B2 | .760 | .776 |
  | Random 30, B2 | .750 | .785 |

  This selection is optimistic on validation, because it is chosen there. But it shows the best subdivision decoding occurs early in training, before the long acoustic-fitting phase, and that 96 ≈ 30 again under that policy.

  The acoustic NLL objective and subdivision accuracy only partly agree. This is the same tension the earlier diagnostics found, now visible over time.
- **Not done:** no learning-rate or batch-size changes. They were not needed to remove the discrepancy, and an uncontrolled search was out of scope.

## 4. Experiments C, D, E — composition, weighting, reliability (Stage 2, 1,200 updates)

[stage2_summary.csv](results/stage2_summary.csv). Δ is validation BAcc vs 96-work B2 with the same initialization seed.

| Variant | Runs | Val BAcc | Test BAcc | Δ val per seed | Reading |
|---|---:|---:|---:|---|---|
| All 96 (reference) | 3 | .714 | .768 | — | |
| C: class-balanced 30 (training label counts only) | 5 draws | .691 | .740 | — | = random 30 (.693) |
| C: composer-stratified 30 | 5 | .691 | .716 | — | = random 30 |
| C: tempo-stratified 30 | 5 | .691 | .747 | — | = random 30 |
| D2: work-weighted objective | 3 | .664 | .665 | −.03, −.09, −.03 | worse; 1× recall collapses to .36 |
| D3 control: per-beat marginal posterior state term, unweighted | 2 (+1 diverged) | .548 | .611 | −.17, −.17 | the marginal objective itself fails |
| D3: class-balanced marginal state term | 3 | .426 | .443 | −.30, −.21, −.35 | worse still |
| E1: confidence-weighted marginal term (low-conf 0.3) | 2 (+1 diverged) | .519 | .590 | −.24, −.16 | not separable from the D3 failure |
| E2: drop 19 suspected misaligned works (note match <80 % or p95 >100 ms) | 3 | .679 | .733 | −.04, −.04, −.03 | worse: removing data hurts |
| F2: original-30 checkpoint + 600 updates on 96 | 3 | .735 | .734 | −.005, +.05, +.02 | inconclusive |

- **Composition (C).** Stratifying 30-work draws by class balance, composer or tempo gives the same validation BAcc as uniform draws. The original 30's validation advantage (.734) is not reproduced by any composition rule tried.
- **Composition correlation.** Across 24 independently drawn 30-work subsets, no composition statistic has a reliable rank correlation with validation BAcc ([composition_vs_performance_n30.csv](results/composition_vs_performance_n30.csv)). Statistics tested: ternary share, 4× share, 6× share, high-confidence share, tempo, composers, intervals. Tempo is the largest at ρ = .34 [−.01, .62]. These analyses do not support a composition explanation; with n = 24 they cannot exclude a modest one.
- **Work weighting (D2)** changes the objective (it is no longer the original weighted likelihood) and lowers validation BAcc on all three seeds. Long works are not harmful influences under the original weighting.
- **Class-balanced and confidence-weighted supervision (D3, E1)** had to use per-beat posterior marginals, because the original sequence-level term −log p(C | O, B) cannot be weighted per label. The *unweighted* marginal control already fails (val .55), and 2 of 9 such runs diverged with non-positive-definite covariances. The weighting questions are therefore **unresolved**: the substitute objective, not the weights, dominates the result.
- **Annotation reliability (E2).** Excluding the 19 works with the poorest score/performance alignment, a training-only rule fixed before running, reduces validation BAcc on all three seeds. There is no evidence that the added works' weaker alignment drives the original decline. The difficult-work ranking and down-weighting interventions were not run, because the decline was already explained.
- **Incremental addition (F).** The scratch growth curve is Stage 1's original-order B2 series (§2). Continuing from the 30-work checkpoint (F2) is not reliably better than scratch-96 on validation and is worse on test. There is no sign of catastrophic change when works are added.

## 5. Answers

1. **Does the decline reproduce across independent subsets?** Under the original ≤10-epoch protocol, yes, for random, nested and original selections. Under matched updates, no: performance rises to about 30–50 works, then plateaus.
2. **How much is subset luck?** At matched updates the original 30 scores about 0.04 above random 30-work draws on validation, with ±.03–.06 SD across draws. On test it equals 96 works. The headline .737 vs .650 came mostly from the schedule, and only partly from a favorable subset.
3. **Do matched updates and controlled checkpointing remove it?** Yes. Checkpoint policy is a further major lever.
4. **Do reliability, class balance, composer, tempo or ambiguity explain it?** Not supported. Composition-controlled sampling, composition correlations and removing misaligned works all fail to explain it. Ambiguity indicators differ between the original 30 and the added 66, and their effect was not tested directly: 4× written / 2× articulated is 12.5 % vs 9.5 % of 4× labels, and 6× written / 3× articulated is 20 % vs 12 % of 6× labels.
5. **Can a 96-work NHMM match the 30-work model without architecture changes?** Yes:
   - matched updates: test .768 vs .769, val .714 vs .734;
   - validation-BAcc checkpointing: test .81 vs .78.
6. **What genuinely improves generalization?** Protocol decisions were made on validation: matched or sufficient update budgets, and checkpoint selection aligned with subdivision accuracy. Reweighting, filtering and stratification did not help. The 18 test works have been examined many times, and the work bootstraps from the earlier study were already wide. Selecting a final protocol therefore needs work-level inner cross-validation over the training works, not run here, followed by an untouched work-disjoint test set, which does not exist yet.
7. **Is the NHMM ready for larger partially annotated data?** Cautiously. More works do not hurt once optimization is matched, but returns are flat beyond about 50 works.

   The larger problem is that subdivision accuracy peaks early and is not what the acoustic likelihood optimizes. Before scaling to large partially labeled beat-tracking corpora, fix two things:
   - a principled stopping/selection criterion, ideally inner cross-validation on work folds;
   - the acoustic-vs-state trade-off.

   Otherwise added data mainly lengthens the acoustic-fitting phase.

## Coverage and limits

**Done:**
- the full Stage 1;
- Stage 2 C (three samplers), D2, D3 (+ control), E1, E2, F2;
- F1 via the nested original-order runs (10/20/30/50/70/96).

**Not done:**
- work-level inner cross-validation;
- learning-rate and batch-size controls;
- the difficult-work ranking and down-weighting interventions;
- confidence-aware ambiguous-state constraints;
- per-class learning-curve figures (the histories contain the recalls).

Failed runs are recorded in `logs/stage2.log`: D3 control seed 2 and E1 seed 2 hit a Cholesky failure. Test per-work BAcc, confusions, NLL, ECE, Brier, transitions and predicted distributions are in each `runs/*/eval_*.json` and in [all_runs.csv](results/all_runs.csv).

## Reproduction

```sh
cd neural_subdivision_hmm/data_scaling
PY=python   # numpy, scipy, torch, pandas, matplotlib
$PY -m unittest discover -s tests          # 10 tests
$PY audit.py
$PY runner.py jobs/repro.jsonl --workers 2
$PY runner.py jobs/stage1.jsonl --workers 30
$PY runner.py jobs/stage2.jsonl --workers 32
$PY analyze.py
```

The job files are the exact specifications: work IDs, seeds, budgets and weightings. Re-running skips finished jobs.

In this repository, model checkpoints (`*.pt`) and per-beat prediction CSVs are omitted for size; the run directories keep each run's `training.json` (spec, history, sampling schedule), `summary.json` and `eval_*.json`. The NHMM reads the classical `subdivision_hmm/` interval and onset artifacts, which are regenerated by that project's scripts.

# Beat-conditioned subdivision HMM: report

Standalone experiment in `subdivision_hmm/`; it reads ASAP/MAESTRO and (optionally) Explore This features without modifying them. Generated tables: [results/analysis.md](results/analysis.md); audit: [results/audit.md](results/audit.md).

## Answer

**Partly.** The formulation works: fixed lattices, a normalised occupancy likelihood and exact HMM inference. Given clean onsets, it recovers notated subdivision well. With performance-MIDI onsets the HMM reaches **0.861 balanced accuracy** on all labelled test beats and **0.944** on the high-confidence subset; chance is 0.20.

With onsets measured from the audio it does not. The best audio-derived configuration reaches **0.615**: the Gaussian template on Explore This's mel flux. Classical audio onset detection with the Poisson model gets **0.318**, predicting 2× for almost everything.

The bottleneck is the acoustic evidence, not the lattice, the likelihood form or the inference. This is the component a neural observation model would replace. A neural HMM is justified as the next step, with two caveats, both stated below:
- The ideal-onset result is close to an upper bound and partly circular.
- No audio-based model reached the reliability that the ideal-onset run shows is attainable.

## Setup

**Data.** ASAP performances with a positive score–performance alignment flag and an exact MAESTRO audio member: 501 performances, 169 works.
- Split is by **work**: 116 train / 35 val / 18 test. Test is the set of works in Explore This's `baseline_val` ASAP.
- One performance per train/val work; all 52 test performances. 55k / 17.5k / 32k beat intervals.
- GTZAN is not used.
- Leakage note: the Explore This checkpoint behind the `mel50` features was trained on `baseline_train`, which includes our train/val ASAP works but not the test works.

**Beats.** Beats are ASAP's ground-truth performance beats; intervals touching `bR` beats are dropped. Audio–annotation lag of the onset curve at beats: median 20 ms (audio), 0 ms (mel50); 11/203 performances exceed 50 ms.

**Reference labels** (`labels.py`). Each beat gets the *notated* subdivision of its measure: the coarsest r ∈ {1,2,3,4,6,8,12,16,24} whose lattice contains every score-MIDI attack in the measure, with a ±0.015-beat tolerance.
- Tuplets enter through their written positions.
- Ties and rests produce no attack.
- Grace notes and quintuplets fall off every lattice, so those measures are marked **unknown** (3.4 %).

"High confidence" means good score/performance note matching and at least 2 beats in the measure that actually sound a diagnostic position of the label.

This is notated subdivision. It is neither perceptual tatum nor performed onset density: the measure label differs from a beat's own attack lattice in 14 % of intervals. Label 6 often marks measures that mix duplets and triplets rather than true sextuplets.

Test coverage: 92 % of labelled beats fall in the vocabulary {1,2,3,4,6}, which is the oracle ceiling; 8× and finer, plus unknown, make up the rest.

**Onset sources** (`onsets.py`):
- `audio100`: librosa log-mel superflux envelope plus peak-picked onsets, at 100 fps.
- `mel50`: positive flux of Explore This's 50 fps log-mel input.
- `perfmidi`: performance-MIDI note-ons, a diagnostic ideal detector. **Never used as labels**, but strongly related to them: performed notes are the score attacks.

**Observation models** (`observation.py`). All ignore ±0.06 beat around the beats, and timing tolerance is in beat phase:
- **Poisson occupancy** is a normalised likelihood of the detected onset phases. Intensity is λ₀ + q·Σₖ N(φ; k/r, σ²). Each slot fires with expected count q, so missing notes cost q per slot and off-lattice onsets are explained by λ₀. Denser lattices pay for empty slots.
- **Gaussian template** is a normalised marginal likelihood of the resampled, standardised onset curve: N(0, s²I + τ_μ²11ᵀ + τ_a² h_r h_rᵀ). It uses positive and negative evidence. It treats correlated samples as independent and is therefore over-confident.
- **Naive slot sum** is an unnormalised potential on the raw non-negative curve, shown only to demonstrate the nested-lattice bias.

**Fitting.**
- Observation parameters are fitted **without labels**, by maximising the uniform-mixture marginal likelihood on train.
- Calibration exponent κ and self-transition probability p_stay are chosen on val by balanced accuracy.
- Test labels are used only for scoring.
- Inference is exact log-space forward–backward plus Viterbi (`hmm.py`, verified against brute-force enumeration).

## Results (test, vocabulary {1,2,3,4,6}, balanced accuracy)

| Onset source | Likelihood | Prior only | Independent | HMM, uniform | HMM, persistence | HMM, high-conf. |
|---|---|---:|---:|---:|---:|---:|
| perf-MIDI (ideal) | Poisson | 0.200 | 0.781 | 0.798 | **0.861** | **0.944** |
| perf-MIDI (ideal) | Gaussian | 0.200 | 0.592 | 0.591 | 0.552 | 0.621 |
| mel50 | Gaussian | 0.200 | 0.534 | 0.533 | **0.615** | 0.669 |
| mel50 | Poisson | 0.200 | 0.388 | 0.371 | 0.478 | 0.498 |
| audio100 | Gaussian | 0.200 | 0.344 | 0.344 | 0.240 | 0.256 |
| audio100 | Poisson | 0.200 | 0.266 | 0.269 | 0.318 | 0.340 |
| any | naive sum | 0.200 | ≤0.37 | ≤0.43 | ≤0.37 | – |

Per-class results, binary vs ternary, transitions and calibration are in `results/analysis.md`.

**Confusions** (rows = reference 1/2/3/4/6):
- **perf-MIDI Poisson HMM:** the main error is **4× → 2×** (1,773 of 13,250 beats). That is the predicted identifiability failure: 16th-note measures where the sounding notes sit mostly on 8th positions. 2× vs 3× is almost never confused (123 + 7 errors). 6× is weakest (0.64): mixed measures are pulled to 3×.
- **audio100 Poisson:** collapses to 2×. The fitted q and λ₀ make 2× explain the sparse, noisy detections most cheaply.
- **mel50 Gaussian:** 4× → 2× again (4,147 beats). 1× is often read as 6× (1,230 beats) because flux from sustained/other voices looks like dense subdivision.

**Persistence.**
- It helps whenever the evidence is informative: perf-MIDI Poisson 0.798 → 0.861, mel50 Gaussian 0.533 → 0.615.
- It cuts predicted changes to near the reference rate: 3.3 and 8.0 per 100 beats vs a reference 5.0, against 25–48 without.
- The ablation is flat over p_stay ∈ [0.5, 0.999] for mel50 Gaussian. For perf-MIDI it peaks at 0.5 and declines slightly with stronger persistence ([figures/p_stay_ablation.png](figures/p_stay_ablation.png)).
- With weak evidence, persistence simply locks in the wrong class. audio100 Gaussian *drops* to 0.24, with a single long run per piece. **Stability is not recovery.**

**Oracle vs inference.** Among HMM errors, the emission itself prefers the wrong class in 74–96 % of cases. Errors are therefore mainly observation failures, not decoding failures. Persistence fixes more beats than it breaks for every informative model (perf-MIDI Poisson: 2,494 fixed vs 967 broken). Coverage limits a further 8 % of beats, whose labels are outside the vocabulary.

**Vocabulary.**
- Adding 8× and 12× raises coverage to 0.978 but lowers balanced accuracy (perf-MIDI 0.861 → 0.640). Finer lattices are the least identifiable.
- Restricting to {2,3,4} raises balanced accuracy, but only by excluding classes.

**Naive sum.** On the raw non-negative curve it chooses the densest lattice (6×) almost everywhere, as predicted and as shown by the synthetic test. Normalised likelihoods are necessary.

**Calibration.** Posteriors are not calibrated (ECE 0.08–0.64). The Gaussian template is strongly over-confident, so its posteriors should not be read as probabilities.

## Synthetic tests (12, all pass)

- **Exactness:** forward–backward and Viterbi match brute force; uniform transitions equal independent decoding.
- **Recovery:** clean 1/2/3/4/6; 4× with 25 % dropped notes plus jitter; dotted syncopation; an accelerando; a 2→3 change with exactly one transition. Persistence bridges long-note beats where independent decoding falls to 1×.
- **Indistinguishable counterexamples:** 4× played on its 2× positions, and 6× played only on 3/6, give posteriors *identical* to a true 2×. No likelihood can separate them. Duplets plus triplets in one beat require 6×.
- **Naive-sum bias:** always picks 6× on a 2× pattern.

## Limitations

- The ideal-onset upper bound is partly circular: labels and observations both derive from the notes.
- Audio onset detection received no tuning, and a better classical detector might narrow the gap. I did not investigate further.
- Single corpus: solo piano only. No texture metadata beyond composer; per-work and per-composer tables are in `analysis.md`.
- The test set has 18 works, and work-level variation is large.

## Neural HMM: design only (not implemented)

p_θ(R | X, B) ∝ p(R₁) Πᵢ ψ_θ(Xᵢ, Bᵢ, Rᵢ) Πᵢ p(Rᵢ | Rᵢ₋₁), with the deterministic lattice and persistence prior unchanged.

- **ψ_θ** is a slot-occupancy model. A small network reads pretrained Explore This (or AlignBeat) frame features, which are frozen, at each candidate position s_{i,k}^{(r)}, interpolated in beat phase. It outputs per-slot onset probabilities q_θ. These replace the fixed q and λ₀ inside the *same* normalised Poisson/Bernoulli likelihood. This keeps the Occam penalty that prevents the density bias, and keeps the identifiability of the classical model.
- **Training:**
  - Maximise the marginal likelihood log Σ_R p(R, X) by forward–backward. This requires no labels.
  - Add the partially observed likelihood log Σ_{R consistent with the labels} p(R, X), clamping only labelled, high-confidence intervals. Unknown intervals stay latent.
  - Performance-MIDI onsets could supervise the per-slot onset head, as "is there a played onset here" rather than as subdivision labels.
- **Comparison:** test against the classical models above on the same work-disjoint split.

## Relationship to time signatures

The ratio R is not a time signature. The full model derives Y = (N, D) = f(G, U), with grouping structure G and notated unit U = 1/D. This experiment estimates neither G nor U, and no numerator/denominator variables were introduced. An HMM over R alone does not estimate time signatures: grouping and the mapping to the notated unit still have to be integrated.

## Reproduction

Requires Python 3.10 with numpy, scipy, librosa 0.11 and matplotlib; the optional `mel50` source also needs an
Explore This checkout and its feature store. Data locations are set in `config.py` (or the
`SUBHMM_ASAP`, `SUBHMM_MAESTRO_ZIP`, `SUBHMM_EXPLORE_THIS`, `SUBHMM_FEATURES` environment variables).

```sh
cd subdivision_hmm
PY=python
$PY -m unittest discover -s tests
$PY scripts/audit_labels.py
$PY scripts/extract_onsets.py
OMP_NUM_THREADS=1 $PY scripts/run_experiment.py
OMP_NUM_THREADS=1 $PY scripts/run_experiment.py --models naive --out-name results_naive_raw.csv
$PY scripts/analyze.py
```

Not committed (regenerated by the scripts): `results/intervals.jsonl`, `results/onsets/`, per-beat predictions `results/experiment/pred_*.csv`. The superseded naive run on the standardised curve, which hid the density bias, is not included.

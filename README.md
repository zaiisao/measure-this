# Measure This

Research notes for the Measure This project on beat, meter, subdivision, and time-signature inference.

## Preliminary result: frozen audio features for metrical inference

A controlled oracle-grid study tested whether Explore This's pretrained contextual audio features help a Bayesian metrical HMM distinguish beats-per-bar cycles (`T_D`) better than its handcrafted 20-dimensional observations. The encoder was frozen and only its pre-grid representation was used. This is a baseline finding to guide Measure This; it is not a Measure This model result.

### Main comparison

The HMM used the same boundary-aware transition topology, state definitions, corrected `T_D` labels, partial-label objective, Gaussian emission family, train/validation split, optimizer, update budget, and two random seeds across the primary representations. PCA and feature normalization were fit on training works only. The evaluation used four-measure, score-supported passages from a work-disjoint ASAP split, with 15 held-out works.

Work-macro validation results (mean across seeds):

| Observation representation | Exact `T_D` accuracy | Beat F1 | Downbeat F1 | Event-cycle accuracy | Correct `T_D` posterior mass |
|---|---:|---:|---:|---:|---:|
| Handcrafted 20D | .300 | .880 | .347 | .300 | .372 |
| Frozen Explore This, PCA-16 | .500 | .816 | .363 | .533 | .481 |
| Frozen Explore This, PCA-32 | .444 | .824 | .402 | .439 | .427 |
| Frozen Explore This, PCA-64 | **.656** | .876 | **.433** | **.661** | **.636** |

For PCA-64 minus handcrafted features, the paired bootstrap over 15 held-out works gave +.356 exact `T_D` accuracy (95% interval [.083, .625]) and +.344 event-cycle accuracy ([.078, .611]). Beat F1 was effectively unchanged (−.003 [−.092, .084]); the downbeat F1 gain was uncertain (+.085 [−.113, .302]). Exact `T_B` accuracy was also nearly unchanged (.794 versus .796).

### What the diagnostics suggest

- Frozen features carry useful metrical cues: held-out linear probes improved downbeat-versus-beat and beat-versus-subdivision discrimination over handcrafted features.
- On fixed candidate grids, the normalized Gaussian sequence likelihood favored the correct `T_D` more often with PCA-64. Mean correct-minus-best-alternative margins were +11.62 nats on 11 binary-grid works and +1.88 nats on four ternary-grid works. These are likelihoods over fixed embeddings, not over audio waveforms.
- Neighboring transformer outputs are correlated at 50 frames per second. When likelihoods were recomputed after retaining every fifth frame plus grid-event frames, the binary PCA-64 margin fell from +11.62 to +1.92 nats. The representation advantage remained, while posterior confidence decreased.
- An initialization control changed the beat/downbeat/cycle tradeoff modestly; it did not account for the representation gain. Remaining errors point to downbeat evidence, the Gaussian emission assumptions, and temporal dependence as areas to improve.

### Scope and caveats

The feature was the output after Explore This's third transformer block and final normalization: 256 dimensions at 50 fps, with bidirectional attention inside 1,500-frame (30-second) chunks. Grid and subgrid prediction layers were not run, and score timing was not supplied to feature extraction.

The pretrained checkpoint's manifest reports ASAP in its supervised training corpus, including an ASAP validation fold used in its original evaluation/selection. The HMM and PCA evaluation split was work-disjoint, but this is not an independent external-pretraining test. The ternary result covers only four held-out works, and the study did not test genuine meter changes.

**Decision:** the observation representation was a material bottleneck, and replacing it improved coherent `T_D`/cycle inference under oracle grids. The evidence does not establish blind-grid readiness. Improve and evaluate downbeat and ternary handling on oracle grids before relying on blind multi-grid or written time-signature predictions.

This note summarizes the comparison only. It does not include audio, feature caches, model checkpoints, or the exploratory experiment directories.

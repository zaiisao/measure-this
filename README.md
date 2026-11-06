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

## Beat This time-signature pipeline

The project-owned AlignBeat training code is linked at the repository root as `AlignBeat/`. Dataset repositories are submodules under `third_party/` for Beat This annotations, ASAP, Harmonix, and FiloSax. The FiloSax repository provides its source documentation and example materials; access to the full dataset is separate. Initialize the submodules after cloning:

```sh
git submodule update --init --recursive
```

Groove MIDI publishes its official data as an archive rather than a Git repository. Download the archive from the [official dataset page](https://magenta.withgoogle.com/datasets/groove), extract its `groove/` directory under `third_party/`, and rename it to `third_party/groove_midi/`. That folder contains `info.csv` and the referenced MIDI files, and is ignored by Git.

Run the single pipeline across all Beat This datasets:

```sh
python scripts/meter_pipeline.py
```

The pipeline copies the Beat This annotation tree to `measure_this_annotations/`. In each matched beat file, it adds the written time signature as a third column on the first beat and on each downbeat where the signature changes. ASAP changes are placed at their timestamped downbeats; Groove MIDI, Harmonix, and FiloSax provide the opening signature. The source Beat This submodule is left unchanged. Summary and review CSVs are written under `results/meter_pipeline/`; `--out` changes that report directory.

Datasets without a direct time-signature source are listed as unresolved until score discovery and score-to-performance alignment are implemented. A song-level signature is kept as candidate evidence and does not produce timestamped change events.

## Beat-conditioned subdivision HMM

[`subdivision_hmm/`](subdivision_hmm/) is a standalone experiment: given ground-truth beats, infer each beat's notated subdivision ratio (1/2/3/4/6) from onset evidence with fixed beat-relative lattices, normalized onset likelihoods and an exact HMM persistence prior, on a work-disjoint ASAP split. With ideal (performance-MIDI) onsets it reaches 0.861 test balanced accuracy (0.944 on high-confidence labels); from audio-derived onsets the best is 0.615, so the acoustic evidence is the bottleneck. See [the report](subdivision_hmm/REPORT.md).

## Neural subdivision HMM and training-data scaling

[`neural_subdivision_hmm/`](neural_subdivision_hmm/) extends the classical HMM with a normalized neural Gaussian emission trained with exact forward–backward and a weighted posterior-state term ([report](neural_subdivision_hmm/REPORT.md), [diagnostics](neural_subdivision_hmm/diagnostics/REPORT.md)). The [data-scaling study](neural_subdivision_hmm/data_scaling/REPORT.md) (141 runs) shows that the earlier 30-work > 96-work result was an optimization-schedule artifact: at matched optimizer updates, 96 works reaches 0.768 test balanced accuracy versus 0.769 for the original 30-work subset and 0.759 for random 30-work subsets, with performance plateauing beyond about 30–50 works.

# Neural Subdivision HMM

Standalone experiment replacing the classical mel50 Gaussian likelihood with a compact, normalized neural Gaussian emission while retaining the five-state HMM and fixed beat-relative subdivision grid. The original `subdivision_hmm/` and `beat_aligned_onsets/` projects are read-only inputs.

## Model

For each beat interval, a shared MLP conditions on a state embedding, a fixed smooth template made from the candidate subdivision positions, and standardized log beat duration. It predicts a mean and a diagonal-plus-low-rank covariance for the existing 48-dimensional mel50 envelope. It does not inspect the observation when constructing its distribution parameters. The exact normalized Gaussian density includes the log determinant and normalizing constant; the low-rank case uses Woodbury and the determinant lemma. The HMM uses a normalized persistence transition and exact log-space forward, forward-backward, and Viterbi algorithms.

Unlabeled intervals use the unconstrained forward likelihood. Exact labels mask all but their state; family-only labels mask to `{2,4}` or `{3,6}`; unrecognized labels allow all five states. Missing emissions contribute zero log density while preserving the latent state and its transitions.

## Reproduce

The runs used Python 3.10, PyTorch 2.6 CPU, NumPy, pandas, and SciPy. Set the repository root as the working directory and use a Python environment with those packages.

```sh
python -m unittest discover -s neural_subdivision_hmm/tests -v
python neural_subdivision_hmm/train.py --name supervised_exact --mode exact --rank 2 --hidden 48 --epochs 10 --patience 3
python neural_subdivision_hmm/train.py --name supervised_only --mode supervised_only --rank 2 --hidden 48 --epochs 10 --patience 3
python neural_subdivision_hmm/train.py --name unsupervised --mode unsupervised --rank 2 --hidden 48 --epochs 10 --patience 3
python neural_subdivision_hmm/train.py --name mixed --mode mixed --rank 2 --hidden 48 --epochs 10 --patience 3
python neural_subdivision_hmm/train.py --name partial_family --mode partial_family --rank 2 --hidden 48 --epochs 10 --patience 3
python neural_subdivision_hmm/train.py --name supervised_diag --mode supervised_only --rank 0 --hidden 48 --epochs 10 --patience 3
python neural_subdivision_hmm/train.py --name capacity16 --mode supervised_only --rank 2 --hidden 16 --epochs 10 --patience 3
python neural_subdivision_hmm/train.py --name transition095 --mode supervised_only --rank 2 --hidden 48 --p-stay 0.95 --epochs 10 --patience 3
python neural_subdivision_hmm/evaluate.py
python neural_subdivision_hmm/make_figures.py
```

The exact settings and validation decoder grid are in [experiment_config.json](experiment_config.json). Training JSON and checkpoints preserve the run histories and weights. `evaluate.py` selects posterior temperature and transition persistence on validation balanced accuracy, then reports both raw generative inference and the secondary calibrated decoder on test. Test labels do not choose model or decoder settings.

## Results

See [the research report](REPORT.md). Figures, predictions, per-work metrics, subgroup summaries, and metric JSON files are under `results/`.

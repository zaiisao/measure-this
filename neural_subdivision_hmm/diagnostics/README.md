# NHMM diagnosis tools

These diagnostic scripts are isolated from the original NHMM trainer and checkpoints. They reuse its normalized Gaussian emission, frozen mel50 data adapter, and exact HMM forward recursion.

## Exact objective decomposition

For each sequence, the code evaluates the unconstrained and constrained forward log likelihoods:

```text
acoustic_nll = -log p(O | B)
joint_nll    = -log p(O, C | B)
state_nll    = joint_nll - acoustic_nll = -log p(C | O, B)
```

More explicitly, `state_nll = log p(O|B) - log p(O,C|B)`. All terms come from the same normalized generative emission and HMM. Unknown masks allow every state, so their posterior state NLL is exactly zero. Missing observations are neutralized while retaining their HMM position and transitions.

The original trainer sums joint sequence NLLs and divides by observed acoustic intervals. For a batch, this equals

```text
acoustic_nll_per_observed
+ lambda_state * state_nll_per_informative * n_informative / n_observed
```

Therefore `lambda_state=1` is exactly the original constrained joint objective. `state_nll_per_informative` is separately logged so runs with different supervision density can be compared; the multiplier `n_informative/n_observed` preserves the original objective's units. Values above one strengthen posterior state supervision without changing the acoustic density or HMM probability model.

The trainer logs acoustic NLL, posterior state NLL, constrained joint NLL, the weighted objective, observed/informative interval counts, and optional component gradient norms.

The data-selection and high-confidence audit is reproduced by `audit_supervision.py`. The λ sweep is in `weighted_train.py`; `--high-confidence-only` relaxes low-confidence singleton masks without dropping their audio, and `--limit-sequences N` keeps the same five fully exact sequences then adds a deterministic set of partial-label sequences. The classical-structured zero-mean emission is in `structured_emission.py`, initialized from the exact fitted baseline; `structured_train.py` accepts the same objective weight and optional fixed-grid/tempo conditioning.

## Tests and commands

From repository root:

```sh
python -m unittest discover -s neural_subdivision_hmm/diagnostics/tests -v
python neural_subdivision_hmm/diagnostics/weighted_train.py --name exact_l1 --mode exact --lambda-state 1 --epochs 10
python neural_subdivision_hmm/diagnostics/weighted_train.py --name exact_l5 --mode exact --lambda-state 5 --epochs 10
python neural_subdivision_hmm/diagnostics/weighted_train.py --name broad_l1 --mode supervised_only --lambda-state 1 --epochs 10
python neural_subdivision_hmm/diagnostics/weighted_train.py --name broad_l5 --mode supervised_only --lambda-state 5 --epochs 10
python neural_subdivision_hmm/diagnostics/weighted_train.py --name broad_l10 --mode supervised_only --lambda-state 10 --epochs 10
python neural_subdivision_hmm/diagnostics/weighted_train.py --name incremental_n30 --mode supervised_only --limit-sequences 30 --lambda-state 10 --epochs 10
python neural_subdivision_hmm/diagnostics/structured_train.py --name structured_l10 --mode supervised_only --lambda-state 10 --condition-on-ratio --condition-on-duration
```

Recommended initial sweep: `lambda_state ∈ {0,1,2,5,10}` on the same exact-five-sequence and 96-sequence supervised sets, with fixed initialization and seed. Select using validation posterior inference; then repeat only promising settings with at least three seeds. Checkpoint files go under `diagnostics/results/`; run summaries go beside this README.

The executed results and interpretation are in [REPORT.md](REPORT.md). Training histories, held-out predictions, paired work bootstraps, state-use summaries, and SVG figures are in `diagnostics/results/`.

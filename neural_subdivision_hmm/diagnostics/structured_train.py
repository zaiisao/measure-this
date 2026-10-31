#!/usr/bin/env python3
"""Train the classical-covariance neural emission with exact HMM supervision."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from neural_subdivision_hmm.data import RATIOS, choose_sequences, load_sequences, sequence_summary, validation_sequences
from neural_subdivision_hmm.diagnostics.objective import batch_objective, evaluate_objective
from neural_subdivision_hmm.diagnostics.structured_emission import StructuredNeuralGaussian
from neural_subdivision_hmm.training import Budget, fit
from neural_subdivision_hmm.utils import save_checkpoint, seed_everything, write_json
from subdivision_hmm.hmm import persistence_log_transition

RESULTS = Path(__file__).resolve().parent / 'results'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--lambda-state', type=float, default=10)
    ap.add_argument('--mode', choices=['exact', 'supervised_only', 'mixed', 'unsupervised', 'partial_family'], default='supervised_only')
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--patience', type=int, default=3); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--condition-on-ratio', action='store_true'); ap.add_argument('--condition-on-duration', action='store_true')
    a = ap.parse_args()
    seed_everything(a.seed)
    splits, norm = load_sequences()
    train, mode = choose_sequences(splits['train'], a.mode)
    val, mode_val = validation_sequences(splits['val'], a.mode)
    model = StructuredNeuralGaussian.from_classical_fit(m=norm['feature_dim'], condition_on_ratio=a.condition_on_ratio,
                                                         condition_on_duration=a.condition_on_duration, scale_floor=.05)
    init = torch.full((len(RATIOS),), -np.log(len(RATIOS))); trans = persistence_log_transition(len(RATIOS), .99)
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = {'structured': True, 'mode': a.mode, 'lambda_state': a.lambda_state, 'seed': a.seed,
              'condition_on_ratio': a.condition_on_ratio, 'condition_on_duration': a.condition_on_duration,
              'p_stay': .99, 's2': 1., 'tm2': 1., 'ta2': 16., 'width': .03, 'scale_floor': .05}

    def loss(m, batch):
        # The model starts at the known classical covariance; leave its scale departures
        # unregularized in this first controlled comparison.
        value, t = batch_objective(m, batch, mode, init, trans, a.lambda_state, reg_weight=0)
        return value, {'ac': float(t.acoustic_nll_sum), 'st': float(t.state_nll_sum), 'joint': float(t.joint_nll_sum),
                       'n': t.observed_intervals, 'c': t.informative_intervals}

    def evaluate(m):
        t = evaluate_objective(m, val, mode_val, init, trans, a.lambda_state)
        return {'val_acoustic_nll_per_interval': float(t.acoustic_nll_per_observed),
                'val_state_nll_per_constraint': float(t.state_nll_per_informative),
                'val_joint_nll_per_interval': float(t.joint_nll_per_observed), 'val_weighted_objective': float(t.loss)}

    result = fit(model, train, loss, evaluate, Budget('epochs', a.epochs, a.patience),
                 criteria=(('val_weighted_objective', 'min'),),
                 train_summary=lambda s: {'train_acoustic_nll_per_interval': s['ac'] / s['n'],
                                          'train_state_nll_per_constraint': s['st'] / max(s['c'], 1),
                                          'train_joint_nll_per_interval': s['joint'] / s['n']},
                 on_best=lambda k, step: save_checkpoint(RESULTS / f'{a.name}.pt', model, config, norm))
    for row in result.history:
        print(f"{a.name} epoch={row['epoch']:02d} acoustic={row['train_acoustic_nll_per_interval']:.3f} "
              f"state/label={row['train_state_nll_per_constraint']:.3f} val={row['val_weighted_objective']:.3f}", flush=True)
    best, best_step = result.best['val_weighted_objective']
    write_json(RESULTS / f'{a.name}_training.json', {
        'name': a.name, 'mode': a.mode, 'lambda_state': a.lambda_state, 'seed': a.seed,
        'best_epoch': next(r['epoch'] for r in result.history if r['step'] == best_step), 'best_val_objective': best,
        'train_counts': sequence_summary(train), 'validation_counts': sequence_summary(val), 'history': result.history,
        'initial_emission_matches_classical': True, 'scale_floor': .05, 'checkpoint': str(RESULTS / f'{a.name}.pt')})


if __name__ == '__main__':
    main()

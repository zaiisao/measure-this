#!/usr/bin/env python3
"""Train a normalized Gaussian NHMM with exact differentiable HMM marginalization
(constrained joint NLL, i.e. the lambda = 1 objective)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from neural_subdivision_hmm.data import RATIOS, choose_sequences, load_sequences, sequence_summary, validation_sequences
from neural_subdivision_hmm.diagnostics.objective import batch_objective, evaluate_objective
from neural_subdivision_hmm.model import NeuralGaussianEmission
from neural_subdivision_hmm.training import Budget, fit
from neural_subdivision_hmm.utils import save_checkpoint, seed_everything, write_json
from subdivision_hmm.hmm import persistence_log_transition

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'checkpoints'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--mode', choices=['exact', 'unsupervised', 'supervised_only', 'mixed', 'partial_family'], required=True)
    ap.add_argument('--rank', type=int, default=2); ap.add_argument('--hidden', type=int, default=48)
    ap.add_argument('--epochs', type=int, default=14); ap.add_argument('--batch-sequences', type=int, default=4)
    ap.add_argument('--lr', type=float, default=8e-4); ap.add_argument('--p-stay', type=float, default=.99)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--patience', type=int, default=4)
    a = ap.parse_args()
    seed_everything(a.seed)
    splits, normalization = load_sequences()
    train, constraint_mode = choose_sequences(splits['train'], a.mode)
    val, val_mode = validation_sequences(splits['val'], a.mode)
    model = NeuralGaussianEmission(ratios=RATIOS, m=normalization['feature_dim'], hidden=a.hidden, rank=a.rank)
    log_init = torch.full((len(RATIOS),), -np.log(len(RATIOS)))
    log_trans = persistence_log_transition(len(RATIOS), a.p_stay)
    OUT.mkdir(parents=True, exist_ok=True)
    config = {'mode': a.mode, 'rank': a.rank, 'hidden': a.hidden, 'p_stay': a.p_stay, 'seed': a.seed}

    def loss(m, batch):
        value, terms = batch_objective(m, batch, constraint_mode, log_init, log_trans, 1.0)
        return value, {'joint': float(terms.joint_nll_sum), 'observed': terms.observed_intervals}

    result = fit(model, train, loss,
                 lambda m: {'val_nll_per_observed_interval': float(
                     evaluate_objective(m, val, val_mode, log_init, log_trans, 1.0).loss)},
                 Budget('epochs', a.epochs, a.patience, batch_size=a.batch_sequences),
                 criteria=(('val_nll_per_observed_interval', 'min'),), lr=a.lr,
                 train_summary=lambda s: {'train_nll_per_observed_interval': s['joint'] / max(s['observed'], 1)},
                 on_best=lambda k, step: save_checkpoint(OUT / f'{a.name}.pt', model, config, normalization))
    for row in result.history:
        print(f"{a.name} epoch={row['epoch']:02d} train_nll={row['train_nll_per_observed_interval']:.4f} "
              f"val_nll={row['val_nll_per_observed_interval']:.4f}", flush=True)
    best, best_step = result.best['val_nll_per_observed_interval']
    write_json(HERE / 'results' / f'{a.name}_training.json', {
        'name': a.name, 'mode': a.mode, 'constraint_mode': constraint_mode, 'val_constraint_mode': val_mode, **config,
        'epochs_run': len(result.history), 'best_epoch': next(r['epoch'] for r in result.history if r['step'] == best_step),
        'best_val_nll': best, 'normalization': normalization, 'train_counts': sequence_summary(train),
        'validation_counts': sequence_summary(val),
        'train_labeled_interval_count': int(sum(np.isin(q.labels, RATIOS).sum() for q in train)),
        'history': result.history, 'checkpoint': str(OUT / f'{a.name}.pt')})


if __name__ == '__main__':
    main()

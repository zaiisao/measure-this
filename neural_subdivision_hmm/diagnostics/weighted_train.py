#!/usr/bin/env python3
"""Train existing generative NHMM with decomposed/weighted posterior supervision.

This preserves the normalized emission density and exact HMM forward likelihood.
No classifier logits or CRF scores are introduced.
"""
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

OUT = Path(__file__).resolve().parent / 'results'


def gradient_norms(model, acoustic_loss, state_loss):
    params = [p for p in model.parameters() if p.requires_grad]

    def norm(loss):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((g.square().sum() for g in grads if g is not None), loss.new_zeros(()))))
    return {'acoustic_grad_l2': norm(acoustic_loss), 'state_contribution_grad_l2': norm(state_loss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--mode', choices=['exact', 'supervised_only', 'mixed', 'partial_family', 'unsupervised'], required=True)
    ap.add_argument('--lambda-state', type=float, default=1.0)
    ap.add_argument('--rank', type=int, default=2); ap.add_argument('--hidden', type=int, default=48)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--batch-sequences', type=int, default=4); ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--p-stay', type=float, default=.99); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gradient-diagnostics', action='store_true')
    ap.add_argument('--high-confidence-only', action='store_true',
                    help='leave low-confidence labels unconstrained; does not remove their audio')
    ap.add_argument('--limit-sequences', type=int, default=0,
                    help='for supervised-only mode, retain the same five fully exact sequences and add a seeded number of other labeled sequences')
    a = ap.parse_args()
    seed_everything(a.seed)
    splits, norm = load_sequences()
    train, mode = choose_sequences(splits['train'], a.mode)
    if a.limit_sequences:
        if a.mode != 'supervised_only' or a.limit_sequences < 5:
            raise ValueError('--limit-sequences requires supervised_only mode and a value >= 5')
        exact = [q for q in train if q.fully_exact]
        extras = [q for q in train if not q.fully_exact]
        extras = [extras[i] for i in np.random.default_rng(20261031).permutation(len(extras))]
        train = exact + extras[:a.limit_sequences - len(exact)]
    val, val_mode = validation_sequences(splits['val'], a.mode)
    model = NeuralGaussianEmission(ratios=RATIOS, m=norm['feature_dim'], hidden=a.hidden, rank=a.rank)
    log_init = torch.full((len(RATIOS),), -np.log(len(RATIOS)))
    log_trans = persistence_log_transition(len(RATIOS), a.p_stay)
    OUT.mkdir(parents=True, exist_ok=True)
    config = {'mode': a.mode, 'rank': a.rank, 'hidden': a.hidden, 'p_stay': a.p_stay, 'seed': a.seed,
              'lambda_state': a.lambda_state}
    first_batch = {}

    def loss(m, batch):
        value, terms = batch_objective(m, batch, mode, log_init, log_trans, a.lambda_state,
                                       high_confidence_only=a.high_confidence_only)
        if a.gradient_diagnostics and 'diag' not in first_batch and terms.informative_intervals:
            state = terms.state_nll_sum / max(terms.observed_intervals, 1) * a.lambda_state
            first_batch['diag'] = gradient_norms(m, terms.acoustic_nll_per_observed, state)
        return value, {'acoustic': float(terms.acoustic_nll_sum), 'state': float(terms.state_nll_sum),
                       'joint': float(terms.joint_nll_sum), 'objective': float(terms.loss) * terms.observed_intervals,
                       'observed': terms.observed_intervals, 'informative': terms.informative_intervals}

    def summary(s):
        n, i = max(s['observed'], 1), max(s['informative'], 1)
        return {'acoustic_nll_per_observed': s['acoustic'] / n, 'state_nll_per_informative': s['state'] / i,
                'state_contribution_per_observed': s['state'] / n, 'joint_nll_per_observed': s['joint'] / n,
                'weighted_objective_per_observed': s['objective'] / n, 'observed_intervals': int(s['observed']),
                'informative_intervals': int(s['informative']),
                'gradient_diagnostics_first_batch': first_batch.pop('diag', None)}

    def evaluate(m):
        t = evaluate_objective(m, val, val_mode, log_init, log_trans, a.lambda_state,
                               high_confidence_only=a.high_confidence_only)
        return {'val_acoustic_nll_per_observed': float(t.acoustic_nll_per_observed),
                'val_state_nll_per_informative': float(t.state_nll_per_informative),
                'val_joint_nll_per_observed': float(t.joint_nll_per_observed),
                'val_weighted_objective_per_observed': float(t.loss)}

    result = fit(model, train, loss, evaluate, Budget('epochs', a.epochs, a.patience, batch_size=a.batch_sequences),
                 criteria=(('val_weighted_objective_per_observed', 'min'),), lr=a.lr, train_summary=summary,
                 on_epoch_start=lambda epoch: first_batch.clear(),
                 on_best=lambda k, step: save_checkpoint(OUT / f'{a.name}.pt', model, config, norm))
    for row in result.history:
        print(f"{a.name} epoch={row['epoch']:02d} acoustic={row['acoustic_nll_per_observed']:.3f} "
              f"state/info={row['state_nll_per_informative']:.3f} joint={row['joint_nll_per_observed']:.3f} "
              f"objective={row['weighted_objective_per_observed']:.3f} val={row['val_weighted_objective_per_observed']:.3f}",
              flush=True)
    best, best_step = result.best['val_weighted_objective_per_observed']
    write_json(OUT.parent / f'{a.name}_diagnostic_training.json', {
        'name': a.name, 'mode': a.mode, 'constraint_mode': mode, 'val_constraint_mode': val_mode, **config,
        'lambda_1_identity': 'at lambda=1, acoustic NLL/observed + posterior-state NLL/observed equals original constrained joint NLL/observed',
        'high_confidence_only': a.high_confidence_only,
        'state_normalization': 'reported per informative constrained interval, then weighted by n_informative/n_observed in objective',
        'limit_sequences': a.limit_sequences, 'selected_performance_ids': [q.performance_id for q in train],
        'epochs_run': len(result.history), 'best_epoch': next(r['epoch'] for r in result.history if r['step'] == best_step),
        'best_val_weighted_objective': best, 'train_counts': sequence_summary(train),
        'validation_counts': sequence_summary(val), 'history': result.history, 'checkpoint': str(OUT / f'{a.name}.pt')})


if __name__ == '__main__':
    main()

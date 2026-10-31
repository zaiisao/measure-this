#!/usr/bin/env python3
"""Validation-selected evaluation for the classical-initialized structured NHMM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neural_subdivision_hmm.data import load_sequences
from neural_subdivision_hmm.diagnostics.structured_emission import StructuredNeuralGaussian
from neural_subdivision_hmm.evaluate import bacc_df, evaluate_model, work_bootstrap

RESULTS = Path(__file__).resolve().parent / 'results'


def load_model(path, norm):
    if path is None:
        return StructuredNeuralGaussian.from_classical_fit(m=norm['feature_dim'])
    ck = torch.load(path, map_location='cpu', weights_only=False); c = ck['config']
    m = StructuredNeuralGaussian.from_classical_fit(m=norm['feature_dim'], s2=c['s2'], tm2=c['tm2'], ta2=c['ta2'], width=c['width'],
                                                    condition_on_ratio=c['condition_on_ratio'],
                                                    condition_on_duration=c['condition_on_duration'],
                                                    scale_floor=c.get('scale_floor', 1e-6))
    m.load_state_dict(ck['model'])
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--name', default='structured_initial'); ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--train-p-stay', type=float, default=.99); a = ap.parse_args(); torch.set_num_threads(2)
    splits, norm = load_sequences(); model = load_model(a.checkpoint, norm); model.eval()
    ev = evaluate_model(model, splits['val'], splits['test'], a.train_p_stay)
    RESULTS.mkdir(exist_ok=True)
    ev['decoded'].to_csv(RESULTS / f'{a.name}_all_test_predictions.csv', index=False)
    ev['exact'].to_csv(RESULTS / f'{a.name}_test_predictions.csv', index=False)
    out = {'name': a.name, 'initialized_from_classical': a.checkpoint is None, 'train_p_stay': a.train_p_stay,
           'validation_selected_p_stay': ev['p_stay'], 'validation_selected_temperature': ev['temperature'],
           'raw_metrics': ev['raw_metrics'], 'calibrated_metrics': ev['decoded_metrics'], 'marginal_nll_per_interval': ev['raw_nll'],
           'family_metrics': ev['family_metrics'], 'predicted_state_occupancy_all_intervals': ev['occupancy'],
           'predicted_transitions_per_100_adjacent_beats': ev['transitions_per_100'],
           'work_bootstrap_bacc_ci95': work_bootstrap(ev['exact'], bacc_df, n=2000),
           'checkpoint': str(a.checkpoint) if a.checkpoint else None}
    (RESULTS / f'{a.name}_evaluation.json').write_text(json.dumps(out, indent=2))
    print(a.name, 'raw/cal', ev['raw_metrics']['balanced_accuracy'], ev['decoded_metrics']['balanced_accuracy'],
          'NLL', ev['raw_nll'], 'family AUC', ev['family_metrics']['auc'])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Evaluate weighted-supervision checkpoints using validation-only decoder selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_subdivision_hmm.data import load_sequences
from neural_subdivision_hmm.evaluate import bacc_df, evaluate_model, model_from_checkpoint, work_bootstrap

RESULTS = Path(__file__).resolve().parent / 'results'


def evaluate_one(name, checkpoint, data):
    model, cfg = model_from_checkpoint(checkpoint)
    ev = evaluate_model(model, data['val'], data['test'], cfg['p_stay'])
    result = {'name': name, 'lambda_state': cfg.get('lambda_state'), 'p_stay_raw': cfg['p_stay'],
              'validation_selected_p_stay': ev['p_stay'], 'validation_selected_temperature': ev['temperature'],
              'raw_test_metrics': ev['raw_metrics'], 'calibrated_test_metrics': ev['decoded_metrics'],
              'family_metrics': ev['family_metrics'], 'work_bootstrap_bacc_95ci': work_bootstrap(ev['exact'], bacc_df, n=2000),
              'predicted_state_occupancy_all_intervals': ev['occupancy'],
              'predicted_transitions_per_100_adjacent_beats': ev['transitions_per_100'],
              'test_marginal_nll_per_interval': ev['raw_nll'],
              'training_metadata': str(RESULTS / f'{name}_diagnostic_training.json')}
    RESULTS.mkdir(parents=True, exist_ok=True)
    ev['decoded'].to_csv(RESULTS / f'{name}_all_test_predictions.csv', index=False)
    ev['exact'].to_csv(RESULTS / f'{name}_test_predictions.csv', index=False)
    (RESULTS / f'{name}_evaluation.json').write_text(json.dumps(result, indent=2))
    print(name, 'lambda', cfg.get('lambda_state'), 'raw/cal BAcc', ev['raw_metrics']['balanced_accuracy'],
          ev['decoded_metrics']['balanced_accuracy'], 'family AUC', ev['family_metrics']['auc'], flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('names', nargs='+'); a = ap.parse_args()
    data, _ = load_sequences()
    results = [evaluate_one(name, RESULTS / f'{name}.pt', data) for name in a.names]
    (RESULTS / 'weighted_comparison.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Matched evaluation against the saved classical HMM and older family reference."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_subdivision_hmm.data import K, RATIOS, load_sequences
from neural_subdivision_hmm.model import NeuralGaussianEmission
from subdivision_hmm import metrics as M
from subdivision_hmm.hmm import exclude_missing, forward_backward, persistence_log_transition, viterbi

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; RES = HERE / 'results'; CKPT = RES / 'checkpoints'
DEVICE = torch.device('cpu')
DECODER_GRID = [(p, t) for p in (.95, .99, .999) for t in (.25, .5, 1.0)]


def work_bootstrap(df, metric, n=1000, seed=19):
    """Percentile interval of `metric` over resampled musical works."""
    rng = np.random.default_rng(seed); works = df.work_id.unique()
    groups = {w: df[df.work_id == w] for w in works}
    values = [metric(pd.concat([groups[w] for w in rng.choice(works, len(works), replace=True)], ignore_index=True))
              for _ in range(n)]
    values = [v for v in values if np.isfinite(v)]
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def bacc_df(d):
    return M.balanced_accuracy(d.label.to_numpy(), d.pred.to_numpy())


@torch.no_grad()
def score_sequences(model, seqs, temperature=1.0, p_stay=.99):
    model.eval(); init = torch.full((K,), -math.log(K), device=DEVICE); trans = persistence_log_transition(K, p_stay)
    out = []; nll_sum = 0.0; n_obs = 0
    for q in seqs:
        observed = torch.from_numpy(q.observed)
        raw = exclude_missing(model.log_prob(torch.from_numpy(q.y), torch.from_numpy(q.log_duration)), observed)
        post, logz = forward_backward(init, trans, raw * temperature)
        path = viterbi(init, trans, raw * temperature)
        nll_sum -= float(logz); n_obs += int(observed.sum())
        posts = post.numpy(); pred = np.array(RATIOS)[path.numpy()]
        for i in range(q.n):
            out.append({'performance_id': q.performance_id, 'work_id': q.work_id, 'stem': q.stem,
                        'beat_index': int(q.beat_index[i]), 'label': int(q.labels[i]), 'observed': bool(q.observed[i]),
                        'pred': int(pred[i]), 'post_max': float(posts[i].max()),
                        **{f'post_{r}': float(posts[i, j]) for j, r in enumerate(RATIOS)},
                        'log_duration_z': float(q.log_duration[i])})
    return pd.DataFrame(out), nll_sum / max(n_obs, 1)


def attach_old_family(df):
    with gzip.open(ROOT / 'beat_aligned_onsets/data/asap_score_reference.csv.gz', 'rt') as f:
        ref = pd.read_csv(f, usecols=['stem', 'beat_index', 'ref_family_ctx4'])
    return df.merge(ref, on=['stem', 'beat_index'], how='left')


def exact_metrics(df):
    z = df[(df.observed) & df.label.isin(RATIOS)].copy()
    y, pred = z.label.to_numpy(), z.pred.to_numpy()
    post = z[[f'post_{r}' for r in RATIOS]].to_numpy(); confmax = post.max(1)
    return z, {'n': len(z), 'class_counts': {str(r): int((y == r).sum()) for r in RATIOS},
               'balanced_accuracy': M.balanced_accuracy(y, pred), 'accuracy': float(np.mean(y == pred)),
               'per_class_recall': M.per_class_recall(y, pred), 'confusion': M.confusion(y, pred).tolist(),
               'ece': M.ece(confmax, pred == y), 'brier': M.brier(y, post), 'mean_confidence': float(confmax.mean()),
               'fraction_confidence_ge_0_9': float(np.mean(confmax >= .9))}


def _family(df):
    z = df[df.ref_family_ctx4.isin(['binary', 'ternary'])].copy()
    z['family_y'] = (z.ref_family_ctx4 == 'ternary').astype(int)
    z['family_score'] = np.log(z.post_3 + z.post_6 + 1e-12) - np.log(z.post_2 + z.post_4 + 1e-12)
    return z


def family_metrics(df, val_df=None):
    """Binary/ternary family AUC and balanced accuracy at a validation-chosen threshold."""
    z = _family(df)
    threshold = 0.0
    if val_df is not None:
        v = _family(val_df); y = v.family_y.to_numpy()
        cands = np.unique(np.r_[0.0, v.family_score.to_numpy(), 1.0])
        threshold = max(cands, key=lambda t: M.balanced_accuracy(y, (v.family_score.to_numpy() >= t).astype(int), (0, 1)))
    z['family_pred'] = (z.family_score >= threshold).astype(int)
    y = z.family_y.to_numpy()
    return z, {'n': len(z), 'binary': int((y == 0).sum()), 'ternary': int((y == 1).sum()),
               'auc': M.auc(y, z.family_score.to_numpy()),
               'balanced_accuracy': M.balanced_accuracy(y, z.family_pred.to_numpy(), (0, 1)),
               'accuracy': float(np.mean(y == z.family_pred)), 'threshold_val': float(threshold)}


def choose_decoder(model, val):
    """(p_stay, temperature, val BAcc) maximizing validation balanced accuracy; ties prefer T=1, p=.99."""
    cands = []
    for p, t in DECODER_GRID:
        _, m = exact_metrics(score_sequences(model, val, t, p)[0])
        cands.append((m['balanced_accuracy'], -abs(t - 1), -abs(p - .99), p, t))
    bacc, _, _, p, t = max(cands)
    return p, t, bacc


def evaluate_model(model, val, test, raw_p_stay=.99):
    """Validation-chosen decoder plus raw and decoded test metrics, as used by every evaluator."""
    p, t, val_bacc = choose_decoder(model, val)
    raw, raw_nll = score_sequences(model, test, 1.0, raw_p_stay)
    dec, _ = score_sequences(model, test, t, p)
    _, raw_metrics = exact_metrics(raw)
    exact, dec_metrics = exact_metrics(dec)
    dec = attach_old_family(dec)
    _, fam = family_metrics(dec, attach_old_family(score_sequences(model, val, t, p)[0]))
    return {'p_stay': p, 'temperature': t, 'val_decoded_bacc': val_bacc, 'raw': raw, 'raw_nll': raw_nll,
            'raw_metrics': raw_metrics, 'decoded': dec, 'exact': exact, 'decoded_metrics': dec_metrics,
            'family_metrics': fam, 'transitions_per_100': M.transitions_per_100(dec),
            'occupancy': {str(r): int((dec.pred == r).sum()) for r in RATIOS}}


def model_from_checkpoint(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ck['config']; norm = ck['duration_normalization']
    model = NeuralGaussianEmission(ratios=RATIOS, m=norm['feature_dim'], hidden=cfg['hidden'], rank=cfg['rank'])
    model.load_state_dict(ck['model']); model.eval()
    return model, cfg


def classical_direct(test_seqs):
    pred = pd.read_csv(ROOT / 'subdivision_hmm/results/experiment/pred_mel50_gauss_hmm.csv')
    by = {(r.performance_id, int(r.beat_index)): r for r in pred.itertuples()}
    init = torch.full((K,), -math.log(K)); trans = persistence_log_transition(K, .99); rows = []; nll = 0; nobs = 0
    for q in test_seqs:
        e = torch.tensor([[float(getattr(by[(q.performance_id, int(bi))], f'll_{r}')) for r in RATIOS]
                          for bi in q.beat_index], dtype=torch.float32)
        post, z = forward_backward(init, trans, e); path = viterbi(init, trans, e); nll -= float(z); nobs += q.n
        p = post.numpy(); pr = np.array(RATIOS)[path.numpy()]
        for i in range(q.n):
            rows.append({'performance_id': q.performance_id, 'work_id': q.work_id, 'stem': q.stem,
                         'beat_index': int(q.beat_index[i]), 'label': int(q.labels[i]), 'observed': bool(q.observed[i]),
                         'pred': int(pr[i]), 'post_max': float(p[i].max()), **{f'post_{r}': float(p[i, j]) for j, r in enumerate(RATIOS)}})
    return pd.DataFrame(rows), nll / max(nobs, 1)


def main():
    torch.set_num_threads(2); splits, norm = load_sequences(); RES.mkdir(exist_ok=True)
    test = splits['test']; val = splits['val']; records = []
    for name in ['supervised_exact', 'unsupervised', 'supervised_only', 'mixed', 'partial_family',
                 'supervised_diag', 'capacity16', 'transition095']:
        path = CKPT / f'{name}.pt'
        if not path.exists(): continue
        model, cfg = model_from_checkpoint(path)
        for p, t in DECODER_GRID:
            _, met = exact_metrics(score_sequences(model, val, t, p)[0])
            records.append({'model': name, 'p_stay': p, 'temperature': t, 'split': 'val', **met})
        ev = evaluate_model(model, val, test, cfg['p_stay'])
        z, cz, nll = ev['raw'], ev['exact'], ev['raw_nll']
        raw_exact, _ = exact_metrics(z)
        ci = work_bootstrap(raw_exact, bacc_df, n=800)
        fam, _ = family_metrics(ev['decoded'])
        family_auc_ci = work_bootstrap(fam, lambda d: M.auc(d.family_y.to_numpy(), d.family_score.to_numpy()), n=800)
        with torch.no_grad():
            _, var, u = model.distribution_parameters(torch.zeros(1))
            covdiag = var.mean().item(); factor = float(u.square().mean().sqrt()) if cfg['rank'] else 0.
        calmet, fammet = ev['decoded_metrics'], ev['family_metrics']
        transition_rate = ev['transitions_per_100']
        result = {'model': name, 'rank': cfg['rank'], 'hidden': cfg['hidden'], 'p_stay_raw': cfg['p_stay'], 'raw_temperature': 1.0,
                  'test_marginal_nll_per_interval': nll, 'raw_metrics': ev['raw_metrics'], 'val_selected_p_stay': ev['p_stay'],
                  'val_selected_temperature': ev['temperature'], 'calibrated_metrics': calmet,
                  'calibrated_work_bootstrap_bacc_95ci': ci, 'family_metrics': fammet,
                  'family_auc_work_bootstrap_95ci': family_auc_ci, 'predicted_transitions_per_100_intervals': transition_rate,
                  'mean_diag_variance': covdiag, 'mean_factor_rms': factor,
                  'test_works': raw_exact.work_id.nunique(), 'test_performances': cz.performance_id.nunique()}
        (RES / f'{name}_evaluation.json').write_text(json.dumps(result, indent=2))
        cz.to_csv(RES / f'{name}_test_predictions.csv', index=False)
        records.append({'model': name, 'p_stay': ev['p_stay'], 'temperature': ev['temperature'], 'split': 'test',
                        'n': calmet['n'], 'balanced_accuracy': calmet['balanced_accuracy'], 'accuracy': calmet['accuracy'],
                        'ece': calmet['ece'], 'brier': calmet['brier'], 'marginal_nll': nll, 'family_auc': fammet['auc'],
                        'family_balanced_accuracy': fammet['balanced_accuracy'], 'test_work_bootstrap_bacc_lo': ci[0],
                        'test_work_bootstrap_bacc_hi': ci[1], 'family_auc_work_bootstrap_lo': family_auc_ci[0],
                        'family_auc_work_bootstrap_hi': family_auc_ci[1], 'pred_transitions_per_100': transition_rate})
        print(name, 'raw', ev['raw_metrics']['balanced_accuracy'], 'cal', calmet['balanced_accuracy'], 'NLL', nll,
              'family AUC', fammet['auc'], flush=True)
    # Direct matched classical HMM recomputed with uniform initial prior, p_stay=.99, kappa=1.
    base, base_nll = classical_direct(test); base = attach_old_family(base); _, bm = exact_metrics(base)
    _, bfm = family_metrics(base)
    base.to_csv(RES / 'classical_gaussian_direct_predictions.csv', index=False)
    records.append({'model': 'classical_gaussian_uniform_init_raw', 'p_stay': .99, 'temperature': 1., 'split': 'test', 'n': bm['n'],
                    'balanced_accuracy': bm['balanced_accuracy'], 'accuracy': bm['accuracy'], 'ece': bm['ece'], 'brier': bm['brier'],
                    'marginal_nll': base_nll, 'family_auc': bfm['auc'], 'family_balanced_accuracy': bfm['balanced_accuracy']})
    pd.DataFrame(records).to_csv(RES / 'matched_metrics.csv', index=False)
    # Preserve the official baseline separately (train prior + validation-selected temperature).
    official = {'source': 'mel50', 'model': 'gauss', 'decoder': 'hmm', 'balanced_accuracy': 0.6154344414248734,
                'accuracy': 0.6889123692014043, 'kappa': 0.5, 'p_stay': 0.99, 'test_n': 29339,
                'note': 'Original matched classical HMM, retained unchanged in subdivision_hmm/results/experiment/results.csv'}
    (RES / 'official_classical_baseline.json').write_text(json.dumps(official, indent=2))
    print('classical direct', bm['balanced_accuracy'], 'NLL', base_nll, 'family AUC', bfm['auc'])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Beat-conditioned subdivision HMM on ASAP with ground-truth beats.

Pipeline (per onset source x observation model x vocabulary):
  1. observation parameters: unsupervised, maximise the uniform-mixture marginal likelihood of the
     TRAIN intervals (no labels used);
  2. decoder hyper-parameters (calibration exponent kappa, self-transition p_stay): chosen on VAL
     by balanced accuracy;
  3. report TEST (work-disjoint). Test labels are only used for scoring.
Outputs results/experiment/*.csv and per-interval predictions for the main configurations.
"""
import argparse
import csv
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from subdivision_hmm import metrics as M
from subdivision_hmm.dataset import Evidence, interval_inputs, load_sequences
from subdivision_hmm.hmm import forward_backward, persistence_log_transition, viterbi
from subdivision_hmm.observation import GaussianTemplate, NaiveSum, PoissonOccupancy, mixture_loglik

HERE = Path(__file__).resolve().parents[1]

OUT = HERE / "results/experiment"
KAPPAS = (0.05, 0.1, 0.25, 0.5, 1.0)
P_STAYS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
POISSON_GRID = dict(q=(0.3, 0.5, 0.7, 0.9), lam0=(0.5, 1, 2, 4, 8), sigma=(0.02, 0.03, 0.05))
GAUSS_GRID = dict(tm2=(1, 4, 16), ta2=(1, 4, 16), width=(0.02, 0.03, 0.05))


def obs_matrix(model_name, params, ratios, ons, ys):
    if model_name == "poisson":
        m = PoissonOccupancy(ratios, **params)
        return np.stack([m.loglik(t, b0, b1) for t, b0, b1 in ons])
    if model_name == "gauss":
        return GaussianTemplate(ratios, **params).loglik_vectors(ys)
    return NaiveSum(ratios).loglik_vectors(ys)


def fit_params(model_name, ratios, ons, ys, rng):
    if model_name == "naive":
        return {}, float("nan")
    grid = POISSON_GRID if model_name == "poisson" else GAUSS_GRID
    idx = rng.choice(len(ons), size=min(6000, len(ons)), replace=False)
    sub_on, sub_y = [ons[i] for i in idx], ys[idx]
    best, best_ll = None, -np.inf
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals))
        ll = mixture_loglik(obs_matrix(model_name, p, ratios, sub_on, sub_y))
        if ll > best_ll:
            best, best_ll = p, ll
    return best, best_ll


def decode(seqs, ll, ratios, log_prior, kappa, p_stay, mode):
    k = len(ratios)
    preds, posts, i = [], [], 0
    lt = persistence_log_transition(k, p_stay if mode == "hmm" else 1.0 / k, dtype=torch.float64).numpy()
    for s in seqs:
        lo = kappa * ll[i:i + len(s)]
        i += len(s)
        if mode == "independent":  # class prior times evidence, no temporal coupling
            post = softmax(lo + log_prior, axis=1)
            path = post.argmax(1)
        else:
            init = log_prior if mode == "hmm" else np.full(k, -np.log(k))
            post, _ = forward_backward(init, lt, lo)
            path = viterbi(init, lt, lo)
        preds.append(path); posts.append(post)
    return np.concatenate(preds), np.concatenate(posts)


def flat(seqs, key):
    return np.array([r[key] for s in seqs for r in s])


def metrics(labels, pred_r, post_max, ratios, mask):
    """Accuracy on labelled in-vocabulary intervals."""
    y, p = labels[mask], pred_r[mask]
    out = {"n": int(mask.sum())}
    out.update({f"acc_{r}": v for r, v in M.per_class_recall(y, p, ratios).items()})
    out["accuracy"] = float(np.mean(p == y)) if len(y) else float("nan")
    out["balanced_accuracy"] = M.balanced_accuracy(y, p, ratios)
    for fam, members in (("binary", (2, 4)), ("ternary", (3, 6))):
        m = np.isin(y, members)
        out[f"acc_{fam}"] = float(np.mean(p[m] == y[m])) if m.sum() else float("nan")
        out[f"family_acc_{fam}"] = float(np.mean(np.isin(p[m], members))) if m.sum() else float("nan")
    out["lenient_contains_notated"] = float(np.mean(p % y == 0)) if len(y) else float("nan")
    conf = post_max[mask]
    out["mean_confidence"] = float(conf.mean()) if len(conf) else float("nan")
    out["ece"] = M.ece(conf, p == y) if len(conf) else float("nan")
    return out


def transitions(seqs, pred_r, labels):
    i, pt, rt, n = 0, 0, 0, 0
    for s in seqs:
        p, l = pred_r[i:i + len(s)], labels[i:i + len(s)]
        i += len(s)
        pt += int(np.sum(np.diff(p) != 0))
        known = l[l > 0]
        rt += int(np.sum(np.diff(known) != 0))
        n += len(s)
    return {"pred_transitions_per_100_beats": 100 * pt / n, "ref_transitions_per_100_beats": 100 * rt / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="audio100,mel50,perfmidi")
    ap.add_argument("--models", default="poisson,gauss,naive")
    ap.add_argument("--vocabs", default="1-2-3-4-6,1-2-3-4,2-3-4,1-2-3-4-6-8-12")
    ap.add_argument("--out-name", default="results.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = load_sequences()
    by_split = {s: [q for q in seqs if q[0]["split"] == s] for s in ("train", "val", "test")}
    print({k: (len(v), sum(map(len, v))) for k, v in by_split.items()}, flush=True)
    combos = [(src, m, v) for src in a.sources.split(",") for m in a.models.split(",") for v in a.vocabs.split(",")]
    combos.sort(key=lambda c: c[2] != "1-2-3-4-6")  # main vocabulary first
    rows = []
    with Pool(min(len(combos), 40)) as pool:
        for part in pool.imap_unordered(run_combo, [(c, by_split) for c in combos]):
            rows.extend(part)

            cols = list(dict.fromkeys(k for r in rows for k in r))
            with (OUT / a.out_name).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)


def run_combo(args):
    (src, model_name, vocab), by_split = args
    rng = np.random.default_rng(0)
    ev = Evidence()
    rows, sel_rows = [], []
    inputs = {s: interval_inputs(by_split[s], ev, src, raw=model_name == "naive") for s in by_split}
    ratios = [int(x) for x in vocab.split("-")]
    lab = {s: flat(by_split[s], "label") for s in by_split}
    ok = {s: inputs[s][2] for s in by_split}
    train_known = lab["train"][np.isin(lab["train"], ratios)]
    prior = np.array([np.mean(train_known == r) for r in ratios]) + 1e-3
    log_prior = np.log(prior / prior.sum())
    params, fit_ll = fit_params(model_name, ratios, inputs["train"][0], inputs["train"][1], rng)
    ll = {s: obs_matrix(model_name, params, ratios, inputs[s][0], inputs[s][1]) for s in ("val", "test")}
    ra = np.array(ratios)
    # candidate configurations
    configs = [("prior_constant", None, None), ("hmm_uniform", 1.0, None)]
    configs += [("independent", k, None) for k in KAPPAS]
    configs += [("hmm_uniform", k, None) for k in KAPPAS]
    configs += [("hmm", k, p) for k in KAPPAS for p in P_STAYS]
    results = {}
    for split in ("val", "test"):
        for mode, kappa, p in configs:
            if mode == "prior_constant":
                pr = np.full(len(lab[split]), ratios[int(np.argmax(prior))]); pm = np.full(len(pr), prior.max() / prior.sum())
            else:
                idx, post = decode(by_split[split], ll[split], ratios, log_prior, kappa, p, mode)
                pr, pm = ra[idx], post.max(1)
            results[(split, mode, kappa, p)] = (pr, pm)
    # choose kappa / p on val per decoder family
    for mode in ("prior_constant", "independent", "hmm_uniform", "hmm"):
        cands = [(m, k, p) for m, k, p in configs if m == mode]
        vmask = np.isin(lab["val"], ratios) & ok["val"]
        best = max(cands, key=lambda c: metrics(lab["val"], results[("val",) + c][0], results[("val",) + c][1], ratios, vmask)["balanced_accuracy"])
        for split in ("val", "test"):
            pr, pm = results[(split,) + best]
            mask = np.isin(lab[split], ratios) & ok[split]
            hc = mask & flat(by_split[split], "high_confidence").astype(bool)
            for subset, m in (("all_labelled", mask), ("high_confidence", hc)):
                row = {"source": src, "model": model_name, "vocab": vocab, "decoder": mode, "split": split,
                       "subset": subset, "kappa": best[1], "p_stay": best[2], "params": json.dumps(params),
                       "coverage_in_vocab": float(np.mean(np.isin(lab[split][lab[split] > 0], ratios))),
                       **metrics(lab[split], pr, pm, ratios, m), **transitions(by_split[split], pr, lab[split])}
                rows.append(row)
        if split == "test":
            sel_rows.append((src, model_name, vocab, mode, best, results[("test",) + best], ll["test"]))
    # transition-strength ablation on test at the val-selected kappa of the hmm family
    kbest = [r for r in rows if r["source"] == src and r["model"] == model_name and r["vocab"] == vocab and r["decoder"] == "hmm"][0]["kappa"]
    tmask = np.isin(lab["test"], ratios) & ok["test"]
    for p in (None,) + P_STAYS:
        key = ("test", "hmm_uniform", kbest, None) if p is None else ("test", "hmm", kbest, p)
        pr, pm = results[key]
        rows.append({"source": src, "model": model_name, "vocab": vocab, "decoder": "ablation_p_stay", "split": "test",
                     "subset": "all_labelled", "kappa": kbest, "p_stay": p if p else 1 / len(ratios), "params": json.dumps(params),
                     **metrics(lab["test"], pr, pm, ratios, tmask), **transitions(by_split["test"], pr, lab["test"])})
    print(src, model_name, vocab, params, flush=True)
    if vocab == "1-2-3-4-6":
        save_predictions(src, model_name, by_split["test"], sel_rows, ratios)
    sel_rows = []
    return rows



def save_predictions(src, model_name, seqs, sel_rows, ratios):
    flat_rows = [r for s in seqs for r in s]
    for _, _, _, mode, best, (pr, pm), ll in sel_rows:
        if mode not in ("independent", "hmm", "prior_constant"):
            continue
        with (OUT / f"pred_{src}_{model_name}_{mode}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["performance_id", "work_id", "composer", "beat_index", "label", "high_confidence", "beat_label",
                        "pred", "post_max", "kappa", "p_stay"] + [f"ll_{r}" for r in ratios])
            for r, p, c, l in zip(flat_rows, pr, pm, ll):
                w.writerow([r["performance_id"], r["work_id"], r["composer"], r["beat_index"], r["label"], int(r["high_confidence"]),
                            r["beat_label"], int(p), float(c), best[1], best[2]] + [f"{v:.4f}" for v in l])


if __name__ == "__main__":
    main()

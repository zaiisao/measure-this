"""Shared machinery for the NHMM training-data scaling study.

Reuses, unchanged: the NeuralGaussianEmission model, the exact HMM recursions, the
decomposed objective of diagnostics/ (lambda-weighted posterior-state term), the
original train-pool selection (supervised_only) and the original evaluator functions.

New here: work-level subset samplers, an explicit optimizer-update budget, dense
learning curves, and optional loss-weighting variants (Experiments D/E). Every
weighting other than 'interval' changes the training objective; it is then no
longer the original weighted-likelihood objective (see `objective_note`).
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NHMM = HERE.parent
sys.path.insert(0, str(NHMM))
from data import RATIOS, K, constraint_for, load_sequences  # noqa: E402
from diagnostics.objective import SequenceTerms, aggregate_terms, sequence_terms  # noqa: E402
from diagnostics.weighted_train import select  # noqa: E402
from model import NeuralGaussianEmission, exclude_missing, persistence_log_transition  # noqa: E402

RUNS = HERE / "runs"
ORIGINAL_NESTED_SEED = 20261031  # diagnostics/weighted_train.py --limit-sequences


# ---------------------------------------------------------------- data / pools
_DATA = None


def data():
    """(splits, normalization, train_pool, val_labeled), loaded once per process."""
    global _DATA
    if _DATA is None:
        splits, norm = load_sequences()
        pool, _ = select(splits["train"], "supervised_only")
        val = [q for q in splits["val"] if q.has_audio and np.isin(q.labels, RATIOS).any()]
        _DATA = (splits, norm, pool, val)
    return _DATA


def pool_by_work(pool):
    out = {q.work_id: q for q in pool}
    if len(out) != len(pool):
        raise ValueError("expected one training sequence per work")
    return out


def original_nested_order(pool):
    """Work order of diagnostics/weighted_train.py --limit-sequences (exact five first)."""
    exact = [q for q in pool if q.fully_exact]
    extras = [q for q in pool if not q.fully_exact]
    rng = np.random.default_rng(ORIGINAL_NESTED_SEED)
    extras = [extras[i] for i in rng.permutation(len(extras))]
    return [q.work_id for q in exact + extras]


def work_stats(q):
    lab = q.labels
    exact = np.isin(lab, RATIOS)
    dur = np.exp(q.log_duration)  # normalized log-duration; relative only
    return {
        "work_id": q.work_id, "composer": q.work_id.split("::")[0], "intervals": int(q.n),
        **{f"n_{r}": int((lab == r).sum()) for r in RATIOS},
        "n_family_only": int(np.isin(lab, (8, 12)).sum()),
        "n_unknown": int((lab == 0).sum()), "n_oov": int(np.isin(lab, (16, 24)).sum()),
        "hc_exact_frac": float(q.high_confidence[exact].mean()) if exact.any() else float("nan"),
        "log_duration_z_mean": float(q.log_duration.mean()),
        "feature_rms": float(np.sqrt(np.mean(q.y[q.observed] ** 2))),
        "fully_exact": bool(q.fully_exact),
    }


# ---------------------------------------------------------------- samplers
def sample_uniform(work_ids, n, seed):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(sorted(work_ids), size=n, replace=False).tolist())


def nested_order(work_ids, seed):
    rng = np.random.default_rng(seed)
    w = sorted(work_ids)
    return [w[i] for i in rng.permutation(len(w))]


def sample_stratified(stats, n, seed, key):
    """Proportional allocation over strata of `key(stat_row)`, then uniform within strata."""
    rng = np.random.default_rng(seed)
    strata = {}
    for s in sorted(stats, key=lambda s: s["work_id"]):
        strata.setdefault(key(s), []).append(s["work_id"])
    total = sum(len(v) for v in strata.values())
    quota = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: int(math.floor(q)) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (quota[k] - alloc[k], str(k)), reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    out = []
    for k, v in sorted(strata.items(), key=lambda kv: str(kv[0])):
        out += rng.choice(v, size=min(alloc[k], len(v)), replace=False).tolist()
    return sorted(out)


def sample_class_balanced(stats, n, seed, tries=4000):
    """Among random n-work draws (training statistics only), keep the one whose exact-label
    class proportions are closest to uniform over the five states."""
    rng = np.random.default_rng(seed)
    ids = sorted(s["work_id"] for s in stats)
    by = {s["work_id"]: s for s in stats}
    best, best_d = None, float("inf")
    for _ in range(tries):
        pick = rng.choice(ids, size=n, replace=False)
        c = np.array([sum(by[w][f"n_{r}"] for w in pick) for r in RATIOS], float)
        d = float(np.abs(c / c.sum() - 1 / len(RATIOS)).sum())
        if d < best_d:
            best, best_d = pick, d
    return sorted(best.tolist())


# ---------------------------------------------------------------- objective
def _log_alpha_beta(log_init, log_trans, em):
    t_len = em.shape[0]
    alphas = [log_init + em[0]]
    for t in range(1, t_len):
        alphas.append(torch.logsumexp(alphas[-1][:, None] + log_trans, 0) + em[t])
    betas = [torch.zeros_like(em[0])]
    for t in range(t_len - 2, -1, -1):
        betas.append(torch.logsumexp(log_trans + (em[t + 1] + betas[-1])[None, :], 1))
    return torch.stack(alphas), torch.stack(betas[::-1])


def marginal_state_nll(log_init, log_trans, emission, observed, allowed, weights):
    """sum_i w_i * -log p(R_i in C_i | O, B), exact marginals of the same HMM (differentiable)."""
    em = exclude_missing(emission, observed)
    a, b = _log_alpha_beta(log_init, log_trans, em)
    logz = torch.logsumexp(a[-1], 0)
    lpost = a + b - logz
    inf = ~allowed.all(1)
    if not bool(inf.any()):
        return lpost.sum() * 0.0, 0.0
    lp_c = torch.logsumexp(lpost.masked_fill(~allowed, -1e30), 1)
    w = weights[inf]
    return -(w * lp_c[inf]).sum(), float(w.sum())


@dataclass
class Weighting:
    scheme: str = "interval"          # interval | work | class_marginal | marginal | confidence_marginal
    class_weights: Optional[dict] = None   # {ratio: w} for class_marginal
    low_conf_weight: float = 1.0           # for confidence_marginal
    work_weights: Optional[dict] = None    # optional per-work multipliers (difficult-work down-weighting)


def objective_note(w: Weighting):
    if w.scheme == "interval":
        return "original: acoustic NLL/observed + lambda * (-log p(C|O,B))/observed (diagnostics.objective)"
    if w.scheme == "work":
        return ("reweighted (not ML): mean over works of [acoustic NLL_w/obs_w + lambda * state NLL_w/obs_w]; "
                "acoustic density unchanged")
    return ("reweighted auxiliary (not ML): acoustic NLL/observed + lambda * weighted sum of per-beat exact posterior "
            "marginal NLLs -log p(R_i in C_i|O,B), normalized to the original state-term scale")


def batch_loss(model, seqs, log_init, log_trans, lam, weighting: Weighting, device, reg_weight=1e-4,
               high_conf=None):
    y = torch.from_numpy(np.concatenate([q.y for q in seqs])).to(device)
    d = torch.from_numpy(np.concatenate([q.log_duration for q in seqs])).to(device)
    obs = torch.from_numpy(np.concatenate([q.observed for q in seqs])).to(device)
    emission = model.log_prob(y, d)
    start, terms, extra = 0, [], []
    for q in seqs:
        n = q.n
        allowed = torch.as_tensor(constraint_for(q, "exact"), dtype=torch.bool, device=device)
        sl = slice(start, start + n)
        terms.append(sequence_terms(log_init, log_trans, emission[sl], allowed, obs[sl]))
        if weighting.scheme in ("class_marginal", "marginal", "confidence_marginal"):
            w = torch.ones(n, device=device)
            if weighting.scheme == "class_marginal":
                w = torch.tensor([weighting.class_weights.get(int(l), 0.0) for l in q.labels], device=device,
                                 dtype=torch.float32)
            if weighting.scheme == "confidence_marginal":
                w = torch.where(torch.from_numpy(q.high_confidence).to(device), 1.0, weighting.low_conf_weight)
            if weighting.work_weights:
                w = w * float(weighting.work_weights.get(q.work_id, 1.0))
            extra.append(marginal_state_nll(log_init, log_trans, emission[sl], obs[sl], allowed, w))
        start += n
    base = aggregate_terms(terms, lambda_state=lam)
    n_obs = max(base.observed_intervals, 1)
    if weighting.scheme == "interval" and not weighting.work_weights:
        loss = base.loss
    elif weighting.scheme in ("interval", "work"):
        ww = [float((weighting.work_weights or {}).get(q.work_id, 1.0)) for q in seqs]
        if weighting.scheme == "interval":   # interval objective with per-work multipliers
            num = sum(w * (t.acoustic_nll + lam * t.state_nll) for w, t in zip(ww, terms))
            loss = num / max(sum(w * t.observed_intervals for w, t in zip(ww, terms)), 1)
        else:
            per = [w * (t.acoustic_nll + lam * t.state_nll) / max(t.observed_intervals, 1) for w, t in zip(ww, terms)]
            loss = torch.stack(per).sum() / sum(ww)
    else:
        num = sum(x[0] for x in extra)
        wsum = sum(x[1] for x in extra)
        # same units as the original state term: per-label NLL * informative/observed
        state = (num / max(wsum, 1e-9)) * (base.informative_intervals / n_obs) if wsum else num * 0.0
        loss = base.acoustic_nll_per_observed + lam * state
    if bool(obs.any()) and reg_weight:
        _, var, u = model.distribution_parameters(d[obs])
        reg = var.log().square().mean()
        if model.rank:
            reg = reg + 0.05 * u.square().mean()
        loss = loss + reg_weight * reg
    return loss, base


# ---------------------------------------------------------------- evaluation helpers
@torch.no_grad()
def val_objective(model, val, log_init, log_trans, lam, device):
    """Original checkpoint criterion: weighted objective (lambda) on validation, interval-weighted."""
    model.eval()
    terms = []
    for i in range(0, len(val), 4):
        _, t = batch_loss(model, val[i:i + 4], log_init, log_trans, lam, Weighting(), device, reg_weight=0)
        terms.append(SequenceTerms(t.acoustic_nll_sum, t.joint_nll_sum, t.state_nll_sum,
                                   t.observed_intervals, t.informative_intervals))
    agg = aggregate_terms(terms, lam)
    return float(agg.loss), float(agg.acoustic_nll_per_observed), float(agg.state_nll_per_informative)


@torch.no_grad()
def raw_val_metrics(model, val, p_stay=0.99):
    from evaluate import score_sequences
    df, _ = score_sequences(model, val, 1.0, p_stay)
    z = df[df.observed & df.label.isin(RATIOS)]
    rec = {str(r): float((z.loc[z.label == r, "pred"] == r).mean()) for r in RATIOS}
    return float(np.mean(list(rec.values()))), rec


# ---------------------------------------------------------------- training
@dataclass
class RunSpec:
    name: str
    work_ids: list
    seed: int = 0
    lam: float = 10.0
    budget: str = "epochs"            # epochs | steps
    epochs: int = 10
    patience: int = 3
    steps: int = 1200
    eval_every: int = 40
    lr: float = 8e-4
    batch_sequences: int = 4
    weighting: dict = field(default_factory=dict)
    init_from: Optional[str] = None   # checkpoint path (F2 continuation)
    tags: dict = field(default_factory=dict)


def train(spec: RunSpec, device="cpu"):
    out = RUNS / spec.name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(spec.seed); np.random.seed(spec.seed); torch.manual_seed(spec.seed)
    torch.set_num_threads(2)
    splits, norm, pool, val = data()
    by = pool_by_work(pool)
    missing = [w for w in spec.work_ids if w not in by]
    if missing:
        raise KeyError(f"work ids not in training pool: {missing[:3]}")
    train_seqs = [by[w] for w in spec.work_ids]
    weighting = Weighting(**spec.weighting)
    model = NeuralGaussianEmission(ratios=RATIOS, m=norm["feature_dim"], hidden=48, rank=2).to(device)
    if spec.init_from:
        model.load_state_dict(torch.load(spec.init_from, map_location=device, weights_only=False)["model"])
    log_init = torch.full((K,), -np.log(K), device=device)
    log_trans = persistence_log_transition(K, 0.99, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=1e-5)
    hist, steps, t0 = [], 0, time.time()
    best = {"val_objective": (float("inf"), 0), "val_bacc": (-1.0, 0)}
    rng = np.random.default_rng(spec.seed)  # recorded sampling schedule: epoch-wise permutations
    schedule = []

    def evaluate_and_maybe_save(epoch):
        obj, vac, vst = val_objective(model, val, log_init, log_trans, spec.lam, device)
        vb, rec = raw_val_metrics(model, val)
        row = {"step": steps, "epoch": epoch, "val_objective": obj, "val_acoustic_nll": vac,
               "val_state_nll_per_label": vst, "val_raw_bacc": vb, **{f"val_recall_{k}": v for k, v in rec.items()},
               "elapsed_s": time.time() - t0}
        for key, val_ in (("val_objective", obj), ("val_bacc", vb)):
            better = val_ < best[key][0] - 1e-5 if key == "val_objective" else val_ > best[key][0] + 1e-9
            if better:
                best[key] = (val_, steps)
                torch.save({"model": model.state_dict(), "config": {"mode": "supervised_only", "rank": 2, "hidden": 48,
                            "p_stay": 0.99, "seed": spec.seed, "lambda_state": spec.lam},
                            "duration_normalization": norm, "step": steps}, out / f"best_{key}.pt")
        return row

    def one_step(batch, train_acc):
        nonlocal steps
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, base = batch_loss(model, batch, log_init, log_trans, spec.lam, weighting, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in {spec.name} at step {steps}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        steps += 1
        train_acc["acoustic"] += float(base.acoustic_nll_sum); train_acc["state"] += float(base.state_nll_sum)
        train_acc["obs"] += base.observed_intervals; train_acc["inf"] += base.informative_intervals

    def train_row(acc):
        return {"train_acoustic_nll": acc["acoustic"] / max(acc["obs"], 1),
                "train_state_nll_per_label": acc["state"] / max(acc["inf"], 1)}

    if spec.budget == "epochs":
        wait = 0
        for epoch in range(1, spec.epochs + 1):
            # global numpy stream, exactly as diagnostics/weighted_train.py (bitwise reproduction)
            order = np.random.permutation(len(train_seqs)); schedule.append(order.tolist())
            acc = {"acoustic": 0.0, "state": 0.0, "obs": 0, "inf": 0}
            for i in range(0, len(order), spec.batch_sequences):
                one_step([train_seqs[j] for j in order[i:i + spec.batch_sequences]], acc)
            prev = best["val_objective"][0]
            hist.append({**evaluate_and_maybe_save(epoch), **train_row(acc)})
            if best["val_objective"][0] < prev - 1e-5:
                wait = 0
            else:
                wait += 1
                if wait >= spec.patience:
                    break
    else:
        acc = {"acoustic": 0.0, "state": 0.0, "obs": 0, "inf": 0}
        epoch, queue = 0, []
        while steps < spec.steps:
            while len(queue) < spec.batch_sequences:  # identical for >= batch_sequences works
                order = rng.permutation(len(train_seqs)).tolist(); schedule.append(order); epoch += 1
                queue += order
            batch_idx, queue = queue[:spec.batch_sequences], queue[spec.batch_sequences:]
            one_step([train_seqs[j] for j in batch_idx], acc)
            if steps % spec.eval_every == 0 or steps == spec.steps:
                hist.append({**evaluate_and_maybe_save(steps * spec.batch_sequences / len(train_seqs)), **train_row(acc)})
                acc = {"acoustic": 0.0, "state": 0.0, "obs": 0, "inf": 0}
    torch.save({"model": model.state_dict(), "config": {"mode": "supervised_only", "rank": 2, "hidden": 48,
                "p_stay": 0.99, "seed": spec.seed, "lambda_state": spec.lam},
                "duration_normalization": norm, "step": steps}, out / "final.pt")
    meta = {"spec": asdict(spec), "objective": objective_note(weighting), "optimizer_steps": steps,
            "epochs_equivalent": steps * spec.batch_sequences / len(train_seqs),
            "best_val_objective": {"value": best["val_objective"][0], "step": best["val_objective"][1]},
            "best_val_bacc": {"value": best["val_bacc"][0], "step": best["val_bacc"][1]},
            "train_intervals": int(sum(q.n for q in train_seqs)), "history": hist,
            "sampling_schedule_permutations": schedule, "wall_seconds": time.time() - t0}
    (out / "training.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------- final evaluation
def evaluate_checkpoint(path, tag, out_dir):
    """Original protocol: decoder (p_stay x temperature) chosen on validation BAcc; raw and decoded test."""
    from evaluate import attach_old_family, exact_metrics, family_metrics, model_from_checkpoint, score_sequences
    splits, _, _, val = data()
    test = splits["test"]
    model, cfg = model_from_checkpoint(path)
    cands = []
    for p in (.95, .99, .999):
        for temp in (.25, .5, 1.0):
            pred, _ = score_sequences(model, val, temp, p)
            _, m = exact_metrics(pred)
            cands.append((m["balanced_accuracy"], -abs(temp - 1), -abs(p - .99), p, temp))
    vbest, _, _, pb, tb = max(cands)
    res = {"tag": tag, "val_selected_p_stay": pb, "val_selected_temperature": tb, "val_decoded_bacc": vbest}
    raw_val, val_nll = score_sequences(model, val, 1.0, .99)
    _, rv = exact_metrics(raw_val)
    res["val_raw"] = summarize(rv); res["val_marginal_nll"] = val_nll
    raw, nll = score_sequences(model, test, 1.0, .99)
    dec, _ = score_sequences(model, test, tb, pb)
    _, rm = exact_metrics(raw)
    ex, dm = exact_metrics(dec)
    res["test_raw"] = summarize(rm); res["test_decoded"] = summarize(dm); res["test_marginal_nll"] = nll
    raw_l = raw[raw.observed & raw.label.isin(RATIOS)]
    res["test_raw_posterior_state_nll"] = float(-np.mean(np.log(np.clip(
        raw_l.apply(lambda r: r[f"post_{int(r.label)}"], axis=1).to_numpy(), 1e-12, None))))
    dec = attach_old_family(dec)
    vdec, _ = score_sequences(model, val, tb, pb)
    _, fm = family_metrics(dec, attach_old_family(vdec))
    res["test_family_auc"] = fm["auc"]; res["test_family_bacc"] = fm["balanced_accuracy"]
    adjacent = trans = 0
    for _, g in dec.groupby("performance_id"):
        g = g.sort_values("beat_index"); same = np.diff(g.beat_index.to_numpy()) == 1
        adjacent += int(same.sum()); trans += int((g.pred.to_numpy()[1:][same] != g.pred.to_numpy()[:-1][same]).sum())
    res["test_transitions_per_100"] = 100 * trans / max(adjacent, 1)
    res["test_pred_distribution"] = {str(r): float((ex.pred == r).mean()) for r in RATIOS}
    res["test_per_work_bacc"] = {w: float(np.mean([(g.loc[g.label == r, "pred"] == r).mean()
                                                   for r in RATIOS if (g.label == r).any()]))
                                 for w, g in ex.groupby("work_id")}
    (Path(out_dir) / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


def summarize(m):
    rec = m["per_class_recall"]
    return {"bacc": m["balanced_accuracy"], "accuracy": m["accuracy"], "recall": rec,
            "pair_2_4": (rec["2"] + rec["4"]) / 2, "pair_3_6": (rec["3"] + rec["6"]) / 2,
            "ece": m["ece"], "brier": m["brier"], "confusion": m["confusion"]}

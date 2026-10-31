"""Shared machinery for the NHMM training-data scaling study.

Reuses, unchanged: the NeuralGaussianEmission model, the exact HMM recursions, the
lambda-weighted objective and weighting variants (diagnostics/objective.py), the shared
training loop (training.py), the original train-pool selection and the shared evaluator.

New here: work-level subset samplers and the run specification. Every weighting other
than 'interval' changes the training objective; it is then no longer the original
weighted-likelihood objective (see `objective_note`).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from neural_subdivision_hmm.data import RATIOS, K, choose_sequences, load_sequences, validation_sequences
from neural_subdivision_hmm.diagnostics.objective import Weighting, batch_objective, evaluate_objective, objective_note
from neural_subdivision_hmm.evaluate import evaluate_model, exact_metrics, model_from_checkpoint, score_sequences
from neural_subdivision_hmm.model import NeuralGaussianEmission
from neural_subdivision_hmm.training import Budget, fit
from neural_subdivision_hmm.utils import save_checkpoint, seed_everything
from subdivision_hmm.hmm import persistence_log_transition

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ORIGINAL_NESTED_SEED = 20261031  # diagnostics/weighted_train.py --limit-sequences


# ---------------------------------------------------------------- data / pools
_DATA = None


def data():
    """(splits, normalization, train_pool, val_labeled), loaded once per process."""
    global _DATA
    if _DATA is None:
        splits, norm = load_sequences()
        _DATA = (splits, norm, choose_sequences(splits["train"], "supervised_only")[0],
                 validation_sequences(splits["val"], "supervised_only")[0])
    return _DATA


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


CRITERIA = (("val_objective", "min"), ("val_raw_bacc", "max"))
CHECKPOINT = {"val_objective": "best_val_objective.pt", "val_raw_bacc": "best_val_bacc.pt"}


def train(spec: RunSpec):
    out = RUNS / spec.name
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(spec.seed)
    splits, norm, pool, val = data()
    by = {q.work_id: q for q in pool}
    missing = [w for w in spec.work_ids if w not in by]
    if missing:
        raise KeyError(f"work ids not in training pool: {missing[:3]}")
    train_seqs = [by[w] for w in spec.work_ids]
    weighting = Weighting(**spec.weighting)
    model = NeuralGaussianEmission(ratios=RATIOS, m=norm["feature_dim"], hidden=48, rank=2)
    if spec.init_from:
        model.load_state_dict(torch.load(spec.init_from, map_location="cpu", weights_only=False)["model"])
    log_init = torch.full((K,), -np.log(K))
    log_trans = persistence_log_transition(K, 0.99)
    config = {"mode": "supervised_only", "rank": 2, "hidden": 48, "p_stay": 0.99, "seed": spec.seed, "lambda_state": spec.lam}
    t0 = time.time()

    def loss(m, batch):
        value, base = batch_objective(m, batch, "exact", log_init, log_trans, spec.lam, weighting)
        return value, {"acoustic": float(base.acoustic_nll_sum), "state": float(base.state_nll_sum),
                       "obs": base.observed_intervals, "inf": base.informative_intervals}

    def evaluate(m):
        t = evaluate_objective(m, val, "exact", log_init, log_trans, spec.lam)
        _, met = exact_metrics(score_sequences(m, val, 1.0, 0.99)[0])
        return {"val_objective": float(t.loss), "val_acoustic_nll": float(t.acoustic_nll_per_observed),
                "val_state_nll_per_label": float(t.state_nll_per_informative), "val_raw_bacc": met["balanced_accuracy"],
                **{f"val_recall_{k}": v for k, v in met["per_class_recall"].items()}, "elapsed_s": time.time() - t0}

    result = fit(model, train_seqs, loss, evaluate,
                 Budget(spec.budget, spec.epochs, spec.patience, spec.steps, spec.eval_every, spec.batch_sequences),
                 criteria=CRITERIA, lr=spec.lr, rng=np.random.default_rng(spec.seed),
                 train_summary=lambda s: {"train_acoustic_nll": s["acoustic"] / max(s["obs"], 1),
                                          "train_state_nll_per_label": s["state"] / max(s["inf"], 1)},
                 on_best=lambda k, step: save_checkpoint(out / CHECKPOINT[k], model, config, norm, step=step))
    save_checkpoint(out / "final.pt", model, config, norm, step=result.steps)
    meta = {"spec": asdict(spec), "objective": objective_note(weighting), "optimizer_steps": result.steps,
            "epochs_equivalent": result.steps * spec.batch_sequences / len(train_seqs),
            "best_val_objective": dict(zip(("value", "step"), result.best["val_objective"])),
            "best_val_bacc": dict(zip(("value", "step"), result.best["val_raw_bacc"])),
            "train_intervals": int(sum(q.n for q in train_seqs)), "history": result.history,
            "sampling_schedule_permutations": result.schedule, "wall_seconds": time.time() - t0}
    (out / "training.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------- final evaluation
def summarize(m):
    rec = m["per_class_recall"]
    return {"bacc": m["balanced_accuracy"], "accuracy": m["accuracy"], "recall": rec,
            "pair_2_4": (rec["2"] + rec["4"]) / 2, "pair_3_6": (rec["3"] + rec["6"]) / 2,
            "ece": m["ece"], "brier": m["brier"], "confusion": m["confusion"]}


def evaluate_checkpoint(path, tag, out_dir):
    """Original protocol: decoder (p_stay x temperature) chosen on validation BAcc; raw and decoded test."""
    splits, _, _, val = data()
    model, _ = model_from_checkpoint(path)
    ev = evaluate_model(model, val, splits["test"])
    raw_val, val_nll = score_sequences(model, val, 1.0, .99)
    raw_l = ev["raw"][ev["raw"].observed & ev["raw"].label.isin(RATIOS)]
    post_true = raw_l[[f"post_{r}" for r in RATIOS]].to_numpy()[np.arange(len(raw_l)), raw_l.label.map(RATIOS.index).to_numpy()]
    ex = ev["exact"]
    res = {"tag": tag, "val_selected_p_stay": ev["p_stay"], "val_selected_temperature": ev["temperature"],
           "val_decoded_bacc": ev["val_decoded_bacc"], "val_raw": summarize(exact_metrics(raw_val)[1]),
           "val_marginal_nll": val_nll, "test_raw": summarize(ev["raw_metrics"]),
           "test_decoded": summarize(ev["decoded_metrics"]), "test_marginal_nll": ev["raw_nll"],
           "test_raw_posterior_state_nll": float(-np.mean(np.log(np.clip(post_true, 1e-12, None)))),
           "test_family_auc": ev["family_metrics"]["auc"], "test_family_bacc": ev["family_metrics"]["balanced_accuracy"],
           "test_transitions_per_100": ev["transitions_per_100"],
           "test_pred_distribution": {str(r): float((ex.pred == r).mean()) for r in RATIOS},
           "test_per_work_bacc": {w: exact_metrics(g)[1]["balanced_accuracy"] for w, g in ex.groupby("work_id")}}
    (Path(out_dir) / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    return res

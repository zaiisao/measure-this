"""Exact decomposition of the constrained generative HMM NLL, and the training objectives built on it.

The state term is a posterior probability under the same generative HMM:
  -log p(C|O,B) = -log p(O,C|B) + log p(O|B).
It is not an independently parameterized classifier or CRF.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from neural_subdivision_hmm.data import constraint_for
from subdivision_hmm.hmm import exclude_missing, forward_log_prob, log_posteriors


@dataclass
class SequenceTerms:
    acoustic_nll: torch.Tensor
    joint_nll: torch.Tensor
    state_nll: torch.Tensor
    observed_intervals: int
    informative_intervals: int


@dataclass
class BatchTerms:
    loss: torch.Tensor
    acoustic_nll_per_observed: torch.Tensor
    state_nll_per_informative: torch.Tensor
    joint_nll_per_observed: torch.Tensor
    acoustic_nll_sum: torch.Tensor
    state_nll_sum: torch.Tensor
    joint_nll_sum: torch.Tensor
    observed_intervals: int
    informative_intervals: int
    lambda_state: float


def sequence_terms(log_init, log_transition, log_emission, allowed=None, observed=None) -> SequenceTerms:
    """Exact sequence acoustic, constrained-joint, and posterior-state NLLs.

    Missing acoustic rows are neutral emissions but remain in the state sequence and
    therefore retain their transitions. A position is informative when its allowed
    state set is a strict subset of all states, regardless of whether its audio exists.
    """
    if observed is None:
        observed = torch.ones(log_emission.shape[0], dtype=torch.bool, device=log_emission.device)
    emissions = exclude_missing(log_emission, observed)
    acoustic_logp = forward_log_prob(log_init, log_transition, emissions)
    joint_logp = forward_log_prob(log_init, log_transition, emissions, allowed)
    informative = 0 if allowed is None else int((~allowed.all(dim=1)).sum().item())
    return SequenceTerms(-acoustic_logp, -joint_logp, acoustic_logp - joint_logp,
                         int(observed.sum().item()), informative)


def aggregate_terms(terms: Sequence[SequenceTerms], lambda_state: float = 1.0) -> BatchTerms:
    """Normalize audio and state objectives explicitly.

    Acoustic NLL is divided by observed acoustic intervals. Posterior state NLL is
    reported per informative constrained interval. To make lambda_state=1 exactly
    reproduce the original constrained-joint NLL averaged per observed audio
    interval, its contribution is state_nll_per_informative *
    informative_intervals / observed_intervals. If there are no informative labels,
    the state term is exactly zero.
    """
    acoustic_sum = torch.stack([x.acoustic_nll for x in terms]).sum()
    state_sum = torch.stack([x.state_nll for x in terms]).sum()
    joint_sum = torch.stack([x.joint_nll for x in terms]).sum()
    n_observed = sum(x.observed_intervals for x in terms)
    n_informative = sum(x.informative_intervals for x in terms)
    acoustic_den = max(n_observed, 1)
    acoustic_mean = acoustic_sum / acoustic_den
    state_mean = state_sum / n_informative if n_informative else state_sum * 0.0
    # Equivalent to acoustic_mean + lambda * state_sum / n_observed.
    state_contribution = state_mean * (n_informative / acoustic_den) if n_informative else state_sum * 0.0
    loss = acoustic_mean + float(lambda_state) * state_contribution
    return BatchTerms(loss, acoustic_mean, state_mean, joint_sum / acoustic_den,
                      acoustic_sum, state_sum, joint_sum, n_observed, n_informative, float(lambda_state))


def objective_from_emissions(log_init, log_transition, emission_sequences, allowed_sequences,
                             observed_sequences, lambda_state: float = 1.0) -> BatchTerms:
    """Convenience function for tests and diagnostics when emissions are precomputed."""
    seqs = [sequence_terms(log_init, log_transition, e, a, o)
            for e, a, o in zip(emission_sequences, allowed_sequences, observed_sequences)]
    return aggregate_terms(seqs, lambda_state=lambda_state)


# ---------------------------------------------------------------- weighting variants
@dataclass
class Weighting:
    scheme: str = "interval"          # interval | work | class_marginal | marginal | confidence_marginal
    class_weights: Optional[dict] = None   # {ratio: w} for class_marginal
    low_conf_weight: float = 1.0           # for confidence_marginal
    work_weights: Optional[dict] = None    # optional per-work multipliers

    def __post_init__(self):
        if self.class_weights:  # keys arrive as strings from JSON job files
            self.class_weights = {int(k): float(v) for k, v in self.class_weights.items()}


MARGINAL_SCHEMES = ("class_marginal", "marginal", "confidence_marginal")


def objective_note(w: Weighting):
    if w.scheme == "interval":
        return "original: acoustic NLL/observed + lambda * (-log p(C|O,B))/observed"
    if w.scheme == "work":
        return ("reweighted (not ML): mean over works of [acoustic NLL_w/obs_w + lambda * state NLL_w/obs_w]; "
                "acoustic density unchanged")
    return ("reweighted auxiliary (not ML): acoustic NLL/observed + lambda * weighted sum of per-beat exact posterior "
            "marginal NLLs -log p(R_i in C_i|O,B), normalized to the original state-term scale")


def marginal_state_nll(log_init, log_trans, emission, observed, allowed, weights):
    """sum_i w_i * -log p(R_i in C_i | O, B) from the exact marginals of the same HMM (differentiable)."""
    lpost, _ = log_posteriors(log_init, log_trans, exclude_missing(emission, observed))
    inf = ~allowed.all(1)
    if not bool(inf.any()):
        return lpost.sum() * 0.0, 0.0
    lp_c = torch.logsumexp(lpost.masked_fill(~allowed, -1e30), 1)
    w = weights[inf]
    return -(w * lp_c[inf]).sum(), float(w.sum())


def _marginal_weights(q, weighting, device):
    if weighting.scheme == "class_marginal":
        w = torch.tensor([weighting.class_weights.get(int(l), 0.0) for l in q.labels], device=device, dtype=torch.float32)
    elif weighting.scheme == "confidence_marginal":
        w = torch.where(torch.from_numpy(q.high_confidence).to(device), 1.0, weighting.low_conf_weight)
    else:
        w = torch.ones(q.n, device=device)
    return w * float((weighting.work_weights or {}).get(q.work_id, 1.0))


def batch_objective(model, seqs, constraint_mode, log_init, log_trans, lam, weighting=None, device="cpu",
                    reg_weight=1e-4, high_confidence_only=False):
    """(loss, BatchTerms) for a batch of sequences under the lambda-weighted objective or a weighting variant."""
    weighting = weighting or Weighting()
    y = torch.from_numpy(np.concatenate([q.y for q in seqs])).to(device)
    d = torch.from_numpy(np.concatenate([q.log_duration for q in seqs])).to(device)
    obs = torch.from_numpy(np.concatenate([q.observed for q in seqs])).to(device)
    emission = model.log_prob(y, d)
    start, terms, extra = 0, [], []
    for q in seqs:
        sl = slice(start, start + q.n)
        allowed_np = constraint_for(q, constraint_mode)
        if high_confidence_only:
            allowed_np[~q.high_confidence] = True
        allowed = torch.as_tensor(allowed_np, dtype=torch.bool, device=device)
        terms.append(sequence_terms(log_init, log_trans, emission[sl], allowed, obs[sl]))
        if weighting.scheme in MARGINAL_SCHEMES:
            extra.append(marginal_state_nll(log_init, log_trans, emission[sl], obs[sl], allowed,
                                            _marginal_weights(q, weighting, device)))
        start += q.n
    base = aggregate_terms(terms, lambda_state=lam)
    n_obs = max(base.observed_intervals, 1)
    if weighting.scheme in MARGINAL_SCHEMES:
        num, wsum = sum(x[0] for x in extra), sum(x[1] for x in extra)
        # same units as the original state term: per-label NLL * informative/observed
        state = (num / wsum) * (base.informative_intervals / n_obs) if wsum else num * 0.0
        loss = base.acoustic_nll_per_observed + lam * state
    elif weighting.scheme == "work" or weighting.work_weights:
        ww = [float((weighting.work_weights or {}).get(q.work_id, 1.0)) for q in seqs]
        if weighting.scheme == "work":
            loss = torch.stack([w * (t.acoustic_nll + lam * t.state_nll) / max(t.observed_intervals, 1)
                                for w, t in zip(ww, terms)]).sum() / sum(ww)
        else:
            loss = (sum(w * (t.acoustic_nll + lam * t.state_nll) for w, t in zip(ww, terms))
                    / max(sum(w * t.observed_intervals for w, t in zip(ww, terms)), 1))
    else:
        loss = base.loss
    if reg_weight and bool(obs.any()):
        # Weak regularization discourages covariance inflation/collapse; the density stays normalized.
        _, var, u = model.distribution_parameters(d[obs])
        reg = var.log().square().mean()
        if model.rank:
            reg = reg + 0.05 * u.square().mean()
        loss = loss + reg_weight * reg
    return loss, base


@torch.no_grad()
def evaluate_objective(model, seqs, constraint_mode, log_init, log_trans, lam, device="cpu",
                       high_confidence_only=False, batch=4) -> BatchTerms:
    """Weighted objective (interval weighting, no regularizer) over a whole split."""
    model.eval()
    parts = []
    for i in range(0, len(seqs), batch):
        _, t = batch_objective(model, seqs[i:i + batch], constraint_mode, log_init, log_trans, lam, device=device,
                               reg_weight=0, high_confidence_only=high_confidence_only)
        parts.append(SequenceTerms(t.acoustic_nll_sum, t.joint_nll_sum, t.state_nll_sum,
                                   t.observed_intervals, t.informative_intervals))
    return aggregate_terms(parts, lam)

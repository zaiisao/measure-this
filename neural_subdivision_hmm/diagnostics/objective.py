"""Exact decomposition of constrained generative HMM NLL.

The state term is a posterior probability under the same generative HMM:
  -log p(C|O,B) = -log p(O,C|B) + log p(O|B).
It is not an independently parameterized classifier or CRF.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from model import exclude_missing, forward_log_prob


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


def sequence_terms(log_init: torch.Tensor, log_transition: torch.Tensor,
                   log_emission: torch.Tensor, allowed: Optional[torch.Tensor] = None,
                   observed: Optional[torch.Tensor] = None) -> SequenceTerms:
    """Return exact sequence acoustic, constrained-joint, and posterior-state NLLs.

    Missing acoustic rows are neutral emissions but remain in the state sequence and
    therefore retain their transitions. A position is informative when its allowed
    state set is a strict subset of all states, regardless of whether its audio exists.
    """
    if log_emission.ndim != 2 or log_emission.shape[0] == 0:
        raise ValueError("emissions must have shape (T,K) with T > 0")
    if observed is None:
        observed = torch.ones(log_emission.shape[0], dtype=torch.bool, device=log_emission.device)
    if observed.shape != (log_emission.shape[0],):
        raise ValueError("observed must have shape (T,)")
    emissions = exclude_missing(log_emission, observed)
    if allowed is not None and allowed.shape != log_emission.shape:
        raise ValueError("allowed must have shape (T,K)")
    acoustic_logp = forward_log_prob(log_init, log_transition, emissions)
    joint_logp = forward_log_prob(log_init, log_transition, emissions, allowed)
    acoustic_nll = -acoustic_logp
    joint_nll = -joint_logp
    state_nll = acoustic_logp - joint_logp
    if float(state_nll.detach()) < -1e-5:
        raise ArithmeticError("constrained event probability exceeded the unconstrained probability")
    if allowed is None:
        informative = 0
    else:
        informative = int((~allowed.all(dim=1)).sum().item())
    return SequenceTerms(acoustic_nll, joint_nll, state_nll,
                         int(observed.sum().item()), informative)


def aggregate_terms(terms: Sequence[SequenceTerms], lambda_state: float = 1.0,
                    eps: float = 1.0) -> BatchTerms:
    """Normalize audio and state objectives explicitly.

    Acoustic NLL is divided by observed acoustic intervals. Posterior state NLL is
    reported per informative constrained interval. To make lambda_state=1 exactly
    reproduce the original constrained-joint NLL averaged per observed audio
    interval, its contribution is state_nll_per_informative *
    informative_intervals / observed_intervals. Thus lambda_state scales the
    original posterior-supervision component, while the diagnostic reports its
    per-label magnitude separately. If there are no informative labels, the state
    term is exactly zero.
    """
    if lambda_state < 0:
        raise ValueError("lambda_state must be nonnegative")
    if not terms:
        raise ValueError("at least one sequence is required")
    ref = terms[0].acoustic_nll
    acoustic_sum = torch.stack([x.acoustic_nll for x in terms]).sum()
    state_sum = torch.stack([x.state_nll for x in terms]).sum()
    joint_sum = torch.stack([x.joint_nll for x in terms]).sum()
    n_observed = sum(x.observed_intervals for x in terms)
    n_informative = sum(x.informative_intervals for x in terms)
    acoustic_den = max(n_observed, 1)
    state_den = max(n_informative, 1)
    acoustic_mean = acoustic_sum / acoustic_den
    state_mean = state_sum / state_den if n_informative else state_sum * 0.0
    # Equivalent to acoustic_mean + lambda * state_sum / n_observed.
    state_contribution = state_mean * (n_informative / acoustic_den) if n_informative else state_sum * 0.0
    loss = acoustic_mean + float(lambda_state) * state_contribution
    joint_mean = joint_sum / acoustic_den
    # Guard this invariant: at lambda=1, this is exactly the old objective's
    # sequence-summed constrained likelihood per observed interval.
    if lambda_state == 1.0 and n_observed:
        if not torch.allclose(loss, joint_mean, atol=2e-5, rtol=2e-6):
            raise ArithmeticError("lambda_state=1 no longer equals constrained joint NLL")
    return BatchTerms(loss, acoustic_mean, state_mean, joint_mean,
                      acoustic_sum, state_sum, joint_sum,
                      n_observed, n_informative, float(lambda_state))


def objective_from_emissions(log_init: torch.Tensor, log_transition: torch.Tensor,
                             emission_sequences: Sequence[torch.Tensor],
                             allowed_sequences: Sequence[Optional[torch.Tensor]],
                             observed_sequences: Sequence[torch.Tensor],
                             lambda_state: float = 1.0) -> BatchTerms:
    """Convenience function for tests and diagnostics when emissions are precomputed."""
    if not (len(emission_sequences) == len(allowed_sequences) == len(observed_sequences)):
        raise ValueError("emissions, constraints, and observation masks must align")
    seqs = [sequence_terms(log_init, log_transition, e, a, o)
            for e, a, o in zip(emission_sequences, allowed_sequences, observed_sequences)]
    return aggregate_terms(seqs, lambda_state=lambda_state)

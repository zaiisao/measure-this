"""Exact HMM inference over per-beat subdivision states, shared by the classical and neural models.

Forward / forward-backward are torch (differentiable, used for training the neural emission).
Viterbi and the persistence transition matrix come from librosa.
"""
import librosa
import numpy as np
import torch


def persistence_log_transition(k, p_stay, *, device=None, dtype=torch.float32):
    """log of a k-state matrix with self-transition p_stay and the rest split uniformly
    (p_stay = 1/k gives uniform rows)."""
    return torch.as_tensor(np.log(librosa.sequence.transition_loop(k, p_stay)), device=device, dtype=dtype)


def apply_constraints(log_emission, allowed=None):
    return log_emission if allowed is None else log_emission.masked_fill(~allowed, -torch.inf)


def exclude_missing(log_emission, observed):
    """Integrate absent observations out by replacing their log density with zero."""
    return torch.where(observed[:, None], log_emission, torch.zeros_like(log_emission))


def forward_log_prob(log_init, log_transition, log_emission, allowed=None):
    """log p(O, constraints) by the log-space forward algorithm (differentiable)."""
    obs = apply_constraints(log_emission, allowed)
    alpha = log_init + obs[0]
    for t in range(1, len(obs)):
        alpha = torch.logsumexp(alpha[:, None] + log_transition, dim=0) + obs[t]
    return torch.logsumexp(alpha, dim=0)


def log_posteriors(log_init, log_transition, log_emission, allowed=None):
    """Per-step log state posteriors (T, K) and log p(O, constraints) (differentiable)."""
    obs = apply_constraints(log_emission, allowed)
    alpha = [log_init + obs[0]]
    for t in range(1, len(obs)):
        alpha.append(torch.logsumexp(alpha[-1][:, None] + log_transition, dim=0) + obs[t])
    beta = [torch.zeros_like(obs[0])]
    for t in range(len(obs) - 2, -1, -1):
        beta.append(torch.logsumexp(log_transition + (obs[t + 1] + beta[-1])[None, :], dim=1))
    logz = torch.logsumexp(alpha[-1], dim=0)
    return torch.stack(alpha) + torch.stack(beta[::-1]) - logz, logz


def forward_backward(log_init, log_transition, log_emission, allowed=None):
    """State posteriors (T, K) and log-likelihood. Accepts numpy or torch inputs."""
    as_numpy = isinstance(log_emission, np.ndarray)
    if as_numpy:
        log_init, log_transition, log_emission = (torch.from_numpy(np.asarray(x, dtype=np.float64))
                                                  for x in (log_init, log_transition, log_emission))
    lp, logz = log_posteriors(log_init, log_transition, log_emission, allowed)
    post = lp.exp()
    post = post / post.sum(-1, keepdim=True)
    return (post.numpy(), float(logz)) if as_numpy else (post, logz)


def viterbi(log_init, log_transition, log_emission, allowed=None):
    """MAP state path via librosa.sequence.viterbi. Emissions are shifted per step by their
    maximum before exponentiation, which leaves the path unchanged."""
    as_torch = isinstance(log_emission, torch.Tensor)
    if as_torch:
        device = log_emission.device
        log_init, log_transition, log_emission = (x.detach().cpu().double().numpy()
                                                  for x in (log_init, log_transition, log_emission))
        if allowed is not None:
            allowed = allowed.cpu().numpy()
    obs = np.asarray(log_emission, dtype=np.float64)
    if allowed is not None:
        obs = np.where(allowed, obs, -np.inf)
    prob = np.exp(obs - obs.max(1, keepdims=True))
    p_init = np.exp(log_init - np.logaddexp.reduce(log_init))  # normalising leaves the path unchanged
    path = librosa.sequence.viterbi(prob.T, np.exp(log_transition), p_init=p_init)
    return torch.as_tensor(path, dtype=torch.long, device=device) if as_torch else path

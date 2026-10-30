"""Exact HMM inference over per-beat-interval subdivision ratios (log space)."""
import numpy as np
from scipy.special import logsumexp


def persistence_transitions(k, p_stay):
    """Self-transition p_stay, remaining mass split uniformly. p_stay = 1/k gives uniform rows."""
    if k == 1:
        return np.zeros((1, 1))
    a = np.full((k, k), (1 - p_stay) / (k - 1))
    np.fill_diagonal(a, p_stay)
    with np.errstate(divide="ignore"):
        return np.log(a)


def forward_backward(log_init, log_trans, log_obs):
    """log_obs (T, K). Returns posteriors (T, K) and log p(observations)."""
    t_len, k = log_obs.shape
    alpha = np.empty((t_len, k))
    alpha[0] = log_init + log_obs[0]
    for t in range(1, t_len):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + log_trans, axis=0) + log_obs[t]
    beta = np.zeros((t_len, k))
    for t in range(t_len - 2, -1, -1):
        beta[t] = logsumexp(log_trans + (log_obs[t + 1] + beta[t + 1])[None, :], axis=1)
    loglik = logsumexp(alpha[-1])
    post = np.exp(alpha + beta - loglik)
    return post / post.sum(1, keepdims=True), float(loglik)


def viterbi(log_init, log_trans, log_obs):
    t_len, k = log_obs.shape
    delta = log_init + log_obs[0]
    back = np.zeros((t_len, k), dtype=int)
    for t in range(1, t_len):
        scores = delta[:, None] + log_trans
        back[t] = scores.argmax(0)
        delta = scores.max(0) + log_obs[t]
    path = np.empty(t_len, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path

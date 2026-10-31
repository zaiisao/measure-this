"""A normalized Gaussian emission model with exact HMM marginalization."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def fixed_grid_templates(ratios=(1, 2, 3, 4, 6), m=48, width=0.03, delta=0.06):
    phase = torch.linspace(delta, 1.0 - delta, m)
    rows = []
    for r in ratios:
        if r == 1:
            rows.append(torch.zeros(m))
        else:
            slots = torch.arange(1, r, dtype=torch.float32) / float(r)
            h = torch.exp(-0.5 * ((phase[:, None] - slots[None, :]) / width) ** 2).sum(1)
            rows.append(h / h.max().clamp_min(1e-6))
    return torch.stack(rows)


def persistence_log_transition(k: int, p_stay: float, *, device=None, dtype=torch.float32):
    if not 0.0 < p_stay < 1.0:
        raise ValueError("p_stay must be strictly between zero and one")
    if k == 1:
        return torch.zeros((1, 1), device=device, dtype=dtype)
    a = torch.full((k, k), (1.0 - p_stay) / (k - 1), device=device, dtype=dtype)
    a.fill_diagonal_(p_stay)
    return a.log()


class NeuralGaussianEmission(nn.Module):
    """Shared state/grid-conditioned Gaussian over a 48-point beat-phase envelope.

    The generator sees only the subdivision-state embedding, the fixed state grid template,
    and the exogenous beat duration. It never sees the observation when constructing parameters.
    Covariance is diagonal or low-rank plus diagonal, with exact normalized log density.
    """

    def __init__(self, ratios=(1, 2, 3, 4, 6), m=48, hidden=48, rank=2,
                 state_dim=8, variance_floor=0.05, template_width=0.03):
        super().__init__()
        if rank < 0:
            raise ValueError("rank cannot be negative")
        self.ratios = tuple(int(r) for r in ratios)
        self.m, self.rank, self.variance_floor = int(m), int(rank), float(variance_floor)
        self.state_embedding = nn.Embedding(len(self.ratios), state_dim)
        self.register_buffer("templates", fixed_grid_templates(self.ratios, self.m, template_width))
        inp = state_dim + self.m + 1
        out = 2 + self.m + self.m + self.m * self.rank
        self.generator = nn.Sequential(
            nn.Linear(inp, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out),
        )
        head = self.generator[-1]
        nn.init.normal_(head.weight, mean=0.0, std=0.004)
        nn.init.zeros_(head.bias)
        # [baseline, template amplitude, residual, raw diagonal variance, low-rank factors]
        with torch.no_grad():
            head.bias[1] = 0.5
            target = 1.0 - self.variance_floor
            head.bias[2 + self.m:2 + 2 * self.m] = math.log(math.expm1(target))

    def distribution_parameters(self, log_duration_z: torch.Tensor):
        """Return mean (N,K,M), diagonal variance (N,K,M), factor (N,K,M,Q)."""
        d = log_duration_z.reshape(-1, 1, 1).expand(-1, len(self.ratios), 1)
        emb = self.state_embedding.weight[None, :, :].expand(len(d), -1, -1)
        grid = self.templates[None, :, :].expand(len(d), -1, -1)
        z = torch.cat([grid, emb, d], dim=-1)
        raw = self.generator(z)
        a, m = self.m, self.m
        baseline = raw[..., 0:1]
        amp = raw[..., 1:2]
        resid = raw[..., 2:2 + a]
        raw_var = raw[..., 2 + a:2 + a + m]
        mu = baseline + amp * grid + 0.1 * resid
        var = F.softplus(raw_var) + self.variance_floor
        if self.rank:
            u = raw[..., 2 + a + m:].reshape(len(d), len(self.ratios), self.m, self.rank) * 0.2
        else:
            u = raw.new_zeros((len(d), len(self.ratios), self.m, 0))
        return mu, var, u

    def log_prob(self, y: torch.Tensor, log_duration_z: torch.Tensor):
        """Exact log N(y; mu, diag(var)+U Uᵀ), including log determinant and constant."""
        if y.ndim != 2 or y.shape[1] != self.m:
            raise ValueError(f"expected observations of shape (N,{self.m})")
        mu, var, u = self.distribution_parameters(log_duration_z)
        delta = y[:, None, :] - mu
        dinv = var.reciprocal()
        quad0 = (delta.square() * dinv).sum(-1)
        logdet = var.log().sum(-1)
        if self.rank:
            # Woodbury identity for (D + U Uᵀ)⁻¹ and determinant lemma.
            du = dinv[..., None] * u
            eye = torch.eye(self.rank, dtype=y.dtype, device=y.device)
            a = eye + torch.einsum("nkmq,nkmr->nkqr", u, du)
            a = 0.5 * (a + a.transpose(-1, -2))
            chol = torch.linalg.cholesky(a + 1e-6 * eye)
            v = torch.einsum("nkmq,nkm->nkq", u, dinv * delta)
            solved = torch.cholesky_solve(v.unsqueeze(-1), chol).squeeze(-1)
            quad = quad0 - (v * solved).sum(-1)
            logdet = logdet + 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        else:
            quad = quad0
        return -0.5 * (self.m * math.log(2.0 * math.pi) + logdet + quad)


def apply_constraints(log_emission: torch.Tensor, allowed: Optional[torch.Tensor]):
    if allowed is None:
        return log_emission
    if allowed.shape != log_emission.shape:
        raise ValueError("allowed-state mask must have the same (T,K) shape as emissions")
    if not torch.all(allowed.any(dim=1)):
        raise ValueError("every interval must allow at least one state")
    return log_emission.masked_fill(~allowed, -torch.inf)


def exclude_missing(log_emission: torch.Tensor, observed: torch.Tensor):
    """Integrate absent observations out by replacing their log density with zero."""
    if observed.ndim != 1 or observed.shape[0] != log_emission.shape[0]:
        raise ValueError("observed mask must have shape (T,)")
    return torch.where(observed[:, None], log_emission, torch.zeros_like(log_emission))


def forward_log_prob(log_init: torch.Tensor, log_transition: torch.Tensor,
                     log_emission: torch.Tensor, allowed: Optional[torch.Tensor] = None):
    """Exact differentiable log p(O, constraints) using the log-space forward algorithm."""
    obs = apply_constraints(log_emission, allowed)
    alpha = log_init + obs[0]
    for t in range(1, len(obs)):
        alpha = torch.logsumexp(alpha[:, None] + log_transition, dim=0) + obs[t]
    return torch.logsumexp(alpha, dim=0)


def forward_backward(log_init: torch.Tensor, log_transition: torch.Tensor,
                     log_emission: torch.Tensor, allowed: Optional[torch.Tensor] = None):
    obs = apply_constraints(log_emission, allowed)
    t_len, k = obs.shape
    alpha = [log_init + obs[0]]
    for t in range(1, t_len):
        alpha.append(torch.logsumexp(alpha[-1][:, None] + log_transition, dim=0) + obs[t])
    beta = [torch.zeros(k, dtype=obs.dtype, device=obs.device) for _ in range(t_len)]
    for t in range(t_len - 2, -1, -1):
        beta[t] = torch.logsumexp(log_transition + (obs[t + 1] + beta[t + 1])[None, :], dim=1)
    logz = torch.logsumexp(alpha[-1], dim=0)
    post = torch.stack([alpha[t] + beta[t] - logz for t in range(t_len)]).exp()
    return post / post.sum(-1, keepdim=True), logz


def viterbi(log_init: torch.Tensor, log_transition: torch.Tensor,
            log_emission: torch.Tensor, allowed: Optional[torch.Tensor] = None):
    obs = apply_constraints(log_emission, allowed)
    delta = log_init + obs[0]
    backs = []
    for t in range(1, len(obs)):
        score = delta[:, None] + log_transition
        best, back = score.max(dim=0)
        delta = best + obs[t]
        backs.append(back)
    path = [int(delta.argmax())]
    for back in reversed(backs):
        path.append(int(back[path[-1]]))
    return torch.tensor(list(reversed(path)), dtype=torch.long, device=obs.device)

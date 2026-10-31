"""A normalized Gaussian emission model with exact HMM marginalization."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from subdivision_hmm.observation import subdivision_templates


def fixed_grid_templates(ratios=(1, 2, 3, 4, 6), m=48, width=0.03, delta=0.06):
    """Classical subdivision templates, each scaled to a maximum of one (zeros for r = 1)."""
    h = torch.as_tensor(subdivision_templates(ratios, m, width, delta), dtype=torch.float32)
    return h / h.amax(1, keepdim=True).clamp_min(1e-6)


class NeuralGaussianEmission(nn.Module):
    """Shared state/grid-conditioned Gaussian over a 48-point beat-phase envelope.

    The generator sees only the subdivision-state embedding, the fixed state grid template,
    and the exogenous beat duration. It never sees the observation when constructing parameters.
    Covariance is diagonal or low-rank plus diagonal, with exact normalized log density.
    """

    def __init__(self, ratios=(1, 2, 3, 4, 6), m=48, hidden=48, rank=2,
                 state_dim=8, variance_floor=0.05, template_width=0.03):
        super().__init__()
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

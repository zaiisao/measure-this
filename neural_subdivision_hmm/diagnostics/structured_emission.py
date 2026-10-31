"""Neuralized version of the classical zero-mean Gaussian template likelihood.

The acoustic distribution remains normalized and structurally constrained:
    y | r,d ~ N(0, s(r,d)^2 I + tm(r,d)^2 11' + ta(r,d)^2 h_r h_r').
The optional conditioner predicts departures from shared base scales using only fixed
ratio/template features and exogenous standardized log-duration, never the observation.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from subdivision_hmm.observation import subdivision_templates


def subdivision_template(ratios: Sequence[int], m: int = 48, width: float = 0.03, delta: float = 0.06):
    """The exact unnormalized template used by the classical GaussianTemplate."""
    return torch.as_tensor(subdivision_templates(ratios, m, width, delta), dtype=torch.float32)


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class StructuredNeuralGaussian(nn.Module):
    """Shared neural scales over the classical Gaussian template covariance.

    Classical initialization exactly recovers the source model at every state and duration.
    The conditioner starts at zero, then may learn state/template and/or duration departures.
    """
    def __init__(self, ratios=(1, 2, 3, 4, 6), m=48, width=0.03, delta=0.06,
                 classical_s2=1.0, classical_tm2=1.0, classical_ta2=16.0,
                 hidden=24, condition_on_ratio=True, condition_on_duration=True,
                 scale_floor=1e-6):
        super().__init__()
        self.ratios = tuple(int(r) for r in ratios)
        self.m, self.width, self.delta = int(m), float(width), float(delta)
        self.condition_on_ratio = bool(condition_on_ratio)
        self.condition_on_duration = bool(condition_on_duration)
        self.scale_floor = float(scale_floor)
        if min(classical_s2, classical_tm2, classical_ta2) <= 0:
            raise ValueError("classical covariance parameters must be positive")
        self.register_buffer("templates", subdivision_template(self.ratios, self.m, self.width, self.delta))
        initial_scales = [math.sqrt(classical_s2), math.sqrt(classical_tm2), math.sqrt(classical_ta2)]
        raw = [_inv_softplus(s - self.scale_floor) for s in initial_scales]
        self.base_raw_scales = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

        input_dim = (self.m + len(self.ratios) if self.condition_on_ratio else 0) + (1 if self.condition_on_duration else 0)
        if input_dim:
            self.conditioner = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 3))
            # Zero correction gives exact classical covariance at initialization.
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
        else:
            self.conditioner = None

    @classmethod
    def from_classical_fit(cls, ratios=(1, 2, 3, 4, 6), *, s2=1.0, tm2=1.0,
                           ta2=16.0, width=0.03, **kwargs):
        """Initialize to the fitted mel50 Gaussian: s2=1, tm2=1, ta2=16, width=.03."""
        return cls(ratios=ratios, classical_s2=s2, classical_tm2=tm2,
                   classical_ta2=ta2, width=width, **kwargs)

    def scales(self, log_duration_z: torch.Tensor):
        n = int(log_duration_z.numel())
        k = len(self.ratios)
        raw = self.base_raw_scales.view(1, 1, 3).expand(n, k, 3)
        if self.conditioner is not None:
            pieces = []
            if self.condition_on_ratio:
                grid = self.templates[None, :, :].expand(n, -1, -1)
                onehot = torch.eye(k, dtype=grid.dtype, device=grid.device)[None, :, :].expand(n, -1, -1)
                pieces.extend([grid, onehot])
            if self.condition_on_duration:
                d = log_duration_z.reshape(n, 1, 1).expand(-1, k, -1)
                pieces.append(d)
            x = torch.cat(pieces, dim=-1)
            raw = raw + self.conditioner(x)
        return F.softplus(raw) + self.scale_floor

    def distribution_parameters(self, log_duration_z: torch.Tensor):
        """Return marginal diagonal variance (N,K), low-rank factors U (N,K,M,2), scales."""
        scale = self.scales(log_duration_z)
        s, tm, ta = scale.unbind(-1)
        diag_variance = s.square()
        one = torch.ones_like(self.templates)
        h = self.templates
        u = torch.stack([tm[..., None] * one[None, :, :], ta[..., None] * h[None, :, :]], dim=-1)
        return diag_variance, u, scale

    def covariance(self, log_duration_z: torch.Tensor):
        diag, u, _ = self.distribution_parameters(log_duration_z)
        eye = torch.eye(self.m, dtype=diag.dtype, device=diag.device)
        return diag[..., None, None] * eye + u @ u.transpose(-1, -2)

    def log_prob(self, y: torch.Tensor, log_duration_z: torch.Tensor):
        """Exact normalized Gaussian log density using Woodbury and determinant lemma."""
        diag, u, _ = self.distribution_parameters(log_duration_z)
        delta = y[:, None, :]
        dinv = diag.reciprocal()[..., None]
        quad0 = (delta.square() * dinv).sum(-1)
        logdet = self.m * diag.log()
        # U has rank 2: one global offset direction and the fixed rhythmic template.
        eye = torch.eye(2, dtype=y.dtype, device=y.device)
        du = dinv[..., None] * u
        a = eye + u.transpose(-1, -2) @ du
        a = 0.5 * (a + a.transpose(-1, -2))
        chol = torch.linalg.cholesky(a + 1e-5 * eye)
        weighted_delta = dinv * delta
        v = (u.transpose(-1, -2) @ weighted_delta.unsqueeze(-1)).squeeze(-1)
        solved = torch.cholesky_solve(v.unsqueeze(-1), chol).squeeze(-1)
        quad = quad0 - (v * solved).sum(-1)
        logdet = logdet + 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return -0.5 * (self.m * math.log(2.0 * math.pi) + logdet + quad)

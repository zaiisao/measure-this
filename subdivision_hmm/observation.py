"""Observation models psi(X_i, B_i, R_i = r) for one beat interval [b0, b1).

All models only look at the interval interior W = (delta, 1 - delta) in beat phase, so the beat
onsets themselves (shared by every hypothesis) carry no weight. Timing tolerances are in beat
phase, i.e. proportional to the local beat duration.

PoissonOccupancy  -- NORMALIZED likelihood of the detected onset phases (a point pattern on W)
    under an inhomogeneous Poisson process with intensity
        lambda_r(phi) = lam0 + q * sum_{k=1}^{r-1} N(phi; k/r, sigma^2).
    Each predicted slot (a metrical pulse position, not a note) emits an onset with expected count
    q (so missing notes are allowed, at cost q per empty slot); lam0 is the rate of onsets not on
    the lattice (ornaments, other voices, false detections). log p = sum_j log lambda_r(phi_j) -
    integral_W lambda_r, a density w.r.t. the unit-rate Poisson process on W. Denser lattices pay
    for slots without onsets (negative evidence), coarser lattices pay lam0-only density for
    onsets they do not predict.

GaussianTemplate  -- NORMALIZED likelihood of the onset-strength curve resampled to M phase points
    y in R^M (per-track robustly standardised), under a linear-Gaussian model
        y = mu*1 + a*h_r + eps, mu ~ N(0, tm2), a ~ N(0, ta2), eps ~ N(0, s2 I),
    h_r = sum_k Gaussian bumps of width w at k/r (h_1 = 0). Amplitudes are marginalised
    analytically: y ~ N(0, s2 I + tm2 11^T + ta2 h_r h_r^T). Peaks where h_r predicts none and
    missing energy at predicted slots both lower the likelihood. The white-noise assumption on
    correlated envelope samples makes it over-confident; that is what the calibration exponent
    kappa in the HMM addresses.

NaiveSum  -- UNNORMALIZED potential: sum of the curve at the r-1 slot positions. Kept only to
    demonstrate the nested-lattice density bias. Not a probability.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

DELTA = 0.06


class PoissonOccupancy:
    normalized = True

    def __init__(self, ratios, q=0.6, lam0=1.0, sigma=0.03, delta=DELTA):
        self.ratios, self.q, self.lam0, self.sigma, self.delta = list(ratios), q, lam0, sigma, delta
        self.slots = [np.arange(1, r) / r for r in self.ratios]
        lo, hi = delta, 1 - delta
        self.expected = np.array([lam0 * (hi - lo) + q * float(np.sum(norm.cdf(hi, s, sigma) - norm.cdf(lo, s, sigma)))
                                  for s in self.slots])

    def interval_phases(self, onset_times, b0, b1):
        ph = (np.asarray(onset_times) - b0) / (b1 - b0)
        return ph[(ph > self.delta) & (ph < 1 - self.delta)]

    def loglik_phases(self, ph):
        out = np.empty(len(self.ratios))
        for i, s in enumerate(self.slots):
            lam = self.lam0 + (self.q * norm.pdf(ph[:, None], s[None, :], self.sigma).sum(1) if len(s) else 0.0)
            out[i] = np.sum(np.log(lam)) - self.expected[i]
        return out

    def loglik(self, onset_times, b0, b1):
        return self.loglik_phases(self.interval_phases(onset_times, b0, b1))


def resample_interval(env, fps, b0, b1, m=48, delta=DELTA):
    phase = np.linspace(delta, 1 - delta, m)
    t = b0 + phase * (b1 - b0)
    return np.interp(t * fps, np.arange(len(env)), env)


def standardise(env):
    med = np.median(env)
    mad = np.median(np.abs(env - med)) * 1.4826 + 1e-6
    return (env - med) / mad


def subdivision_templates(ratios, m=48, width=0.03, delta=DELTA):
    """(K, m) Gaussian bumps at the slots k/r of each ratio's lattice (zeros for r = 1)."""
    phase = np.linspace(delta, 1 - delta, m)
    return np.stack([np.exp(-0.5 * ((phase[:, None] - np.arange(1, r)[None, :] / r) / width) ** 2).sum(1)
                     for r in ratios])


class GaussianTemplate:
    normalized = True

    def __init__(self, ratios, s2=1.0, tm2=4.0, ta2=4.0, width=0.03, m=48, delta=DELTA):
        self.ratios, self.m = list(ratios), m
        self.inv, self.logdet = [], []
        for h in subdivision_templates(self.ratios, m, width, delta):
            cov = s2 * np.eye(m) + tm2 * np.ones((m, m)) + ta2 * np.outer(h, h)
            _, ld = np.linalg.slogdet(cov)
            self.inv.append(np.linalg.inv(cov)); self.logdet.append(ld)
        self.const = -0.5 * m * np.log(2 * np.pi)

    def loglik_vectors(self, y):
        """y (N, M) -> (N, K)"""
        return np.stack([self.const - 0.5 * ld - 0.5 * np.einsum("nm,mk,nk->n", y, inv, y)
                         for inv, ld in zip(self.inv, self.logdet)], 1)


class NaiveSum:
    normalized = False

    def __init__(self, ratios, m=48, delta=DELTA):
        self.ratios = list(ratios)
        phase = np.linspace(delta, 1 - delta, m)
        self.idx = [np.array([int(np.argmin(np.abs(phase - k / r))) for k in range(1, r)], dtype=int) for r in self.ratios]

    def loglik_vectors(self, y):
        return np.stack([y[:, i].sum(1) if len(i) else np.zeros(len(y)) for i in self.idx], 1)


def mixture_loglik(ll, log_prior=None):
    """Unsupervised fit criterion: sum_i log sum_r p(r) psi_r (uniform p(r) by default)."""
    k = ll.shape[1]
    lp = np.full(k, -np.log(k)) if log_prior is None else log_prior
    return float(logsumexp(ll + lp, axis=1).sum())

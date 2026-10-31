import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'subdivision_hmm'))
from structured_emission import StructuredNeuralGaussian
from observation import GaussianTemplate


class StructuredGaussianTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = StructuredNeuralGaussian.from_classical_fit(m=12, hidden=10)
        self.d = torch.tensor([-0.8, 0.0, 0.7])
        self.y = torch.randn(3, 12)

    def test_normalized_log_density_matches_dense_mvn(self):
        ll = self.model.log_prob(self.y, self.d)
        scales = self.model.scales(self.d)
        h = self.model.templates
        eye = torch.eye(self.model.m)
        for i in range(len(self.d)):
            for k in range(len(self.model.ratios)):
                s, tm, ta = scales[i, k]
                cov = s.square() * eye + tm.square() * torch.ones_like(eye) + ta.square() * torch.outer(h[k], h[k])
                reference = torch.distributions.MultivariateNormal(torch.zeros(self.model.m), covariance_matrix=cov).log_prob(self.y[i])
                self.assertTrue(torch.allclose(ll[i, k], reference, atol=2e-5, rtol=2e-5))

    def test_classical_initialization_matches_fitted_covariance(self):
        # Values selected by subdivision_hmm/scripts/run_experiment.py::fit_params:
        # tm2=1, ta2=16, width=.03; GaussianTemplate fixes s2=1.
        diag, u, scales = self.model.distribution_parameters(self.d)
        self.assertTrue(torch.allclose(scales[..., 0], torch.ones_like(scales[..., 0]), atol=2e-6))
        self.assertTrue(torch.allclose(scales[..., 1], torch.ones_like(scales[..., 1]), atol=2e-6))
        self.assertTrue(torch.allclose(scales[..., 2], torch.full_like(scales[..., 2], 4.0), atol=2e-6))
        dense = self.model.covariance(self.d)
        for i in range(len(self.d)):
            for k, h in enumerate(self.model.templates):
                classical = torch.eye(self.model.m) + torch.ones(self.model.m, self.model.m) + 16.0 * torch.outer(h, h)
                self.assertTrue(torch.allclose(dense[i, k], classical, atol=1e-5, rtol=1e-5))

    def test_initial_emission_matches_original_gaussian_template_density(self):
        model=StructuredNeuralGaussian.from_classical_fit(m=48)
        y=torch.randn(4,48)
        actual=model.log_prob(y,torch.zeros(4)).detach().numpy()
        source=GaussianTemplate([1,2,3,4,6],s2=1.,tm2=1.,ta2=16.,width=.03,m=48)
        expected=source.loglik_vectors(y.numpy())
        self.assertTrue(torch.allclose(torch.from_numpy(actual),torch.from_numpy(expected).float(),atol=2e-4,rtol=2e-5))

    def test_covariance_positive_definite(self):
        eig = torch.linalg.eigvalsh(self.model.covariance(self.d))
        self.assertGreater(float(eig.min()), 0.0)

    def test_optional_conditioning_preserves_initial_solution(self):
        model = StructuredNeuralGaussian.from_classical_fit(m=12, condition_on_ratio=True, condition_on_duration=True)
        a = model.scales(torch.tensor([-2.0, 0.0, 3.0]))
        self.assertTrue(torch.allclose(a[..., 0], torch.ones_like(a[..., 0]), atol=2e-6))
        self.assertTrue(torch.allclose(a[..., 1], torch.ones_like(a[..., 1]), atol=2e-6))
        self.assertTrue(torch.allclose(a[..., 2], torch.full_like(a[..., 2], 4.0), atol=2e-6))

    def test_finite_nonzero_gradients(self):
        loss = -self.model.log_prob(self.y, self.d).mean()
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)

    def test_full_feature_dimension_optimizer_steps_are_stable(self):
        model=StructuredNeuralGaussian.from_classical_fit(m=48,condition_on_ratio=True,
            condition_on_duration=True,scale_floor=.05)
        opt=torch.optim.Adam(model.parameters(),lr=2e-4)
        y=torch.randn(40,48);d=torch.linspace(-1,1,40)
        for _ in range(5):
            opt.zero_grad(set_to_none=True);loss=-model.log_prob(y,d).mean()
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            opt.step()


if __name__ == '__main__':
    unittest.main()

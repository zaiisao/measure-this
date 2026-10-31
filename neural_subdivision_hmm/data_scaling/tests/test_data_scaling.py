import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common  # noqa: E402

IDS = [f"Composer{i % 4}::work_{i}" for i in range(40)]


class TestSamplers(unittest.TestCase):
    def test_uniform_reproducible_and_valid(self):
        a = common.sample_uniform(IDS, 12, 7)
        self.assertEqual(a, common.sample_uniform(IDS, 12, 7))
        self.assertNotEqual(a, common.sample_uniform(IDS, 12, 8))
        self.assertEqual(len(set(a)), 12)
        self.assertTrue(set(a) <= set(IDS))

    def test_nested_prefixes(self):
        order = common.nested_order(IDS, 3)
        self.assertEqual(sorted(order), sorted(IDS))
        for small, big in ((5, 10), (10, 30)):
            self.assertTrue(set(order[:small]) <= set(order[:big]))

    def test_stratified_allocation(self):
        stats = [{"work_id": w, "composer": w.split("::")[0]} for w in IDS]
        pick = common.sample_stratified(stats, 20, 1, key=lambda s: s["composer"])
        counts = {c: sum(p.startswith(c) for p in pick) for c in {s["composer"] for s in stats}}
        self.assertEqual(len(pick), 20)
        self.assertTrue(all(v == 5 for v in counts.values()))


class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.norm, cls.pool, cls.val = common.data()

    def test_work_disjoint_splits(self):
        train = {q.work_id for q in self.pool}
        val = {q.work_id for q in self.splits["val"]}
        test = {q.work_id for q in self.splits["test"]}
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(len(train), 96)

    def test_original_order_matches_recorded_subset(self):
        meta = json.loads((HERE.parent / "diagnostics/incremental_n30_l10_s0_diagnostic_training.json").read_text())
        by = {q.performance_id: q.work_id for q in self.pool}
        self.assertEqual([by[p] for p in meta["selected_performance_ids"]], common.original_nested_order(self.pool)[:30])

    def _setup(self):
        torch.manual_seed(0)
        from model import NeuralGaussianEmission, persistence_log_transition
        m = common.NeuralGaussianEmission(ratios=common.RATIOS, m=self.norm["feature_dim"], hidden=8, rank=1)
        li = torch.full((5,), -np.log(5)); lt = persistence_log_transition(5, .99)
        short = []
        for q in self.pool[:3]:
            import copy
            c = copy.copy(q)
            for f in ("beat_index", "y", "log_duration", "observed", "labels", "exact_allowed", "family_allowed", "high_confidence"):
                setattr(c, f, getattr(q, f)[:40])
            short.append(c)
        return m, li, lt, short

    def test_interval_scheme_is_original_objective(self):
        from diagnostics.weighted_train import batch_terms
        m, li, lt, seqs = self._setup()
        ours, _ = common.batch_loss(m, seqs, li, lt, 10.0, common.Weighting(), "cpu", reg_weight=0)
        orig, _, _ = batch_terms(m, seqs, "exact", li, lt, "cpu", 10.0, regularizer_weight=0)
        self.assertTrue(torch.allclose(ours, orig))

    def test_work_scheme_single_sequence_equals_interval(self):
        m, li, lt, seqs = self._setup()
        a, _ = common.batch_loss(m, seqs[:1], li, lt, 10.0, common.Weighting("work"), "cpu", reg_weight=0)
        b, _ = common.batch_loss(m, seqs[:1], li, lt, 10.0, common.Weighting(), "cpu", reg_weight=0)
        self.assertTrue(torch.allclose(a, b, atol=1e-4))

    def test_marginal_variants(self):
        m, li, lt, seqs = self._setup()
        ac, base = common.batch_loss(m, seqs, li, lt, 0.0, common.Weighting("marginal"), "cpu", reg_weight=0)
        self.assertTrue(torch.allclose(ac, base.acoustic_nll_per_observed))  # lambda 0 -> acoustic only
        zero = common.Weighting("class_marginal", class_weights={r: 0.0 for r in common.RATIOS})
        z, base = common.batch_loss(m, seqs, li, lt, 10.0, zero, "cpu", reg_weight=0)
        self.assertTrue(torch.allclose(z, base.acoustic_nll_per_observed))
        one = common.Weighting("confidence_marginal", low_conf_weight=1.0)
        a, _ = common.batch_loss(m, seqs, li, lt, 10.0, one, "cpu", reg_weight=0)
        b, _ = common.batch_loss(m, seqs, li, lt, 10.0, common.Weighting("marginal"), "cpu", reg_weight=0)
        self.assertTrue(torch.allclose(a, b))
        a.backward()  # differentiable

    def test_marginal_posterior_matches_forward_backward(self):
        from model import forward_backward
        torch.manual_seed(1)
        em = torch.randn(7, 5); li = torch.full((5,), -np.log(5))
        lt = torch.log(torch.full((5, 5), .05) + torch.eye(5) * .75)
        allowed = torch.zeros(7, 5, dtype=torch.bool); allowed[:, 2] = True
        nll, w = common.marginal_state_nll(li, lt, em, torch.ones(7, dtype=torch.bool), allowed, torch.ones(7))
        post, _ = forward_backward(li, lt, em)
        self.assertAlmostEqual(float(nll), float(-torch.log(post[:, 2]).sum()), places=4)
        self.assertEqual(w, 7.0)

    def test_step_budget_accounting(self):
        old = common.RUNS
        with tempfile.TemporaryDirectory() as tmp:
            common.RUNS = Path(tmp)
            try:
                ids = [q.work_id for q in self.pool[:3]]
                meta = common.train(common.RunSpec(name="t", work_ids=ids, budget="steps", steps=5, eval_every=5))
            finally:
                common.RUNS = old
        self.assertEqual(meta["optimizer_steps"], 5)
        self.assertAlmostEqual(meta["epochs_equivalent"], 5 * 4 / 3)
        used = sum(len(p) for p in meta["sampling_schedule_permutations"])
        self.assertGreaterEqual(used, 20)


if __name__ == "__main__":
    unittest.main()

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm import forward_backward, persistence_transitions, viterbi  # noqa: E402
from observation import GaussianTemplate, NaiveSum, PoissonOccupancy  # noqa: E402

R = [1, 2, 3, 4, 6]


def make_onsets(beats, pattern_per_beat, rng=None, jitter=0.0, drop=0.0):
    """pattern_per_beat[i]: list of phases played in beat i (beat onsets included automatically)."""
    t = []
    for i in range(len(beats) - 1):
        b0, b1 = beats[i], beats[i + 1]
        t.append(b0)
        for ph in pattern_per_beat[i]:
            if rng is not None and rng.random() < drop:
                continue
            j = rng.normal(0, jitter) if rng is not None and jitter else 0.0
            t.append(b0 + (ph + j) * (b1 - b0))
    return np.sort(t)


def lattice(r):
    return [k / r for k in range(1, r)]


def decode(beats, onsets, p_stay=0.95, model=None):
    model = model or PoissonOccupancy(R)
    ll = np.stack([model.loglik(onsets, beats[i], beats[i + 1]) for i in range(len(beats) - 1)])
    k = len(R)
    path = viterbi(np.full(k, -np.log(k)), persistence_transitions(k, p_stay), ll)
    post, _ = forward_backward(np.full(k, -np.log(k)), persistence_transitions(k, p_stay), ll)
    return np.array(R)[path], post, ll


class TestInference(unittest.TestCase):
    def test_forward_backward_and_viterbi_match_brute_force(self):
        rng = np.random.default_rng(0)
        k, t = 3, 5
        li = np.log(rng.dirichlet(np.ones(k)))
        lt = np.log(rng.dirichlet(np.ones(k), size=k))
        lo = rng.normal(size=(t, k))
        joint = {}
        for seq in itertools.product(range(k), repeat=t):
            joint[seq] = li[seq[0]] + lo[0, seq[0]] + sum(lt[seq[i - 1], seq[i]] + lo[i, seq[i]] for i in range(1, t))
        z = logsumexp(list(joint.values()))
        post, ll = forward_backward(li, lt, lo)
        self.assertAlmostEqual(ll, z, places=8)
        for i in range(t):
            for s in range(k):
                m = logsumexp([v for q, v in joint.items() if q[i] == s])
                self.assertAlmostEqual(post[i, s], np.exp(m - z), places=8)
        self.assertEqual(tuple(viterbi(li, lt, lo)), max(joint, key=joint.get))

    def test_uniform_transition_equals_independent(self):
        lo = np.random.default_rng(1).normal(size=(10, 5))
        path = viterbi(np.zeros(5), persistence_transitions(5, 1 / 5), lo)
        self.assertTrue(np.array_equal(path, lo.argmax(1)))


class TestPoissonOccupancy(unittest.TestCase):
    beats = np.arange(0, 21) * 0.5

    def test_clean_patterns(self):
        for r in R:
            on = make_onsets(self.beats, [lattice(r)] * 20)
            pred, _, _ = decode(self.beats, on)
            self.assertTrue(np.all(pred == r), f"r={r}: {pred}")

    def test_missing_notes_and_jitter(self):
        rng = np.random.default_rng(3)
        on = make_onsets(self.beats, [lattice(4)] * 20, rng, jitter=0.015, drop=0.25)
        pred, _, _ = decode(self.beats, on)
        self.assertGreaterEqual(np.mean(pred == 4), 0.9)

    def test_syncopation_dotted_rhythm(self):
        # dotted-eighth + sixteenth: only phase 3/4 is played; 1/2 is silent
        on = make_onsets(self.beats, [[0.75]] * 20)
        pred, _, _ = decode(self.beats, on)
        self.assertTrue(np.all(pred == 4))

    def test_tempo_change(self):
        ibi = np.linspace(0.8, 0.35, 30)
        beats = np.concatenate([[0], np.cumsum(ibi)])
        on = make_onsets(beats, [lattice(3)] * 30)
        pred, _, _ = decode(beats, on)
        self.assertTrue(np.all(pred == 3))

    def test_subdivision_change(self):
        on = make_onsets(self.beats, [lattice(2)] * 10 + [lattice(3)] * 10)
        pred, post, _ = decode(self.beats, on)
        self.assertTrue(np.array_equal(pred, [2] * 10 + [3] * 10))
        self.assertEqual(int(np.sum(np.diff(pred) != 0)), 1)

    def test_persistence_bridges_ambiguous_beats(self):
        # half the beats hold a long note (no interior onsets): persistence keeps the triplet level
        pat = [lattice(3) if i % 2 == 0 else [] for i in range(20)]
        indep, _, _ = decode(self.beats, make_onsets(self.beats, pat), p_stay=1 / 5)
        hmm, _, _ = decode(self.beats, make_onsets(self.beats, pat), p_stay=0.99)
        self.assertTrue(np.any(indep == 1))
        self.assertTrue(np.all(hmm == 3))


class TestIdentifiability(unittest.TestCase):
    """Counterexamples: different true lattices that produce identical onset evidence."""
    beats = np.arange(0, 21) * 0.5

    def test_sparse_fine_lattice_is_indistinguishable(self):
        # true 4x played only on its 2x positions == true 2x; true 6x played only on 3/6 == true 2x
        on_2 = make_onsets(self.beats, [lattice(2)] * 20)
        on_4_sparse = make_onsets(self.beats, [[2 / 4]] * 20)
        on_6_sparse = make_onsets(self.beats, [[3 / 6]] * 20)
        _, p2, _ = decode(self.beats, on_2)
        _, p4, _ = decode(self.beats, on_4_sparse)
        _, p6, _ = decode(self.beats, on_6_sparse)
        self.assertTrue(np.allclose(p2, p4) and np.allclose(p2, p6))
        self.assertTrue(np.all(p2.argmax(1) == R.index(2)))  # Occam: the coarsest consistent lattice

    def test_two_against_three_mixture(self):
        # phases 1/3, 1/2, 2/3 in one beat: only 6 contains all; 2 and 3 each explain a subset
        on = make_onsets(self.beats, [[1 / 3, 1 / 2, 2 / 3]] * 20)
        pred, _, _ = decode(self.beats, on)
        self.assertTrue(np.all(pred == 6))


class TestEnvelopeModels(unittest.TestCase):
    def envelopes(self, r, n=40, noise=0.3, seed=0):
        rng = np.random.default_rng(seed)
        phase = np.linspace(0.06, 0.94, 48)
        h = np.exp(-0.5 * ((phase[:, None] - np.arange(1, r)[None, :] / r) / 0.03) ** 2).sum(1)
        return 3 * h[None, :] + noise * rng.normal(size=(n, 48))

    def test_gaussian_template_recovers_level(self):
        g = GaussianTemplate(R)
        for r in R:
            ll = g.loglik_vectors(self.envelopes(r))
            self.assertGreaterEqual(np.mean(np.array(R)[ll.argmax(1)] == r), 0.95, f"r={r}")

    def test_naive_sum_is_biased_to_dense_lattices(self):
        # onset strength is non-negative: any background energy adds up over more slots
        ll = NaiveSum(R).loglik_vectors(np.abs(self.envelopes(2)) + 0.3)
        pred = np.array(R)[ll.argmax(1)]
        self.assertTrue(np.all(pred == 6))  # the densest lattice containing the midpoint wins


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from subdivision_hmm.metrics import auc, ece


class TestNonFiniteMetrics(unittest.TestCase):
    def test_auc_marks_nonfinite_scores_invalid(self):
        self.assertTrue(np.isnan(auc([0, 1, 0], [0.1, np.nan, 0.2])))

    def test_ece_marks_nonfinite_confidence_invalid(self):
        self.assertTrue(np.isnan(ece([0.8, np.nan], [True, False])))


if __name__ == "__main__":
    unittest.main()

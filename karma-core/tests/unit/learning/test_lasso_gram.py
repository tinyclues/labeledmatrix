import unittest
from karma.core.utils.utils import use_seed
from karma.learning.lasso_gram import lasso_gram

import numpy as np

class LassoGramTestCase(unittest.TestCase):
    def test_max_features(self):
        n_row = 100
        n_col = 20
        with use_seed(42):
            X = np.random.randn(n_row, n_col)
            w = np.random.randn(n_col)
            y = X.dot(w) + 0.1 * np.random.randn(n_row)
        XX, Xy = np.dot(X.T, X), np.dot(X.T, y)
        _, active, _ = lasso_gram(Xy, XX, max_features=5)
        self.assertLessEqual(len(active), 5)

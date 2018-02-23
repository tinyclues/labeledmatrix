import unittest

import numpy as np
from cyperf.matrix.karma_sparse import KarmaSparse

from karma.core.utils.utils import use_seed
from karma.learning.lasso import best_lasso_model_cv_from_moments
from karma.learning.regression import create_meta_of_regression


class RegressionUtilsTestCase(unittest.TestCase):
    def test_create_meta_of_regression(self):
        y_hat = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([0, 1, 0, 1])

        meta = create_meta_of_regression(y_hat, y)

        self.assertEqual(set(meta.keys()), {'curves', 'train_MSE'})
        self.assertEqual(meta['train_MSE'], 0.275)
        np.testing.assert_array_equal(meta['curves'].x, [0., 0., 0.5, 0.5, 1.])
        np.testing.assert_array_equal(meta['curves'].y, [0., 0.5, 0.5, 1., 1.])

        y_hat = np.array([0.5, 0.5, 1.5, 2])
        y = np.array([0, 1, 2, 3])

        meta = create_meta_of_regression(y_hat, y)

        np.testing.assert_array_equal(meta['curves'].x, [0., 0.25, 0.5, 0.75, 1.])
        np.testing.assert_array_almost_equal(meta['curves'].y, [0., 0.5, 0.8333, 1., 1.], decimal=3)


class LassoTestCase(unittest.TestCase):

    def test_best_lasso_model_cv_from_moments(self):
        with use_seed(123):
            xx = np.random.rand(10 ** 4, 100)
            w = np.random.randint(0, 3, size=xx.shape[1])
            yy = xx.dot(w) + np.random.randn(xx.shape[0]) * 0.1 + 4.
        predictions, intercept, betas = best_lasso_model_cv_from_moments(xx, yy)

        self.assertLess(np.max(np.abs((w - betas))), 0.01)
        self.assertLess(np.max(np.abs(intercept - 4)), 0.1)
        self.assertLess(np.std(yy - intercept - xx.dot(betas)), 0.5 * np.std(yy))

        predictions_sp, intercept_sp, betas_sp = best_lasso_model_cv_from_moments(KarmaSparse(xx), yy)
        np.testing.assert_array_almost_equal(betas_sp, betas, decimal=6)
        np.testing.assert_array_almost_equal(intercept_sp, intercept, decimal=6)

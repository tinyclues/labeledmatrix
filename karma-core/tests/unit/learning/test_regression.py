import unittest

import numpy as np

from karma.learning.regression import create_meta_of_regression


class RegressionUtilsTestCase(unittest.TestCase):
    def test_create_meta_of_regression(self):
        y_hat = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([0, 1, 0, 1])

        meta = create_meta_of_regression(y_hat, y, train_mse=15)

        self.assertEqual(set(meta.keys()), {'curves', 'train_MSE'})
        self.assertEqual(meta['train_MSE'], 15)
        np.testing.assert_array_equal(meta['curves'].x, [0., 0., 0.5, 0.5, 1.])
        np.testing.assert_array_equal(meta['curves'].y, [0., 0.5, 0.5, 1., 1.])

        y_hat = np.array([0.5, 0.5, 1.5, 2])
        y = np.array([0, 1, 2, 3])

        meta = create_meta_of_regression(y_hat, y)

        np.testing.assert_array_equal(meta['curves'].x, [0., 0.25, 0.5, 0.75, 1.])
        np.testing.assert_array_almost_equal(meta['curves'].y, [0., 0.5, 0.8333, 1., 1.], decimal=3)

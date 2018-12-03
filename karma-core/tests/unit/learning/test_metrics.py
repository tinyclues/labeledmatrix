#
# Copyright tinyclues, All rights reserved
#

import unittest
import numpy as np
from numpy.testing import assert_allclose
from karma.learning.metrics import normalized_log_loss_from_prediction, auc_from_prediction
from karma.types import Missing
from sklearn.metrics import roc_auc_score
from karma.core.utils import use_seed
from karma.core.curve import gain_curve_from_prediction


def skl_auc(predicted_values, true_values):
    return 2. * roc_auc_score(true_values, predicted_values) - 1.


class MetricsTestCase(unittest.TestCase):
    def test_normalized_log_loss_from_prediction(self):
        p = 0.25
        exp = np.mean([np.log(0.9), np.log(0.5), np.log(0.3), np.log(0.8)]) / (p * np.log(p) + (1 - p) * np.log(1 - p))
        self.assertEqual(exp, normalized_log_loss_from_prediction([0.1, 0.5, 0.3, 0.2], [0, False, 1, 0.]))

        with self.assertRaises(ValueError) as e:
            _ = normalized_log_loss_from_prediction([0.1, 0.2, 0.3], [-1, 0.5, 17])
        self.assertEqual(str(e.exception), 'Normalized logloss can be computed only for binary true_values')

        with self.assertRaises(ValueError) as e:
            _ = normalized_log_loss_from_prediction([0.1, 0.2, None], [0, 1, 0])
        self.assertEqual(str(e.exception), "Input contains NaN, infinity or a value too large for dtype('float64').")

        with self.assertRaises(ValueError) as e:
            _ = normalized_log_loss_from_prediction([0.1, 0.2], [0, 1, 0])
        self.assertEqual(str(e.exception), 'Found input variables with inconsistent numbers of samples: [2, 3]')

        with self.assertRaises(TypeError) as e:
            _ = normalized_log_loss_from_prediction([0.1, 0.2, Missing], [0, 1, 0])
        self.assertEqual(str(e.exception), "float() argument must be a string or a number")

    def test_normalized_log_loss_edge_cases(self):
        self.assertEqual(normalized_log_loss_from_prediction([0.7] * 1000, [1] * 999 + [0]), 45.214456435843609)

        self.assertTrue(np.isnan(normalized_log_loss_from_prediction([0.1, 0.5, 0.3, 0.2], [0, 0, False, False])))
        self.assertTrue(np.isnan(normalized_log_loss_from_prediction([0.1, 0.5, 0.3, 0.2], [1, 1, True, True])))

        self.assertEqual(normalized_log_loss_from_prediction([0., 0., 0, 0], [0, 0, False, False]), 0.)
        self.assertEqual(normalized_log_loss_from_prediction([1., 1., 1, 1], [1, 1, True, True]), 0.)

        self.assertTrue(np.isnan(normalized_log_loss_from_prediction([1., 1., 1, 1], [0, 0, False, False])))
        self.assertTrue(np.isnan(normalized_log_loss_from_prediction([0., 0., 0, 0], [1, 1, True, True])))

        self.assertEqual(normalized_log_loss_from_prediction([0.1, 0., 0.3, 0.2], [1, 0, 1, 0]), 1.9531911120563956)
        self.assertTrue(np.isnan(normalized_log_loss_from_prediction([0.1, 0.5, 0., 0.2], [1, 0, 1, 0])))

        self.assertTrue(np.isnan(normalized_log_loss_from_prediction([0.1, 1., 0.3, 0.2], [1, 0, 1, 0])))
        self.assertEqual(normalized_log_loss_from_prediction([0.1, 0.5, 1., 0.2], [1, 0, 1, 0]), 1.6856790653439793)

    def test_auc_from_prediction(self):
        with use_seed(42):
            x = np.random.uniform(size=100)
            y = x > 0.6
            self.assertEqual(auc_from_prediction(x, y), skl_auc(x, y))
            self.assertEqual(auc_from_prediction(x, y), gain_curve_from_prediction(x, y).auc)
            self.assertEqual(auc_from_prediction(x, y), 1.)  # structural given y definition

        with use_seed(42):
            x = np.random.uniform(size=100)
            y = np.random.binomial(1, 0.3, 100)
            assert_allclose(auc_from_prediction(x, y), gain_curve_from_prediction(x, y).auc, rtol=0.001)
            assert_allclose(auc_from_prediction(x, y), skl_auc(x, y), rtol=0.001)

    def test_auc_from_prediction_edge_cases(self):
        x = np.random.uniform(size=10)
        y = [1.] * 10
        self.assertEqual(auc_from_prediction(x, y), 1.)
        # self.assertEqual(auc_from_prediction(x, y), gain_curve_from_prediction(x, y).auc) -> gain_curve_from_pred returns 0.
        y = [0.] * 10
        self.assertEqual(auc_from_prediction(x, y), 1.)
        # self.assertEqual(auc_from_prediction(x, y), gain_curve_from_prediction(x, y).auc) -> gain_curve_from_pred returns -1.

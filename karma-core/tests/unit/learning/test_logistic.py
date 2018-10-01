import unittest
import numpy as np
from mock import patch

from karma.core.utils.utils import use_seed
from karma.learning.logistic import logistic_coefficients_and_posteriori
from karma.learning.utils import VirtualHStack


class LogisticCoefficientsAndPosterioriTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with use_seed(157312):
            n_sample = 1000
            cls.X = (np.random.rand(n_sample, 10),
                     np.random.rand(n_sample))
            cls.vhs_X = VirtualHStack(cls.X, nb_threads=2, nb_inner_threads=2)
            cls.y = cls.X[0].dot(np.random.randint(20, size=10)) + cls.X[1] * 42 + np.random.randint(20)

    def test_close_on_raise(self):
        def fake_loss(*args):
            raise KeyboardInterrupt

        with patch('karma.learning.logistic.logistic_loss_and_grad', side_effect=fake_loss):
            with self.assertRaises(KeyboardInterrupt):
                _ = logistic_coefficients_and_posteriori(self.vhs_X, self.y, max_iter=100)

        self.assertIsNone(self.vhs_X.pool)

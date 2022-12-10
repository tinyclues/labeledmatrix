import unittest

import numpy as np
from mock import patch
from numpy.testing import assert_almost_equal, assert_equal

from karma.core.utils.utils import use_seed
from karma.learning.logistic import (logistic_coefficients_and_posteriori, expit, logistic_loss_and_grad,
                                     logistic_loss_and_grad_elastic_net)
from karma.learning.utils import VirtualHStack


class LogisticCoefficientsAndPosterioriTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with use_seed(157312):
            cls.n_sample, cls.n_features = n_sample, n_features = 10 ** 5, 5
            cls.X = (np.random.rand(n_sample, n_features - 1),
                     np.random.rand(n_sample))
            cls.coeffs = np.hstack((np.random.randint(20, size=n_features - 1) - 10,
                                    .2))
            vhs_X = VirtualHStack(cls.X, nb_threads=2, nb_inner_threads=2)
            cls.y = np.random.rand(n_sample) < expit(vhs_X.dot(cls.coeffs) + 0.42)

    def test_close_on_raise(self):
        def fake_loss(*args):
            raise KeyboardInterrupt

        vhs_X = VirtualHStack(self.X, nb_threads=2, nb_inner_threads=2)
        with patch('karma.learning.logistic.logistic_loss_and_grad', side_effect=fake_loss):
            with self.assertRaises(KeyboardInterrupt):
                _ = logistic_coefficients_and_posteriori(vhs_X, self.y, max_iter=100)

        self.assertIsNone(vhs_X.pool)

    def test_elastic_net_loss(self):
        vhs_X = VirtualHStack(self.X)
        n_features = vhs_X.shape[1]
        l1_coeff = np.random.rand() + 0.1
        kwargs = dict(X=vhs_X, y=self.y,
                      alpha=np.random.randint(10, size=n_features), w_priori=self.coeffs,
                      alpha_intercept=1, intercept_priori=0,
                      sample_weight=None)
        w = self.coeffs + np.random.randn(len(self.coeffs))
        w_extended = np.hstack((np.maximum(w - kwargs['w_priori'], 0), np.maximum(kwargs['w_priori'] - w, 0)))
        out, grad = logistic_loss_and_grad(np.hstack((w, 0.42)), **kwargs)
        out_en, grad_en = logistic_loss_and_grad_elastic_net(np.hstack((w_extended, 0.42)), l1_coeff=l1_coeff, **kwargs)

        assert_almost_equal(out + l1_coeff * kwargs['alpha'].dot(np.abs(w - kwargs['w_priori'])), out_en, decimal=10)
        assert_almost_equal(grad[:n_features] + l1_coeff * kwargs['alpha'], grad_en[:n_features], decimal=10)
        assert_almost_equal(-grad[:n_features] + l1_coeff * kwargs['alpha'], grad_en[n_features:2 * n_features],
                            decimal=10)
        assert_almost_equal(grad[-1], grad_en[-1])

        # test the zero equivalency
        out_en, grad_en = logistic_loss_and_grad_elastic_net(np.hstack((w_extended, 0.42)), l1_coeff=0, **kwargs)

        assert_equal(out, out_en)
        assert_equal(grad[:n_features], grad_en[:n_features])
        assert_equal(-grad[:n_features], grad_en[n_features:2 * n_features])
        assert_equal(grad[-1], grad_en[-1])

    def test_elastic_net_loss_args_creation(self):
        w = self.coeffs + np.random.randn(len(self.coeffs))
        with patch('karma.learning.logistic.fmin_l_bfgs_b') as mock:
            with self.assertRaises(ValueError):
                _ = logistic_coefficients_and_posteriori(self.X, self.y, max_iter=100, l1_coeff=1,
                                                         w_warm=np.hstack((w, 0.042)), w_priori=self.coeffs)
        w_warm, bounds = mock.call_args[0][1], mock.call_args[1]['bounds']
        assert_almost_equal(np.maximum(w - self.coeffs, 0), w_warm[:self.n_features])
        assert_almost_equal(np.maximum(self.coeffs - w, 0), w_warm[self.n_features:2 * self.n_features])
        self.assertEqual(2 * self.n_features + 1, len(w_warm))
        self.assertEqual(2 * self.n_features + 1, len(bounds))
        self.assertEqual({(0, None)}, set(bounds[:2 * self.n_features]))

    def test_elastic_net_equivalency(self):
        with use_seed(123):
            w = self.coeffs + np.random.randn(len(self.coeffs))
            kwargs = dict(max_iter=100, w_warm=np.hstack((w, 0.042)), w_priori=self.coeffs + 0.1)
            res = logistic_coefficients_and_posteriori(self.X, self.y, C_priori=1, l1_coeff=0, **kwargs)
            res_eps = logistic_coefficients_and_posteriori(self.X, self.y, C_priori=1 + 1e-10, l1_coeff=0, **kwargs)
            res_en = logistic_coefficients_and_posteriori(self.X, self.y, C_priori=1, l1_coeff=1e-10, **kwargs)

        assert_almost_equal(np.hstack(res[2]), np.hstack(res_eps[2]), decimal=10)
        assert_almost_equal(np.hstack(res[2]), np.hstack(res_en[2]), decimal=3)

    def test_limit_cases(self):
        with use_seed(123):
            w = self.coeffs + np.random.randn(len(self.coeffs))
            w_priori = self.coeffs + 0.1
            kwargs = dict(max_iter=100, w_warm=np.hstack((w, 0.042)), w_priori=w_priori)
            res = logistic_coefficients_and_posteriori(self.X, self.y, C_priori=1e-10, l1_coeff=0, **kwargs)
            res_en = logistic_coefficients_and_posteriori(self.X, self.y, C_priori=1, l1_coeff=1e10, **kwargs)

        assert_almost_equal(np.hstack(res[2]), w_priori, decimal=5)
        assert_almost_equal(np.hstack(res_en[2]), w_priori, decimal=5)

    def test_correlated_features(self):
        with use_seed(123):
            n = 10 ** 5
            x = np.random.rand(n)
            X = (2 * x, x / 5)
            y = np.random.rand(n) < expit(x)

            betas = np.hstack(logistic_coefficients_and_posteriori(X, y, C_priori=1, l1_coeff=0, max_iter=100)[2])
            betas_en = np.hstack(logistic_coefficients_and_posteriori(X, y, C_priori=1, l1_coeff=1, max_iter=100)[2])
            betas_en_2 = np.hstack(logistic_coefficients_and_posteriori(X, y, C_priori=1, l1_coeff=10, max_iter=100)[2])

            assert_almost_equal(betas, [0.563, -0.86], decimal=3)
            assert_almost_equal(betas_en, [0.492, -0.094], decimal=3)
            assert_almost_equal(betas_en_2, [0.476, 0], decimal=3)
            self.assertEqual(betas_en_2[1], 0)

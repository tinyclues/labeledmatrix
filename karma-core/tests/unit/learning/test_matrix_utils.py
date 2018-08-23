import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from karma.core.utils.utils import use_seed
from scipy.sparse import rand
from itertools import product
from karma.learning.matrix_utils import *


class MatrixUtilsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with use_seed(123):
            cls.mat1 = np.arange(-10, 10).reshape(5, 4)
            cls.mat2 = np.arange(-10, 5).reshape(5, 3)

    def test_gram_quantiles(self):
        with use_seed(42):
            sparse_G = KarmaSparse(rand(100, 10, 0.1))

        assert_almost_equal(gram_quantiles(sparse_G, 0.1), [0.])
        assert_almost_equal(gram_quantiles(sparse_G, 0.9), [0.20081272])
        assert_almost_equal(gram_quantiles(sparse_G, [0.1, 0.9]), [0., 0.20081272])

    def test_second_moment(self):
        assert_almost_equal(second_moment(np.array([[1, 2], [-1, 3], [-1, 0]])), [[3., -1.], [-1., 13.]])
        assert_almost_equal(second_moment(KarmaSparse(self.mat1)), second_moment(self.mat1))
        assert_almost_equal(second_moment(KarmaSparse(self.mat2)), second_moment(self.mat2))

    def test_direct_product_second_moment(self):
        assert_almost_equal(direct_product_second_moment([[1, 2], [-1, 3], [-1, 0]], [1, 2, 3]),
                            [[14., -10.], [-10., 40.]])

        expected = direct_product_second_moment(self.mat1, self.mat2)
        for a, b in product([self.mat1, self.mat1.astype(np.float32), KarmaSparse(self.mat1)],
                            [self.mat2, self.mat2.astype(np.float32), KarmaSparse(self.mat2)]):

            if not is_karmasparse(a) and is_karmasparse(b):
                continue
            assert_almost_equal(direct_product_second_moment(a, b), expected)

    def test_direct_product(self):
        # scalar case
        assert_almost_equal(direct_product([3, 4], [1.2, 2]), [[3.6], [8]])

        expected = direct_product(self.mat1, self.mat2)
        for a, b in product([self.mat1, self.mat1.astype(np.float32), KarmaSparse(self.mat1)],
                            [self.mat2, self.mat2.astype(np.float32), KarmaSparse(self.mat2)]):

            if not is_karmasparse(a) and is_karmasparse(b):
                continue
            assert_almost_equal(direct_product(a, b), expected)

    def test_direct_product_dot(self):
        # scalar case
        assert_almost_equal(direct_product_dot([3, 4], [1.2, 2], np.array([2])), [7.2, 16.])

        w = np.random.rand(self.mat1.shape[1] * self.mat2.shape[1])

        expected1 = direct_product_dot(self.mat1, self.mat2, w, 1)
        expected2 = direct_product_dot(self.mat1, self.mat2, w, 2)

        for a, b in product([self.mat1, self.mat1.astype(np.float32), KarmaSparse(self.mat1)],
                            [self.mat2, self.mat2.astype(np.float32), KarmaSparse(self.mat2)]):

            if not is_karmasparse(a) and is_karmasparse(b):
                continue
            assert_almost_equal(direct_product_dot(a, b, w, 1), expected1)
            assert_almost_equal(direct_product_dot(a, b, w, 2), expected2)

    def test_direct_product_dot_transpose(self):
        # scalar case
        assert_almost_equal(direct_product_dot_transpose([3, 4], [1.2, 2], np.array([2, -1])), [-0.8])

        w = np.random.rand(self.mat1.shape[0])

        expected1 = direct_product_dot_transpose(self.mat1, self.mat2, w, 1)
        expected2 = direct_product_dot_transpose(self.mat1, self.mat2, w, 2)

        for a, b in product([self.mat1, self.mat1.astype(np.float32), KarmaSparse(self.mat1)],
                            [self.mat2, self.mat2.astype(np.float32), KarmaSparse(self.mat2)]):

            if not is_karmasparse(a) and is_karmasparse(b):
                continue
            assert_almost_equal(direct_product_dot_transpose(a, b, w, 1), expected1)
            assert_almost_equal(direct_product_dot_transpose(a, b, w, 2), expected2)

    def test_quantile_boundaries(self):
        matrix = np.repeat([0.78, 0.78, 0.2, 0.87, 0.87, 0.87, 0.5, 0.6, 0.9, 0.9], 5).reshape(10, 5)
        sparse_matrix = KarmaSparse(matrix)
        for m in [matrix, sparse_matrix.tocsr(), sparse_matrix.tocsc()]:
            np.testing.assert_equal(quantile_boundaries(m, 4, axis=0), [[0.6] * 5, [0.78] * 5, [0.87] * 5])
            np.testing.assert_equal(quantile_boundaries(m.T, 4, axis=1), [[0.6, 0.78, 0.87]] * 5)

        matrix = np.repeat([0.78, 0.5], 5).reshape(2, 5)
        sparse_matrix = KarmaSparse(matrix)
        for m in [matrix, sparse_matrix.tocsr(), sparse_matrix.tocsc()]:
            np.testing.assert_equal(quantile_boundaries(m, 4, axis=0), [[0.5] * 5, [0.5] * 5])
            np.testing.assert_equal(quantile_boundaries(m.T, 4, axis=1), [[0.5, 0.5]] * 5)

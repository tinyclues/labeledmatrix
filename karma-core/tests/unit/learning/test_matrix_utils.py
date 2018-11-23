import unittest
from itertools import product

from cyperf.matrix.karma_sparse import DTYPE
from numpy.testing import assert_almost_equal, assert_allclose
from scipy.sparse import rand

from karma.core.utils.utils import use_seed
from karma.learning.matrix_utils import *


class MatrixUtilsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with use_seed(123):
            cls.mat1 = np.arange(-10, 10).reshape(5, 4).astype(dtype=DTYPE)
            cls.mat2 = np.arange(-10, 5).reshape(5, 3).astype(dtype=DTYPE)

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

        w = np.random.rand(self.mat1.shape[1] * self.mat2.shape[1]).astype(dtype=DTYPE)

        expected1 = direct_product_dot(self.mat1, self.mat2, w, 1)
        expected2 = direct_product_dot(self.mat1, self.mat2, w, 2)

        for a, b in product([self.mat1, self.mat1.astype(np.float32), KarmaSparse(self.mat1)],
                            [self.mat2, self.mat2.astype(np.float32), KarmaSparse(self.mat2)]):

            if not is_karmasparse(a) and is_karmasparse(b):
                continue
            assert_allclose(direct_product_dot(a, b, w, 1), expected1, 1e-5)
            assert_allclose(direct_product_dot(a, b, w, 2), expected2, 1e-5)

    def test_direct_product_dot_transpose(self):
        # scalar case
        assert_almost_equal(direct_product_dot_transpose([3, 4], [1.2, 2], np.array([2, -1])), [-0.8])

        w = np.random.rand(self.mat1.shape[0]).astype(dtype=DTYPE)

        expected1 = direct_product_dot_transpose(self.mat1, self.mat2, w, 1)
        expected2 = direct_product_dot_transpose(self.mat1, self.mat2, w, 2)

        for a, b in product([self.mat1.astype(np.float32), KarmaSparse(self.mat1)],
                            [self.mat2.astype(np.float32), KarmaSparse(self.mat2)]):

            if not is_karmasparse(a) and is_karmasparse(b):
                continue
            assert_allclose(direct_product_dot_transpose(a, b, w, 1), expected1, 1e-5)
            assert_allclose(direct_product_dot_transpose(a, b, w, 2), expected2, 1e-5)

    def test_quantile_boundaries(self):
        matrix = np.repeat([0.78, 0.78, 0.2, 0.87, 0.87, 0.87, 0.5, 0.6, 0.9, 0.9], 5).reshape(10, 5)
        sparse_matrix = KarmaSparse(matrix)
        for m in [sparse_matrix.tocsr(), sparse_matrix.tocsc()]:
            np.testing.assert_equal(quantile_boundaries(m, 4, axis=0),
                                    np.array([[0.6] * 5, [0.78] * 5, [0.87] * 5], dtype=DTYPE))
            np.testing.assert_equal(quantile_boundaries(m.T, 4, axis=1), np.array([[0.6, 0.78, 0.87]] * 5, dtype=DTYPE))

        matrix = np.repeat([0.78, 0.5], 5).reshape(2, 5)
        sparse_matrix = KarmaSparse(matrix)
        for m in [sparse_matrix.tocsr(), sparse_matrix.tocsc()]:
            np.testing.assert_equal(quantile_boundaries(m, 4, axis=0), np.array([[0.5] * 5, [0.5] * 5], dtype=DTYPE))
            np.testing.assert_equal(quantile_boundaries(m.T, 4, axis=1), np.array([[0.5, 0.5]] * 5, dtype=DTYPE))

    def test_to_array_if_needed_dtype(self):
        matrix = np.random.rand(100, 10)

        for source_dtype, expected_dtype in [(np.bool, np.int32),
                                             (np.int64, np.int64),
                                             (np.uint64, np.float64),
                                             (np.int32, np.int32),
                                             (np.uint32, np.int64),
                                             (np.int16, np.int32),
                                             (np.uint16, np.int32),
                                             (np.float64, np.float64),
                                             (np.float32, np.float64),
                                             (np.float16, np.float64)]:
            self.assertEqual(expected_dtype, to_array_if_needed(matrix.astype(source_dtype), min_dtype=np.int32).dtype,
                             msg={'source_dtype': source_dtype, 'min_dtype': np.int32})

        for source_dtype, expected_dtype in [(np.bool, np.float32),
                                             (np.int64, np.float32),
                                             (np.uint64, np.float32),
                                             (np.int32, np.float32),
                                             (np.uint32, np.float32),
                                             (np.int16, np.float32),
                                             (np.uint16, np.float32),
                                             (np.float64, np.float64),
                                             (np.float32, np.float32),
                                             (np.float16, np.float32)]:
            self.assertEqual(expected_dtype,
                             to_array_if_needed(matrix.astype(source_dtype), min_dtype=np.float32).dtype,
                             msg={'source_dtype': source_dtype, 'min_dtype': np.float32})

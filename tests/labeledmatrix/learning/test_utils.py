import unittest

import numpy as np
from cyperf.matrix.karma_sparse import KarmaSparse, DTYPE
from mock import patch
from numpy.testing import assert_array_equal, assert_almost_equal

from labeledmatrix.core.utils import use_seed
from labeledmatrix.learning.utils import VirtualDirectProduct, BasicVirtualHStack, VirtualHStack, NB_THREADS_MAX


class VirtualHStackTestCase(unittest.TestCase):
    def test_init(self):
        hstack = BasicVirtualHStack([np.ones((5, 2)), np.ones((5, 3)), 2, np.int32(3), np.asarray(2, dtype=np.int8)])
        self.assertTrue(hstack.is_block)
        np.testing.assert_array_equal(hstack.X[0], np.ones((5, 2)))
        np.testing.assert_array_equal(hstack.X[1], np.ones((5, 3)))
        np.testing.assert_array_equal(hstack.X[2], np.zeros((5, 2)))
        np.testing.assert_array_equal(hstack.X[3], np.zeros((5, 3)))
        np.testing.assert_array_equal(hstack.X[4], np.zeros((5, 2)))
        np.testing.assert_array_equal(hstack.dims, [0, 2, 5, 7, 10, 12])
        self.assertEqual(hstack.shape, (5, 12))

        hstack_copy = BasicVirtualHStack(hstack)
        self.assertEqual(id(hstack_copy.X), id(hstack.X))
        self.assertEqual(hstack_copy.is_block, hstack.is_block)
        self.assertEqual(hstack_copy.shape, hstack.shape)

        hstack = BasicVirtualHStack(np.zeros((15, 8)))
        self.assertEqual(hstack.shape, (15, 8))

        hstack_copy = BasicVirtualHStack(hstack)
        self.assertEqual(id(hstack_copy.X), id(hstack.X))
        self.assertEqual(hstack_copy.is_block, hstack.is_block)
        self.assertEqual(hstack_copy.shape, hstack.shape)

        with self.assertRaises(ValueError) as e:
            BasicVirtualHStack([])
        self.assertEqual('Cannot create a VirtualHStack of an empty list',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            BasicVirtualHStack([np.ones((5, 2)), 'a'])
        self.assertEqual('Cannot create a VirtualHStack with <type \'str\'>',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            BasicVirtualHStack([np.ones((5, 2)), np.ones((6, 3))])
        self.assertEqual('Cannot create a VirtualHStack for an array of length 6 while other array has length 5',
                         str(e.exception))

    def test_split_by_dims(self):
        hstack = BasicVirtualHStack([np.ones((5, 2)), np.ones((5, 3)), 1])
        self.assertEqual(map(len, hstack.split_by_dims(np.ones(6))), [2, 3, 1])
        with self.assertRaises(AssertionError):
            hstack.split_by_dims(np.ones(7))

    def test_adjust_array_to_total_dimension(self):
        hstack = BasicVirtualHStack([np.ones((5, 2)), np.ones((5, 3)), 1])
        np.testing.assert_array_equal(hstack.adjust_array_to_total_dimension(5), [5] * 6)
        np.testing.assert_array_equal(hstack.adjust_array_to_total_dimension([5] * 3 + [6] * 3), [5] * 3 + [6] * 3)
        np.testing.assert_array_equal(hstack.adjust_array_to_total_dimension([[1, 2], 5, 6]), [1, 2, 5, 5, 5, 6])

        with self.assertRaises(ValueError) as e:
            hstack.adjust_array_to_total_dimension([5] * 5)
        self.assertEqual('parameter is invalid: expected float or an array-like of length 6 or 3, '
                         'got array-like of length 5',
                         str(e.exception))

        hstack = BasicVirtualHStack(np.zeros((15, 8)))
        with self.assertRaises(ValueError) as e:
            hstack.adjust_array_to_total_dimension([5] * 5, 'tt')
        self.assertEqual('parameter \'tt\' is invalid: expected float or an array-like of length 8, '
                         'got array-like of length 5',
                         str(e.exception))

    def test_adjust_to_block_dimensions(self):
        hstack = BasicVirtualHStack([np.ones((5, 2)), np.ones((5, 3)), 1])
        self.assertEqual(map(len, hstack.adjust_to_block_dimensions([[1, 2], 5, 6])), [2, 3, 1])
        with self.assertRaises(ValueError):
            hstack.adjust_to_block_dimensions(np.ones(7))

    def test_promote_types(self):
        hstack = BasicVirtualHStack([np.ones((5, 2), dtype=np.float16),
                                     np.full((5, 1), 3, dtype=np.float32),
                                     np.ones((5, 3), dtype=np.float64),
                                     1])
        self.assertEqual(hstack.X[0].dtype, np.float32)  # dtype is promoted
        self.assertEqual(hstack.X[1].dtype, np.float32)
        self.assertEqual(hstack.X[2].dtype, np.float64)
        self.assertEqual(hstack.X[3].dtype, np.float64)
        self.assertEqual(hstack.materialize().dtype, np.float64)
        np.testing.assert_array_equal(hstack.materialize(), [[1, 1, 3, 1, 1, 1, 0]] * 5)

        hstack = BasicVirtualHStack(np.ones((5, 2), dtype=np.float16))
        self.assertEqual(hstack.X.dtype, np.float32)  # dtype is promoted
        self.assertEqual(hstack.materialize().dtype, np.float32)

    def test_set_parallelism(self):
        hstack = VirtualHStack([np.ones((5, 2), dtype=np.float16),
                                np.full((5, 1), 3, dtype=np.float32),
                                np.ones((5, 3), dtype=np.float64),
                                1], nb_threads=0, nb_inner_threads=4)

        self.assertEqual(hstack.nb_threads, 0)
        self.assertEqual(hstack.nb_inner_threads, 4)
        self.assertTrue(hstack.is_block)
        self.assertTrue(hstack.pool is None)

        hstack = VirtualHStack([np.ones((5, 2), dtype=np.float16)],
                               nb_threads=4, nb_inner_threads=4)

        self.assertEqual(hstack.nb_threads, 1)
        self.assertEqual(hstack.nb_inner_threads, 4)
        self.assertTrue(hstack.is_block)
        self.assertTrue(hstack.pool is None)

        hstack = VirtualHStack([np.ones((5, 2), dtype=np.float16),
                                np.full((5, 1), 3, dtype=np.float32)],
                               nb_threads=4)
        self.assertTrue(hstack.is_block)
        self.assertTrue(hstack.pool is not None)
        self.assertEqual(hstack.nb_threads, 2)
        self.assertEqual(hstack.nb_inner_threads, NB_THREADS_MAX)

    def test_row_nnz(self):
        rand = lambda x: np.random.randint(25, size=(10, x))
        for i in range(20):
            blocks = [
                rand(3),
                KarmaSparse(rand(5)),
                VirtualDirectProduct(KarmaSparse(rand(2)), rand(7)),
                2  # zeros
            ]
            X = VirtualHStack(blocks).materialize()
            row_nnz = np.count_nonzero(X) / 10.
            # we need the hack below to take into account the fact that VirtualDirectProduct is an approximation
            row_nnz_adjusted = row_nnz + (blocks[2].nnz - blocks[2].materialize().nnz) / 10.
            # FIXME
            with patch('karma.learning.utils.VirtualDirectProduct.materialize') as mock_materialize:
                np.testing.assert_almost_equal(row_nnz_adjusted, VirtualHStack(blocks).row_nnz)
                np.testing.assert_almost_equal(row_nnz_adjusted,
                                               VirtualHStack(blocks, nb_threads=2, nb_inner_threads=4).row_nnz)
                np.testing.assert_almost_equal(row_nnz, VirtualHStack(X).row_nnz)
            self.assertEqual(0, mock_materialize.call_count)


class VirtualDirectProductTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with use_seed(124567):
            cls.vdp = VirtualDirectProduct(KarmaSparse(np.random.randint(-1, 2, size=(10, 2)),
                                                       shape=(10, 2)),
                                           (2 * np.random.rand(10, 3) - 1).astype(DTYPE))
            cls.vdpt = cls.vdp.T
            cls.dp = cls.vdp.materialize()
            cls.dpt = cls.vdp.materialize().T

    def test_virtual_direct_product_method(self):
        self.assertIsInstance(self.dp, KarmaSparse)
        self.assertFalse(self.vdp.is_transposed)

        self.assertEqual(self.vdp.shape, self.dp.shape)
        self.assertEqual(self.vdp.dtype, self.dp.dtype)
        self.assertEqual(self.vdp.nnz, self.dp.nnz)
        self.assertEqual(self.vdp.ndim, self.dp.ndim)
        self.assertEqual(self.vdp.min(), self.dp.min())
        self.assertEqual(self.vdp.max(), self.dp.max())

        ind = [4, 3, 4, -1]
        assert_array_equal(self.vdp[ind], self.dp[ind])

        assert_almost_equal(self.vdp.sum(), self.dp.sum(), 7)
        assert_almost_equal(self.vdp.sum(axis=0), self.dp.sum(axis=0), 6)
        assert_almost_equal(self.vdp.sum(axis=1), self.dp.sum(axis=1), 6)

        w = np.random.rand(self.vdp.shape[1])
        assert_almost_equal(self.vdp.dot(w), self.dp.dot(w), 6)

        w = np.random.rand(self.vdp.shape[0])
        assert_almost_equal(self.vdp.transpose_dot(w), self.dp.dense_vector_dot_left(w), 6)

        assert_almost_equal(self.vdp.second_moment(), self.dp.T.dot(self.dp), 6)

    def test_virtual_direct_product_transpose_method(self):
        self.assertTrue(self.vdpt.is_transposed)

        self.assertEqual(self.vdpt.shape, self.dpt.shape)
        self.assertEqual(self.vdpt.dtype, self.dpt.dtype)
        self.assertEqual(self.vdpt.nnz, self.dpt.nnz)
        self.assertEqual(self.vdpt.ndim, self.dpt.ndim)

        assert_almost_equal(self.vdp.sum(), self.vdpt.sum(), 5)
        assert_almost_equal(self.vdpt.sum(), self.dpt.sum(), 5)
        assert_almost_equal(self.vdpt.sum(axis=0), self.dpt.sum(axis=0), 6)
        assert_almost_equal(self.vdpt.sum(axis=1), self.dpt.sum(axis=1), 6)

        w = np.random.rand(self.vdpt.shape[1])
        assert_almost_equal(self.vdpt.dot(w), self.dpt.dot(w), 6)

        w = np.random.rand(self.vdpt.shape[0])
        assert_almost_equal(self.vdpt.transpose_dot(w), self.dpt.dense_vector_dot_left(w), 6)

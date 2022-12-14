import unittest

import numpy as np
import scipy.sparse as sp
from cyperf.matrix.karma_sparse import KarmaSparse
from numpy.testing import assert_allclose

from labeledmatrix.learning.matrix_utils import kl_div
from labeledmatrix.learning.nmf import nmf, nmf_fold, NMF, GNMF
from labeledmatrix.core.utils import use_seed


def shuffle_indices(sp_matrix):
    mat = sp_matrix.copy()
    for i in range(mat.indptr.shape[0] - 1):
        r = np.arange(mat.indptr[i], mat.indptr[i + 1])
        np.random.shuffle(r)
        mat.indices[mat.indptr[i]: mat.indptr[i + 1]] = mat.indices[r]
        mat.data[mat.indptr[i]: mat.indptr[i + 1]] = mat.data[r]
        mat._sorted_indices = False
    return mat


class LibNMFTestCase(unittest.TestCase):
    def assert_almost_binary_array(self, arr):
        self.assertIsInstance(arr, np.ndarray)
        self.assertTrue(np.sum(arr) > 0.5)
        np.testing.assert_almost_equal(np.abs(arr - 0.5 * np.ones(arr.shape)), 0.5 * np.ones(arr.shape))

    def test_shape(self):
        matrix = np.random.rand(15, 10)
        w, h = nmf(matrix, rank=4, max_iter=4)
        self.assertEqual(w.shape, (15, 4))
        self.assertEqual(h.shape, (10, 4))

    def test_one_dim(self):
        matrix = np.random.rand(1, 10)
        w, h = nmf(matrix, rank=None)
        self.assertEqual(w.shape, (1, 1))
        self.assertEqual(h.shape, (10, 1))

    def test_dtype(self):
        for dtype in [np.float64, np.float32]:
            matrix = np.random.rand(3, 10).astype(dtype)
            w, h = nmf(matrix, rank=None)
            self.assertEqual(w.dtype, dtype)
            self.assertEqual(h.dtype, dtype)

        matrix = (100 * np.random.rand(3, 10)).astype(np.int)
        w, h = nmf(matrix, rank=None)
        self.assertEqual(w.dtype, np.float32)
        self.assertEqual(h.dtype, np.float32)

        matrix = KarmaSparse(np.random.rand(3, 10))
        w, h = nmf(matrix, rank=None)
        self.assertEqual(w.dtype, matrix.dtype)
        self.assertEqual(h.dtype, matrix.dtype)

    def test_reconstruction(self):
        with use_seed(120):
            matrix = np.dot(np.random.poisson(2, size=(10, 4)),
                            np.random.poisson(6, size=(4, 10)))
        matrix[0, 0] = 0
        matrix[-1, -1] = 0
        seed = int(matrix.sum())
        # dense
        w_d, h_d = nmf(matrix, rank=4, max_iter=500, seed=seed)
        # sparse
        w_s, h_s = nmf(KarmaSparse(matrix), rank=4, max_iter=500, seed=seed)

        self.assertLessEqual(kl_div(matrix, np.dot(w_d, h_d.T)), np.max(matrix) / 10.)
        self.assertLessEqual(np.max(np.abs(matrix - np.dot(w_d, h_d.T))), np.max(matrix) / 10.)
        self.assertLessEqual(kl_div(matrix, np.dot(w_s, h_s.T)), np.max(matrix) / 10.)
        self.assertLessEqual(np.max(np.abs(matrix - np.dot(w_s, h_s.T))), np.max(matrix) / 10.)

        # we got almost the same results (up to float32 / float64 differences)
        assert_allclose(w_d, w_s, 1e-4)
        assert_allclose(h_d, h_s, 1e-4)

    def test_seed(self):
        with use_seed(100):
            matrix = np.dot(np.random.poisson(2, size=(10, 4)),
                            np.random.poisson(6, size=(4, 10)))
        matrix[0, 0] = 0
        matrix[-1, -1] = 0
        seed = int(matrix.sum())
        w_1, h_1 = nmf(matrix, rank=4, max_iter=500, seed=seed)
        np.random.rand()
        w_2, h_2 = nmf(matrix, rank=4, max_iter=500, seed=seed)
        np.random.rand()
        w_3, h_3 = nmf(matrix, rank=4, max_iter=500, seed=None)

        # we got exactly the same results for 1 and 2
        self.assertLessEqual(np.max(np.abs(w_1 - w_2) + np.abs(h_1 - h_2)), 10 ** -10)
        # but not necessarily for 1 and 3
        self.assertGreaterEqual(np.max(np.abs(w_1 - w_3) + np.abs(h_1 - h_3)), 10 ** -10)

    def test_sparse(self):
        sp_mat = KarmaSparse(np.array([[0, 0, 0.1], [2, 1, 0], [1, 2, 0]]), format="csc")
        w1, h1 = nmf(sp_mat, rank=2, max_iter=100, seed=2000)
        self.assertLessEqual(np.max(np.abs(sp_mat.toarray() - np.dot(w1, h1.transpose()))), 0.15)

    def test_rank_detection(self):
        with use_seed(3335):
            matrix = np.dot(np.random.zipf(1.5, size=(30, 10)), np.random.zipf(2, size=(10, 30)))
        seed = int(matrix.sum())
        w1, h1 = nmf(matrix, rank=None, seed=seed)
        self.assertLessEqual(np.mean(np.abs(matrix - np.dot(w1, h1.T))) / np.mean(matrix), 0.01)
        self.assertTrue(w1.shape[1] in range(7, 14))
        # using that value for factorization
        w, h = nmf(matrix, rank=w1.shape[1], seed=seed)
        self.assertLessEqual(kl_div(matrix, np.dot(w1, h1.T)), 1.2 * kl_div(matrix, np.dot(w, h.T)))

    def test_persistent_wrt_initial_data(self):
        matrix_csr = KarmaSparse(sp.rand(20, 10, 0.1, format="csr"))
        matrix_csc = matrix_csr.tocsc()
        matrix_shuffled = shuffle_indices(matrix_csr.to_scipy_sparse())
        self.assertTrue(np.all(matrix_shuffled.toarray() == matrix_csr.toarray()))

        matrix_dense = matrix_csr.toarray()
        rank = 5
        w0, h0 = np.random.rand(20, rank), np.random.rand(rank, 10)
        # csr
        nmf_csr = NMF(matrix_csr, rank)
        nmf_csr.factorisation_initial(w0, h0)
        nmf_csr.iterate(1)
        w_csr, h_csr = nmf_csr.w, nmf_csr.h
        # csc
        nmf_csc = NMF(matrix_csc, rank)
        nmf_csc.factorisation_initial(w0, h0)
        nmf_csc.iterate(1)
        w_csc, h_csc = nmf_csc.w, nmf_csc.h
        # dense
        nmf_dense = NMF(matrix_dense, rank)
        nmf_dense.factorisation_initial(w0, h0)
        nmf_dense.iterate(1)
        w_dense, h_dense = nmf_dense.w, nmf_dense.h
        # csr_shuflled
        nmf_shuffled = NMF(matrix_shuffled, rank)
        nmf_shuffled.factorisation_initial(w0, h0)
        nmf_shuffled.iterate(1)
        w_shuffled, h_shuffled = nmf_shuffled.w, nmf_shuffled.h
        # test
        self.assertLessEqual(np.abs(w_csc - w_shuffled).sum(), 10 ** -8)
        self.assertLessEqual(np.abs(h_csc - h_shuffled).sum(), 10 ** -8)
        self.assertLessEqual(np.abs(w_csc - w_csr).sum(), 10 ** -8)
        self.assertLessEqual(np.abs(w_dense - w_csr).sum(), 10 ** -4)
        self.assertLessEqual(np.abs(h_csc - h_csr).sum(), 10 ** -8)
        self.assertLessEqual(np.abs(h_dense - h_csr).sum(), 10 ** -4)
        # it has moved from initial data
        self.assertGreaterEqual(np.abs(h_dense - h0).sum(), 10 ** -4)

    def test_nmf_fold(self):
        with use_seed(100):
            l, r = np.random.rand(10, 3), np.random.rand(3, 6)
        m = l.dot(r)
        ll = nmf_fold(m, r, 100)
        lll = nmf_fold(m, r, 1)
        self.assertLessEqual(np.mean(np.abs(ll - l)), 0.01)
        self.assertGreaterEqual(np.mean(np.abs(lll - l)), 0.01)

        # sparse matrix
        with use_seed(100):
            l = np.random.randint(0, 2, size=(10, 5))
            r = np.random.randint(0, 2, size=(5, 7))
        m = KarmaSparse(l.dot(r))
        ll = nmf_fold(m, r, 400)
        self.assertLessEqual(np.mean(np.abs(ll - l)), 0.01)

        lll = nmf_fold(m, KarmaSparse(r), 400)
        self.assertTrue(np.allclose(lll, ll))

    def test_iterate(self):
        with use_seed(100):
            w, h = np.random.rand(1000, 5), np.random.rand(5, 20)
        m = w.dot(h)
        m_sparse = KarmaSparse(m).truncate_by_count(3, axis=1)
        for matrix in [m, m_sparse]:
            for metric in ['KL', 'euclid']:
                with use_seed(25010):
                    nmf_solver = NMF(matrix, 5, metric=metric)
                    nmf_solver.smart_init(svd_init=True)
                    dist_init = nmf_solver.dist()
                    nmf_solver.iterate(10)
                    self.assertLess(nmf_solver.dist(), dist_init / 1.2)

    def test_graph_nmf(self):
        with use_seed(100):
            w, h = np.random.rand(1000, 5), np.random.rand(5, 20)
        m = w.dot(h)
        m_sparse = KarmaSparse(m).truncate_by_count(3, axis=1)

        one_edge = np.eye(1000)
        one_edge[0, 1] = one_edge[1, 0] = 1
        one_edge = KarmaSparse(one_edge)

        for matrix in [m, m_sparse]:
            for metric in ['KL', 'euclid']:
                with use_seed(25010):
                    nmf_solver = GNMF(matrix, 5, metric=metric, adjacency=10000 * one_edge)
                    nmf_solver.smart_init(svd_init=True)
                    dist_init = nmf_solver.dist()
                    nmf_solver.iterate(10)
                    self.assertLess(nmf_solver.dist(), dist_init / 1.2)
                    np.testing.assert_array_almost_equal(nmf_solver.w[0], nmf_solver.w[1], decimal=3)

#
# Copyright tinyclues, All rights reserved
#

import numpy as np
from scipy.sparse import isspmatrix as is_scipy_sparse
from cyperf.matrix.karma_sparse import KarmaSparse, is_karmasparse, DTYPE
from karma.learning.matrix_utils import safe_dot, idiv, normalize, truncate_by_count, nonzero_mask
from numexpr import evaluate

__all__ = ['co_clustering']


def co_clustering(matrix, ranks=(2, 2), max_iter=50, nb_preruns=20, pre_iter=4):
    """
    >>> matrix = np.array([[5, 5, 5, 0, 0, 0],
    ...                    [5, 5, 5, 0, 0, 0],
    ...                    [0, 0, 0, 5, 5, 5],
    ...                    [0, 0, 0, 5, 5, 5],
    ...                    [4, 4, 0, 4, 4, 4],
    ...                    [4, 4, 4, 0, 4, 4]])
    >>> import scipy.sparse as sp
    >>> if np.random.rand() > 0.5:
    ...     matrix = KarmaSparse(matrix)
    >>> wmap, hmap = co_clustering(matrix, ranks=[3, 2], max_iter=10)
    >>> (wmap[0] == wmap[1]) and (wmap[2] == wmap[3]) and \
        (wmap[4] == wmap[5]) and (wmap[5] != wmap[3] != wmap[1])
    True
    >>> (hmap[0] == hmap[1] == hmap[2]) and \
        (hmap[3] == hmap[4] == hmap[5]) and (hmap[5] != hmap[2])
    True
    >>> from scipy.linalg import block_diag
    >>> matrix = block_diag(np.random.rand(100, 10), np.random.rand(50, 20))
    >>> matrix = matrix + np.random.rand(100+50, 10+20) / 5.
    >>> wmap, hmap = co_clustering(sp.csr_matrix(matrix), ranks=[2, 2], max_iter=4)
    >>> len(wmap[wmap == 1.0]) in [50, 100]
    True
    >>> len(hmap[hmap == 1.0]) in [10, 20]
    True
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> from sklearn.metrics import euclidean_distances
    >>> from sklearn.metrics.cluster import normalized_mutual_info_score
    >>> centers = [[1, 1], [-1, -1], [1, -1]]
    >>> X, labels_true = make_blobs(n_samples=200, centers=centers,
    ...                             cluster_std=0.3, random_state=0)
    >>> matrix = euclidean_distances(X)
    >>> if np.random.rand() > 0.5:
    ...     matrix = KarmaSparse(matrix)
    >>> np.random.seed(10)
    >>> wmap, hmap = co_clustering(matrix, ranks=[3, 3], max_iter=30)
    >>> normalized_mutual_info_score(labels_true, wmap) > 1 - 10 ** (-5)
    True

    """
    clust = CoClustering(matrix, ranks)
    clust.smart_run(nb_preruns, pre_iter, max_iter)
    return clust.w.indices, clust.h.indices


class CoClustering(object):
    def __init__(self, matrix, ranks):
        self._epsilon = 1e-10

        if is_scipy_sparse(matrix):
            matrix = KarmaSparse(matrix, format="csr", copy=False)

        if matrix.min() < 0:
            raise ValueError("in matrix, entries should be positives")
        if matrix.sum(axis=1).min() == 0:
            raise ValueError("in matrix, not all entries in a row can be zero")
        if matrix.sum(axis=0).min() == 0:
            raise ValueError("in matrix, not all entries in a column can be zero")

        self.n, self.m = matrix.shape
        if ranks[0] > self.n or ranks[1] > self.m:
            print('ranks {0} is large than matrix shape {1}'.format(ranks, (self.n, self.m)))

        self.ranks = (min(ranks[0], self.n), min(ranks[1], self.m))
        self.matrix = matrix.clip(self._epsilon)
        self.set_marginal()

    def set_marginal(self):
        # a priory marginal
        self.p_x = self.matrix.sum(axis=1)
        self.p_y = self.matrix.sum(axis=0)
        self.pcond_y_vs_x = normalize(self.matrix, norm='l1', axis=1)
        self.pcond_x_vs_y = normalize(self.matrix, norm='l1', axis=0)

        if is_karmasparse(self.matrix):
            self.pcond_y_vs_x = self.pcond_y_vs_x.tocsr()
            self.pcond_x_vs_y = self.pcond_x_vs_y.tocsc()

    def smart_run(self, nb_preruns, pre_iter, max_iter):
        self.initial_clustering()
        self.iterate(pre_iter)
        best_w, best_h = self.w, self.h
        best_dist = idiv(self.matrix, self.approximate_matrix())  # To fix the common mask

        for _ in xrange(nb_preruns):
            self.initial_clustering()
            self.iterate(pre_iter)
            cand_dist = idiv(self.matrix, self.approximate_matrix())
            if cand_dist < best_dist:
                best_dist = cand_dist
                best_w, best_h = self.w, self.h

        # True main loop
        self.initial_clustering(w=best_w, h=best_h)
        self.iterate(max_iter)

    def initial_clustering(self, w=None, h=None):
        if w is None:
            w_indices = np.random.randint(0, self.ranks[0], self.n)
            self.w = KarmaSparse((np.ones(self.n, dtype=DTYPE), w_indices, np.arange(self.n + 1)),
                                 shape=((self.n, self.ranks[0])),
                                 format="csr", copy=False, has_canonical_format=True)
        else:
            self.w = w
        self.w_old = self.w

        if h is None:
            h_indices = np.random.randint(0, self.ranks[1], self.m)
            self.h = KarmaSparse((np.ones(self.m, dtype=DTYPE), h_indices, np.arange(self.m + 1)),
                                 shape=((self.m, self.ranks[1])),
                                 format="csr", copy=False, has_canonical_format=True)
        else:
            self.h = h
        self.h_old = self.h

    def iterate(self, max_iter):
        for _ in xrange(max_iter):
            self.idiv_update_right()
            self.idiv_update_left()
            if self.check_convergence():
                break

    def check_convergence(self):
        if np.all(self.w_old.indices == self.w.indices) and np.all(self.h_old.indices == self.h.indices):
            return True
        else:
            self.w_old, self.h_old = self.w, self.h
            return False

    def idiv_update_right(self):
        self.get_hat_matrix()
        self.get_cond_y_vs_haty()
        logg = self.pcond_y_vs_haty.dot(normalize(self.hat_matrix, norm='l1', axis=1).transpose())
        evaluate('log(logg)', out=logg)
        self.w.indices[:] = np.argmax(self.pcond_y_vs_x.dot(logg), axis=1)

    def idiv_update_left(self):
        self.get_hat_matrix()
        self.get_cond_x_vs_hatx()
        logg = self.pcond_x_vs_hatx.dot(normalize(self.hat_matrix, norm='l1', axis=0))
        evaluate('log(logg)', out=logg)
        self.h.indices[:] = np.argmax(self.pcond_x_vs_y.transpose().dot(logg), axis=1)

    def get_hat_matrix(self):
        # we should use logic from
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.multi_dot.html
        if self.n >= self.m:
            self.hat_matrix = self.w.transpose().dot(self.matrix.dot(self.h))
        else:
            self.hat_matrix = self.w.transpose().dot(self.matrix).dot(self.h)

        self.hat_matrix = np.asarray(self.hat_matrix)
        self.hat_matrix.clip(self._epsilon, out=self.hat_matrix)

    def get_cond_x_vs_hatx(self):
        self.pcond_x_vs_hatx = self.w.copy()
        self.pcond_x_vs_hatx.data[:] = self.p_x / self.hat_matrix.sum(axis=1)[self.pcond_x_vs_hatx.indices]

    def get_cond_y_vs_haty(self):
        self.pcond_y_vs_haty = self.h.copy()
        self.pcond_y_vs_haty.data[:] = self.p_y / self.hat_matrix.sum(axis=0)[self.pcond_y_vs_haty.indices]

    def approximate_matrix(self):
        self.get_hat_matrix()
        self.get_cond_x_vs_hatx()
        self.get_cond_y_vs_haty()
        if not is_karmasparse(self.matrix):
            return self.pcond_x_vs_hatx.dot(self.hat_matrix).dot(self.pcond_y_vs_haty.transpose())
        elif self.n <= self.m:  # to write condition correctly
            return safe_dot(self.pcond_x_vs_hatx.dot(self.hat_matrix),
                            self.pcond_y_vs_haty.transpose(),
                            self.matrix)
        else:
            return safe_dot(self.pcond_x_vs_hatx,
                            self.pcond_y_vs_haty.dot(self.hat_matrix.transpose()).transpose(),
                            self.matrix)

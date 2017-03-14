#
# Copyright tinyclues, All rights reserved
#

import numpy as np
from scipy.sparse import isspmatrix as is_scipy_sparse
from cyperf.matrix.karma_sparse import KarmaSparse, is_karmasparse
from karma.learning.matrix_utils import safe_dot, safe_argmax, safe_sum, safe_min, idiv
from karma.learning.matrix_utils import normalize

__all__ = ['co_clustering']


def co_clustering(matrix, ranks=[2, 2], max_iter=50, nb_preruns=20, pre_iter=4):
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

    ### >>> n = 100000
    ### >>> matrix = sp.rand(n, n, 0.0001) + sp.eye(n)
    ### >>> wmap, hmap = co_clustering(matrix, ranks=[20, 20], max_iter=50)
    ### >>> matrix = block_diag(np.random.rand(1000, 1000),
    ### ...              np.random.rand(1000, 1000), np.random.rand(1000, 1000))
    ### ...               + np.random.rand(3000, 3000)/2.
    ### >>> import time
    ### >>> t0 = time.time()
    ### >>> wmap, hmap = co_clustering(matrix, ranks=[2, 2],
    ### ...                max_iter=10, nb_preruns=0, pre_iter=0)
    ### >>> time.time() - t0
    ### 1.2s
    """
    clust = CoClustering(matrix, ranks)
    clust.smart_run(nb_preruns, pre_iter, max_iter)
    return clust.wmap, clust.hmap


class CoClustering(object):
    def __init__(self, matrix, ranks):
        self._epsilon = 10 ** (-10)
        if is_scipy_sparse(matrix):
            matrix = KarmaSparse(matrix, format="csr")
        if safe_min(matrix) < 0:
            raise ValueError("in tensor, entries should be positives")
        self.n, self.m = matrix.shape
        if (ranks[0] > self.n) or (ranks[1] > self.m):
            print 'ranks {0} is large than matrix shape {1}'.format(ranks, (self.n, self.m))
        self.ranks = [min(ranks[0], self.n), min(ranks[1], self.m)]
        if np.min(matrix.sum(axis=1)) == 0:
            raise ValueError("in tensor, not all entries in a row can be zero")
        if np.min(matrix.sum(axis=0)) == 0:
            raise ValueError("in tensor, not all entries in a column can be zero")
        self.matrix = matrix.clip(self._epsilon)
        self.set_marginal()

    def set_marginal(self):
        # a priory marginal
        self.p_x = safe_sum(self.matrix, axis=1)
        self.p_y = safe_sum(self.matrix, axis=0)
        self.pcond_y_vs_x = normalize(self.matrix, norm='l1', axis=1)
        self.pcond_x_vs_y = normalize(self.matrix, norm='l1', axis=0)

    def smart_run(self, nb_preruns, pre_iter, max_iter):
        self.initial_clustering()
        self.iterate(pre_iter)
        best_wmap = self.wmap
        best_hmap = self.hmap
        best_dist = idiv(self.matrix, self.approximate_matrix())  # To fix the common mask
        for prerun in xrange(nb_preruns):
                self.initial_clustering()
                self.iterate(pre_iter)
                cand_dist = idiv(self.matrix, self.approximate_matrix())
                if cand_dist < best_dist:
                    best_dist, best_wmap, best_hmap = cand_dist, self.wmap, self.hmap
        # True main loop
        self.initial_clustering(wmap=best_wmap, hmap=best_hmap)
        self.iterate(max_iter)

    def initial_clustering(self, wmap=None, hmap=None):
        if wmap is None:
            self.wmap = np.random.randint(0, self.ranks[0], self.n)
        else:
            self.wmap = wmap
        self.old_wmap = self.wmap

        if hmap is None:
            self.hmap = np.random.randint(0, self.ranks[1], self.m)
        else:
            self.hmap = hmap
        self.old_hmap = self.hmap
        self.update_initial()

    def update_initial(self):
        self.matrix_convert_w()
        self.matrix_convert_h()

    def iterate(self, max_iter):
        for _ in xrange(max_iter):
            self.idiv_update()
            if self.check_convergence():
                break

    def idiv_update(self):
        self.idiv_update_right()
        self.idiv_update_left()

    def idiv_update_right(self):
        self.get_hat_matrix()
        self.get_cond_y_vs_haty()
        logg = np.log(safe_dot(self.pcond_y_vs_haty,
                               normalize(self.hat_matrix, norm='l1', axis=1).transpose()))
        self.wmap = safe_argmax(safe_dot(self.pcond_y_vs_x, logg), axis=1)
        self.matrix_convert_w()

    def idiv_update_left(self):
        self.get_hat_matrix()
        self.get_cond_x_vs_hatx()
        logg = np.log(safe_dot(self.pcond_x_vs_hatx,
                               normalize(self.hat_matrix, norm='l1', axis=0)))
        self.hmap = safe_argmax(safe_dot(self.pcond_x_vs_y.transpose(), logg), axis=1)
        self.matrix_convert_h()

    def check_convergence(self):
        if np.all(self.old_wmap == self.wmap) and np.all(self.old_hmap == self.hmap):
            return True
        else:
            self.old_wmap = self.wmap
            self.old_hmap = self.hmap
            return False

    # utils functions below
    def matrix_convert_w(self):
        self.w = np.zeros((self.n, self.ranks[0]), dtype=np.ubyte)
        self.w[np.arange(self.n), self.wmap] = 1

    def matrix_convert_h(self):
        self.h = np.zeros((self.m, self.ranks[1]), dtype=np.ubyte)
        self.h[np.arange(self.m), self.hmap] = 1

    def get_hat_matrix(self):
        self.hat_matrix = safe_dot(safe_dot(self.w.transpose(), self.matrix), self.h)
        self.hat_matrix = self.hat_matrix.clip(self._epsilon)

    def get_cond_x_vs_hatx(self):
        self.pcond_x_vs_hatx = np.zeros((self.n, self.ranks[0]))
        self.pcond_x_vs_hatx[np.arange(self.n), self.wmap] = \
            self.p_x / self.hat_matrix.sum(axis=1)[self.wmap]

    def get_cond_y_vs_haty(self):
        self.pcond_y_vs_haty = np.zeros((self.m, self.ranks[1]))
        self.pcond_y_vs_haty[np.arange(self.m), self.hmap] = \
            self.p_y / self.hat_matrix.sum(axis=0)[self.hmap]

    def approximate_matrix(self):
        self.get_hat_matrix()
        self.get_cond_x_vs_hatx()
        self.get_cond_y_vs_haty()
        if not is_karmasparse(self.matrix):
            return safe_dot(safe_dot(self.pcond_x_vs_hatx, self.hat_matrix),
                            self.pcond_y_vs_haty.transpose())
        elif self.ranks[0] >= self.ranks[1]:
            return safe_dot(safe_dot(self.pcond_x_vs_hatx, self.hat_matrix),
                            self.pcond_y_vs_haty.transpose(),
                            self.matrix)
        else:
            return safe_dot(self.pcond_x_vs_hatx,
                            safe_dot(self.pcond_y_vs_haty,
                                     self.hat_matrix.transpose()).transpose(),
                            self.matrix)

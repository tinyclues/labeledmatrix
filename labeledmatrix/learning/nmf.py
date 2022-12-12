#
# Copyright tinyclues, All rights reserved
#
import time

import numpy as np
from scipy.sparse import isspmatrix as is_scipy_sparse
from scipy.sparse.linalg import cg
from cyperf.matrix.karma_sparse import KarmaSparse, is_karmasparse, ks_diag

from .matrix_utils import kl_div, normalize, safe_dot, safe_min, cast_2dim_float32_transpose
from .randomize_svd import nmf_svd_init
from labeledmatrix.core.random import use_seed


ADD_TIME = 20
EPSILON = 10 ** (-10)

__all__ = ['nmf', 'nmf_fold']


@use_seed()
def nmf(matrix, rank=20, max_model_rank=100, max_iter=150, svd_init=False, verbose=False):
    """
    matrix should be non-negative numpy.array (dim = 2) with non-zero columns.
    if n,m = matrix.shape then nmf(matrix, rank)
    returns two non-negative matrices of shapes (n,rank) and (m,rank)
    respectively

    Example: ::
        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> matrix = np.random.rand(15, 10)  # matrix = KarmaSparse(matrix)
        >>> w, h = nmf(matrix, rank=4, max_iter=4)
        >>> w.shape
        (15, 4)
        >>> h.shape
        (10, 4)
    """
    if is_scipy_sparse(matrix):
        matrix = KarmaSparse(matrix)
    if verbose:
        print(f"NMF : Matrix dimensions are {matrix.shape}")
        if is_karmasparse(matrix):
            print(f"NMF will use KarmaSparse Matrix with density={matrix.density}")
    max_model_rank = min(max_model_rank or 100, min(matrix.shape))

    if rank is None:
        # nmf with coordinate selection via poisson AIC
        result = NMF_P(matrix, max_model_rank)
        result.precision_initial()
        result.smart_init(svd_init=svd_init)
        result.iterate()
        rank = result.get_estimate()
        if verbose:
            print(f'NMF Guesser detected **rank** == {rank}')
        ind = np.argsort(result.beta)[:rank]
        # Change updates and reduce factors
        result.rank = rank
        result.factorisation_initial(w=result.w[:, ind], h=result.h[ind])
        # reset to MNF iteration (to avoid coping matrix)
        result.iterate = super(NMF_P, result).iterate
        result.brunet_update = super(NMF_P, result).brunet_update
        result.brunet_left = super(NMF_P, result).brunet_left
        result.brunet_right = super(NMF_P, result).brunet_right
        result.iterate(max_iter // 3)
    else:
        result = NMF(matrix, rank=rank)
        result.smart_init(svd_init=svd_init)
        result.iterate(max_iter)

    if verbose:
        print(f"NMF : Used rank is equal to {result.rank}")

    # renormalizing
    diag = np.atleast_2d(result.h.sum(axis=1))
    h = result.h.T / diag
    w = diag * result.w  # w rows will have the norm of origin matrix rows
    return w, h


class NMF():
    def __init__(self, matrix, rank, metric="KL", renorm=False, verbose=False):
        self.epsilon = EPSILON
        if is_scipy_sparse(matrix):
            matrix = KarmaSparse(matrix)
        if safe_min(matrix) < 0:
            raise ValueError("in tensor, entries should be positives")

        self.n, self.m = matrix.shape
        min_dim = min(self.n, self.m)
        if rank > min_dim:
            if verbose:
                print(f'NMF WARNING : Given rank {rank} is too large and will be truncated to {min_dim}')
            rank = min_dim

        if is_karmasparse(matrix):
            self.is_sparse = True
            self.matrix = matrix.copy()
            self.matrix_csr = matrix.tocsr()
            self.matrix_csc = matrix.tocsc()
        elif isinstance(matrix, np.ndarray):
            self.matrix = cast_2dim_float32_transpose(matrix).copy()
            self.is_sparse = False
        else:
            raise ValueError("Wrong matrix type : {}".format(type(matrix)))

        self.metric, self.rank, self.renorm = metric, rank, renorm

    def smart_init(self, init_iter=2, init_runs=15, svd_init=False):
        if svd_init:
            self.factorisation_initial(svd_init=True)
        else:
            self.factorisation_initial()
            self.iterate(init_iter)
            dist = self.dist()
            w, h = self.w.copy(), self.h.copy()
            for _ in range(init_runs):
                self.factorisation_initial()
                self.iterate(init_iter)
                dist_cand = self.dist()
                if dist > dist_cand:
                    dist, w, h = dist_cand, self.w.copy(), self.h.copy()
            self.factorisation_initial(w=w, h=h)

    def dist(self):
        if self.metric == 'KL':
            if self.is_sparse:
                return kl_div(self.matrix_csr, safe_dot(self.w, self.h, self.matrix_csr))
            else:
                return kl_div(self.matrix, safe_dot(self.w, self.h))
        elif self.metric == 'euclid':
            diff_matrix = (self.matrix - safe_dot(self.w, self.h))
            if is_karmasparse(diff_matrix):
                return diff_matrix.sum_power(2)
            else:
                return np.sum(diff_matrix.ravel() ** 2)
        else:
            raise NotImplementedError('Unknown metric')

    def factorisation_initial(self, w=None, h=None, svd_init=False):
        if svd_init:
            if self.is_sparse:
                self.w, self.h = nmf_svd_init(self.matrix_csr, self.rank, self.epsilon)
            else:
                self.w, self.h = nmf_svd_init(self.matrix, self.rank, self.epsilon)
        else:
            self.w = np.random.rand(self.n, self.rank) if w is None else w
            self.h = np.random.rand(self.rank, self.m) if h is None else h
            self.w = np.asarray(self.w.clip(self.epsilon), order='C')
            self.h = np.asarray(self.h.clip(self.epsilon), order='F')
        self.w, self.h = np.asarray(self.w, dtype=self.matrix.dtype), np.asarray(self.h, dtype=self.matrix.dtype)

    def iterate(self, maxiter):
        if self.metric == 'KL':
            for i in range(maxiter):
                self.brunet_update()
        elif self.metric == 'euclid':
            for i in range(maxiter):
                self.euclid_update()

    def brunet_update(self):
        """
        realise one step of update for kullback divergence
        """
        self.brunet_right()
        self.brunet_left()

    def product_wh(self, left):
        """
        computes m/w.dot(h) with a mask if self.issparse == True
        """
        if self.is_sparse:
            if left:
                return safe_dot(self.w, self.h, self.matrix_csr, mask_mode="divide")
            else:
                return safe_dot(self.w, self.h, self.matrix_csc, mask_mode="divide")
        else:
            return self.matrix / safe_dot(self.w, self.h)

    def brunet_right(self):
        self.h *= safe_dot(self.w.transpose(), self.product_wh(left=False))
        self.h /= self.w.sum(axis=0).reshape(self.rank, 1)
        self.h.clip(self.epsilon, out=self.h)

    def brunet_left(self):
        self.w *= safe_dot(self.product_wh(left=True), self.h.transpose())
        self.w /= self.h.sum(axis=1).transpose()
        self.w.clip(self.epsilon, out=self.w)
        if self.renorm:
            self.w = normalize(self.w, norm='l1', axis=0)

    def euclid_update(self):
        """
        realise one step of update for euclidean distance
        """
        self.euclid_right()
        self.euclid_left()

    def euclid_right(self):
        self.h *= safe_dot(self.w.transpose(), self.matrix) / \
                  safe_dot(safe_dot(self.w.transpose(), self.w), self.h)
        self.h.clip(self.epsilon, out=self.h)

    def euclid_left(self):
        self.w *= safe_dot(self.matrix, self.h.transpose()) / \
                  safe_dot(self.w, safe_dot(self.h, self.h.transpose()))
        self.w.clip(self.epsilon, out=self.w)


class GNMF(NMF):
    """
    From http://www.cad.zju.edu.cn/home/dengcai/Publication/Journal/TPAMI-GNMF.pdf
    In article's notation X = UV^T decomposition corresponds to M = X^T = VU^T = WH in our notation,
    so in each equation we replace V by W and U by H^T
    """

    def __init__(self, matrix, rank, adjacency, metric='KL', weight_type='adjacency'):
        super(GNMF, self).__init__(matrix, rank, metric)
        assert adjacency.shape == (self.n, self.n)
        if weight_type == 'adjacency':
            self.weights = adjacency
        elif weight_type == 'heat':
            raise NotImplementedError()
        elif weight_type == 'matrix':
            self.weights = safe_dot(self.matrix, self.matrix.transpose(), mat_mask=adjacency)
        else:
            raise NotImplementedError('Unknown weight_type: {}'.format(weight_type))
        self.degree_matrix = ks_diag(self.weights.sum(axis=1))
        self.maximal_degree = self.degree_matrix.max()
        self.graph_laplacian = self.degree_matrix - self.weights

    def brunet_left(self):
        arr = self.w * safe_dot(self.product_wh(left=True), self.h.transpose())

        for k, v in enumerate(self.h.sum(axis=1)):
            matrix = self.graph_laplacian + ks_diag(v * np.ones(self.n))

            # largest abs of eigenvalue of matrix is less equal to 2 * self.maximal_degree + v,
            # so we can't guarantee that it will be close enough to 0 to use decomposition into series to inverse
            # so solving linear system seems to be the fastest way here
            self.w[:, k] = cg(matrix.to_scipy_sparse(copy=False), arr[:, k], x0=self.w[:, k])[0]

        self.w.clip(self.epsilon, out=self.w)

    def euclid_left(self):
        self.w *= (safe_dot(self.matrix, self.h.transpose()) + safe_dot(self.weights, self.w)) / \
                  (safe_dot(self.w, safe_dot(self.h, self.h.transpose())) + safe_dot(self.degree_matrix, self.w))
        self.w.clip(self.epsilon, out=self.w)


class NMF_P(NMF):
    def _mean(self):
        return self.matrix_csr.mean() if self.is_sparse else self.matrix.mean()

    def precision_initial(self, a=5000., b=20., beta='random', strength=1., noise=None, one_side=True):
        if noise is not None:  # overwriting parameters principle parameter and using normalization
            a = noise
            b = 1.5 * (a - 1) * self._mean() / self.rank
        self.a, self.b, self.rank_list = a, b, []

        if one_side:
            self.cost = strength * (self.n + 2 * (self.a - 1.))
        else:
            self.cost = strength * (self.n + self.m + 2 * (self.a - 1.))

        if beta != 'random':
            self.beta = self.a / self.b * np.ones(self.rank)
        else:
            self.beta = np.random.gamma(self.a, 1. / self.b, size=(self.rank,))
        self.threshold = self.cost / 2. / self.b

    def dist(self):
        if self.is_sparse:
            kl = kl_div(self.matrix_csr, safe_dot(self.w, self.h, self.matrix_csr))
        else:
            kl = kl_div(self.matrix, safe_dot(self.w, self.h))
        return kl + 0.5 * self._penalty().dot(self.beta) - self.cost * np.log(self.beta.sum())

    def brunet_right(self):
        self.h *= (safe_dot(self.w.transpose(), self.product_wh(left=False)) /
                   (self.w.sum(axis=0).reshape(self.rank, 1) + safe_dot(np.diag(self.beta), self.h)))
        self.h.clip(self.epsilon, out=self.h)

    def brunet_left(self):
        self.w *= (safe_dot(self.product_wh(left=True), self.h.transpose()) /
                   (self.h.sum(axis=1).transpose() + safe_dot(self.w, np.diag(self.beta))))
        self.w.clip(self.epsilon, out=self.w)

    def _penalty(self):
        res = (self.w * self.w).sum(axis=0)
        res += (self.h * self.h).sum(axis=1)
        res += 2. * self.b
        return res

    def brunet_beta(self):
        self.beta = self.cost / self._penalty()

    def brunet_update(self):
        self.brunet_left()
        self.brunet_right()
        self.brunet_beta()

    def iterate(self, maxiter=150):
        MAX_ITER = 150
        if maxiter is None:
            t1 = time.time()
            self.brunet_update()
            maxiter = MAX_ITER + int(ADD_TIME / (max(time.time() - t1, 1)))
        for i in range(maxiter):
            self.brunet_update()
            self.add_rank_estimate()
            if i > MAX_ITER // 2 and self.check_convergence():
                break

    def get_estimate(self):
        return min(max(self.rank_list[-1], 2), self.rank)

    def check_convergence(self, nb_repeat=60):
        if len(self.rank_list) < nb_repeat:
            return False
        else:
            return all([x == self.rank_list[-1] < self.rank
                        for x in self.rank_list[-nb_repeat:]])

    def add_rank_estimate(self, frac=0.1):
        guess_rank = np.sum(self.beta < frac * self.threshold)
        self.rank_list.append(guess_rank)


def nmf_fold(matrix, right_factor, max_iter=30):
    """
    Let M ~ L.dot(R) be a NMF approximation.
    `nmf_fold` computes L (left_factor) given M (matrix) and right_factor (R).
    Result is unique due to the convexity of the problem
    and can be obtain through a simple iteration procedure.
    Returned left_factor is always dense even if right_factor is sparse.
    """
    assert matrix.shape[1] == right_factor.shape[1]
    if safe_min(right_factor) < 0:
        raise ValueError("All elements of the 'Right factor' must be positive")
    if safe_min(matrix) < 0:
        raise ValueError("All elements of the 'Matrix' must be positive")

    right_margin = right_factor.sum(axis=1).clip(EPSILON)
    rank = right_factor.shape[0]
    left_factor = np.full((matrix.shape[0], rank), 1.0 / rank, dtype=np.float)

    if is_karmasparse(matrix) and matrix.format != "csr":
        matrix = matrix.tocsr()  # taking a good alignment
    elif not is_karmasparse(matrix):
        matrix = KarmaSparse(matrix)

    for _ in range(max_iter):
        error = safe_dot(left_factor, right_factor, matrix, mask_mode="divide")
        left_factor *= safe_dot(error, right_factor.transpose(), dense_output=True)
        left_factor /= right_margin
        left_factor.clip(EPSILON, out=left_factor)
    return left_factor

#
# Copyright tinyclues, All rights reserved
#

import numpy as np
from karma.learning.matrix_utils import safe_dot
from sklearn.utils.extmath import norm
from six.moves import range


def randomized_range_finder(A, size, n_iter):
    """
    Computes an orthonormal matrix whose range approximates the range of A.
    """
    R = np.random.normal(size=(A.shape[1], size))
    Y = safe_dot(A, R)
    del R
    for i in range(n_iter):
        Y = safe_dot(A, safe_dot(A.transpose(), Y))
    Q, _ = np.linalg.qr(Y, mode='reduced')
    return Q


def randomized_svd(M, n_components, n_iter=2):
    """Computes a truncated randomized SVD
    Parameters
    ----------
    M: ndarray or sparse matrix
        Matrix to decompose
    n_components: int
        Number of singular values and vectors to extract.
    n_iter: int (default is 2)
        Number of power iterations (can be used to deal with very noisy
        problems).
    """
    n_oversamples = 10

    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_samples > n_features:
        transpose = True
        M = M.transpose()
    else:
        transpose = False

    Q = randomized_range_finder(M, n_random, n_iter)
    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_dot(Q.transpose(), M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, S, V = np.linalg.svd(B, full_matrices=False)
    del B
    U = safe_dot(Q, Uhat)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, S[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], S[:n_components], V[:n_components, :]


def nmf_svd_init(matrix, rank, eps):
    """
    """
    U, S, V = randomized_svd(matrix, rank, n_iter=1)
    W, H = np.zeros(U.shape), np.zeros(V.shape)
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
    for j in range(1, rank):
        x, y = U[:, j], V[j, :]
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n
        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v
    W[W < eps] = 0
    H[H < eps] = 0
    avg = matrix.mean()
    W[W == 0] = np.abs(avg * np.random.randn(len(W[W == 0])) / 100)
    H[H == 0] = np.abs(avg * np.random.randn(len(H[H == 0])) / 100)
    return W, H


def rank_from_variance_ratio(matrix, ratio, max_dim=1000):
    """
    Uses SVD decomposition to determine "rank" value that allows to explain "ratio"
    fraction of total variance.
    Uses "randomized_svd" approximation to reduce the runtime.
    Code is inspired by "sklearn.decomposition.pca"

    :Example:
        >>> np.random.seed(50)
        >>> r = np.random.randint(5, 30)
        >>> a = np.random.randn(100, r).dot(np.random.randn(r, 100))
        >>> rank_from_variance_ratio(a, 0.999) == r
        True
        >>> a = np.random.randn(100, 20).dot(np.random.randn(20, 100)) + \
                np.random.randn(100, 10).dot(np.random.randn(10, 100))
        >>> rank_from_variance_ratio(a, 0.9)
        20
        >>> rank_from_variance_ratio(a, 0.8)
        15
        >>> rank_from_variance_ratio(a, 0.6)
        10
    """
    spectrum = randomized_svd(matrix, max_dim)[1]
    explained_variance = spectrum ** 2
    explained_variance_ratio = explained_variance / explained_variance.sum()
    ratio_cumsum = explained_variance_ratio.cumsum()
    n_components = np.sum(ratio_cumsum <= ratio) + 1
    return n_components

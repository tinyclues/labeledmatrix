#
# Copyright tinyclues, All rights reserved
#

import numpy as np
from cyperf.matrix.karma_sparse import KarmaSparse, is_karmasparse, ks_kron

from karma.learning.matrix_utils import pairwise_buddy
from six.moves import range

__all__ = ['sparse_tail_clustering']


def sparse_tail_clustering(matrix, mults, k, min_density=0.):
    """
    >>> from karma.learning.matrix_utils import normalize
    >>> import scipy.sparse as sp
    >>> mat = normalize(KarmaSparse(sp.rand(20, 10**3, 0.5)), norm="l1", axis=1)
    >>> mults = np.sort(np.random.rand(20))
    >>> len(sparse_tail_clustering(mat, mults, 1, 0.4))
    20
    >>> v = np.array([[7, 0, 3],
    ...               [5, 0, 5],
    ...               [3, 3, 4],
    ...               [0, 10, 0],
    ...               [0, 9, 1]])
    >>> sparse_tail_clustering(v, [1, 2, 4, 4, 1], 2, 0.)
    [2, 2, 2, 3, 3]
    """
    mults = np.array(mults)
    n = len(mults)
    indices = np.arange(n)
    if not is_karmasparse(matrix):
        sp_matrix = KarmaSparse(matrix, format="csc")
    else:
        sp_matrix = matrix.tocsc()

    clusters = {}
    for _ in range(n - k):
        absolute_j = indices.compress(mults)[mults.compress(mults).argmin()]
        m = mults[absolute_j]
        v = sp_matrix[absolute_j].tocsr()
        # remove "v"
        mults[absolute_j] = 0
        pos = np.zeros((n, 1))
        pos[absolute_j] = 1
        sp_matrix = sp_matrix - ks_kron(v, pos, format="csc")
        if v.nnz:
            density = pairwise_buddy(v, sp_matrix, cutoff=0, nb_keep=1)
            if len(density.indices):
                absolute_jj = density.indices[0]
                my_density = density.data[0]
            else:
                break
            if my_density >= min_density:
                vv = sp_matrix[absolute_jj].tocsr()
                mm = mults[absolute_jj]
                clusters[absolute_j] = absolute_jj
                mults[absolute_jj] = m + mm
                update = (v * m + vv * mm) / (m + mm) - vv
                pos = np.zeros((n, 1))
                pos[absolute_jj] = 1
                sp_matrix = sp_matrix + ks_kron(update, pos, format="csc")
    res = []
    for each in range(n):
        a = each
        while a in clusters:
            a = clusters[a]
        res.append(a)
    return res

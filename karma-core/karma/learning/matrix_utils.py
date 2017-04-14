#
# Copyright tinyclues, All rights reserved
#

import numpy as np

from scipy.stats.mstats import mquantiles
from scipy.stats import threshold
from scipy.linalg import block_diag
from scipy.sparse import isspmatrix as is_scipysparse
from cyperf.tools import logit
from cyperf.matrix.rank_dispatch import matrix_rank_dispatch
from cyperf.matrix.argmax_dispatch import sparse_argmax_dispatch
from cyperf.matrix.karma_sparse import (KarmaSparse, is_karmasparse,
                                        truncate_by_count_axis1_dense,
                                        truncate_by_count_axis1_sparse, ks_hstack, ks_vstack)
from cyperf.matrix.routine import idiv_2d, idiv_flat


class SparseUtilsException(Exception):
    pass


def truncate_by_budget(matrix, density, volume, axis=1):
    """
    Return a sparse matrices for wich sum, by row, of non-zero
    elements with respect to density parameter is bigger or equal to volume.

    :param matrix: square similarity matrix that is used to define neighborhoods
                   of a given point
    :param density: diagonal matrix or flat numpy array
    :param volume:
    :param axis:
    :return: a scipy.sparse.csr matrix for wich sum, by row, of non-zero
             elements with respect to density parameter is bigger or equal to
             volume

    Example: ::
        >>> similarity = np.array([[1., 0.4, 0.2, 0.1],
        ...                        [0.3, 1., 0.5, 0],
        ...                        [0.4, 0.5, 1., 0.6],
        ...                        [0.3, 0.8, 0.6, 1.]])
        >>> density = np.array([1.03, 1.02, 1.01, 1.])
        >>> truncate_by_budget(similarity, density, 2).toarray()  # top(nb_h = 2)
        array([[ 1. ,  0.4,  0. ,  0. ],
               [ 0. ,  1. ,  0.5,  0. ],
               [ 0. ,  0. ,  1. ,  0.6],
               [ 0. ,  0.8,  0. ,  1. ]])
        >>> truncate_by_budget(similarity, density, 3).toarray()  # top(nb_h = 3)
        array([[ 1. ,  0.4,  0.2,  0. ],
               [ 0.3,  1. ,  0.5,  0. ],
               [ 0. ,  0.5,  1. ,  0.6],
               [ 0. ,  0.8,  0.6,  1. ]])
        >>> density = np.array([2., 1.01, 0.2, 1.0])
            >>> truncate_by_budget(similarity, density, 2.).toarray()
            array([[ 1. ,  0. ,  0. ,  0. ],
                   [ 0.3,  1. ,  0.5,  0. ],
                   [ 0. ,  0.5,  1. ,  0.6],
                   [ 0. ,  0.8,  0. ,  1. ]])
        >>> similarity = np.array([[0., 0., 0., 0.],
        ...                        [0., 1., 0., 0],
        ...                        [0., 2., 0., 0.],
        ...                        [0., 3., 0., 1.]])
        >>> density = np.array([1.03, 1.02, 1.01, 1.])
        >>> truncate_by_budget(similarity, density, 2).toarray()  # top(nb_h = 2)
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  2.,  0.,  0.],
               [ 0.,  3.,  0.,  1.]])
    """
    density = density if len(density.shape) == 1 else density.diagonal()
    if not is_karmasparse(matrix):
        matrix = KarmaSparse(matrix)
    return matrix.truncate_by_budget(density, volume, axis)


def truncate_by_count(matrix, max_rank, axis):
    """
    Returns for the given `axis` the `max_rank` greater elements
    Args:
        matrix: sparse matrix or numpy array
        max_rank: number or array of size corresponding to axis
        axis: 0, 1 or None

    Exemple: ::
        >>> mat = np.array([[0.4, 0.51, 0, 0.1, 0],
        ...                 [0, 0, 0, 0, 0],
        ...                 [0, 0.1, 0.4, 0.52, 0],
        ...                 [0.1, 0, 0.5, 0.4, 0]])
        >>> res = truncate_by_count(KarmaSparse(mat), max_rank=1, axis=1).toarray()
        >>> res
        array([[ 0.  ,  0.51,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.52,  0.  ],
               [ 0.  ,  0.  ,  0.5 ,  0.  ,  0.  ]])
        >>> np.allclose(res, truncate_by_count(mat, max_rank=1, axis=1))
        True
        >>> np.allclose(truncate_by_count(KarmaSparse(mat), max_rank=2, axis=None).toarray(),
        ...             truncate_by_count(mat, max_rank=2, axis=None))
        True
        >>> truncate_by_count(mat, max_rank=2, axis=None)
        array([[ 0.  ,  0.51,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.52,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])
    """
    if axis not in [0, 1, None]:
        raise ValueError('axis must be 0, 1 or None')
    if not np.all(max_rank >= 0):
        raise ValueError('max_rank must be non-negative number or array')
    if len(matrix.shape) != 2:
        raise ValueError('truncate_by_count works only for 2-dimensional matrices')

    def truncate_by_count_dense(matrix, ranks, axis):
        if matrix.shape[1 - axis] != ranks.shape[0]:
            raise ValueError('matrix and max_rank dimensions aren\'t equal: {} != {}'.format(matrix.shape[1 - axis],
                                                                                             ranks.shape[0]))
        if axis == 1:
            density = 1. * np.mean(ranks) / matrix.shape[1]
            if density > 0.25:
                return truncate_by_count_axis1_dense(np.ascontiguousarray(matrix), ranks)
            else:
                return truncate_by_count_axis1_sparse(np.ascontiguousarray(matrix), ranks)
        elif axis == 0:
            return truncate_by_count_dense(matrix.transpose(), ranks, axis=1).transpose()

    if not np.isscalar(max_rank):
        return truncate_by_count_dense(np.asarray(matrix), np.asarray(max_rank, dtype=np.int), axis)
    elif is_karmasparse(matrix):
        return matrix.truncate_by_count(max_rank, axis)
    elif is_scipysparse(matrix):
        return KarmaSparse(matrix).truncate_by_count(max_rank, axis)
    elif axis is None:
        shape = matrix.shape
        matrix = matrix.copy().flatten()
        matrix[np.argpartition(-matrix, max_rank)[max_rank:]] = 0
        matrix = matrix.reshape(shape)
        return matrix
    else:
        max_rank = np.full(matrix.shape[1 - axis], max_rank, dtype=np.int32)
        return truncate_by_count_dense(matrix, max_rank, axis)


def nonzero_mask(matrix):
    """
    For a given matrix (sparse or dense), returns a matrix of values:
        - 0: no data
        - 1: a data is defined

    :param matrix: a scipy.sparse or np.array
    :return: a scipy.sparse or np.array
    """
    if is_scipysparse(matrix):
        matrix = matrix.copy()
        matrix.eliminate_zeros()
        matrix.data[:] = 1
    elif is_karmasparse(matrix):
        matrix = matrix.nonzero_mask()
    else:
        matrix = matrix.copy()
        matrix[matrix != 0] = 1
    return matrix


def complement(matrix, other):
    """
    For a given `matrix` (sparse or dense) set all elements that are non-zero in `other`
    to zeros.

    :param matrix: a scipy.sparse or np.array
    :return: a KarmaSparse or np.array
    >>> import scipy.sparse as sp
    >>> mat = np.array([[4, 0.5, 0],
    ...                 [0, 12, 0],
    ...                 [0, 1, 6],
    ...                 [1, 0, 5]])
    >>> ks = KarmaSparse(mat)
    >>> spmat = sp.csr_matrix(mat)
    >>> diag_idx = range(min(mat.shape))
    >>> mask = np.zeros(mat.shape)
    >>> mask[diag_idx, diag_idx] = 1
    >>> complement(mat, (diag_idx, diag_idx))
    array([[ 0. ,  0.5,  0. ],
           [ 0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ],
           [ 1. ,  0. ,  5. ]])
    >>> np.all(complement(mat, (diag_idx, diag_idx))
    ...    == complement(ks, (diag_idx, diag_idx)).toarray())
    ...    == complement(spmat, (diag_idx, diag_idx)).toarray()
    ...    == complement(ks, mask).toarray()
    ...    == complement(ks, KarmaSparse(mask)).toarray()
    ...    == complement(mat, mask))
    True
    """
    if is_karmasparse(matrix):
        return matrix.complement(other)
    elif is_scipysparse(matrix):
        return KarmaSparse(matrix).complement(other)
    else:
        matrix = matrix.copy()
        if isinstance(other, tuple):
            rows_indices, cols_indices = other  # already indices
        else:
            rows_indices, cols_indices = other.nonzero()  # matrix
        matrix[rows_indices, cols_indices] = 0
    return matrix


def truncate_by_cumulative(matrix, per, axis):
    """
    >>> mat = np.array([[0.4, 0.5, 0, 0.1, 0],
    ...                 [0, 0, 0, 0, 0],
    ...                 [0, 0.1, 0.4, 0.5, 0],
    ...                 [0.1, 0, 0.5, 0.4, 0]])
    >>> truncate_by_cumulative(KarmaSparse(mat), per=0.1, axis=1).toarray()
    array([[ 0.4,  0.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0.4,  0.5,  0. ],
           [ 0. ,  0. ,  0.5,  0.4,  0. ]])
    >>> truncate_by_cumulative(KarmaSparse(mat), per=0.5, axis=1).toarray()
    array([[ 0. ,  0.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0.5,  0. ],
           [ 0. ,  0. ,  0.5,  0. ,  0. ]])
    """
    per = max(min(1, per), 0)
    if is_scipysparse(matrix):
        return KarmaSparse(matrix).truncate_by_cumulative(per, axis)
    elif is_karmasparse(matrix):
        return matrix.truncate_by_cumulative(per, axis)
    else:
        raise NotImplementedError()


def buddies_matrix(matrix, cutoff=0.001, nb_keep=200, top=None, cumtop=None):
    """
    Sparsify the matrix lines: remove cumtop of the total weigth, then keeps only top elts
    Then computes pairwise_minsum with params nb_keep and cutoff
        (those params sparsifies the result: keep at most nb_keep elt per line,
         remove elts <= cutoff).
    >>> mat = np.array([[0.4, 0.5, 0, 0.1, 0],
    ...                 [0, 0.1, 0.4, 0.5, 0],
    ...                 [0, 0, 0, 0, 0],
    ...                 [0.1, 0, 0.5, 0.4, 0]])
    >>> buddies_matrix(KarmaSparse(mat)).toarray()
    array([[ 1. ,  0.2,  0. ,  0.2],
           [ 0.2,  1. ,  0. ,  0.8],
           [ 0. ,  0. ,  0. ,  0. ],
           [ 0.2,  0.8,  0. ,  1. ]])
    >>> buddies_matrix(KarmaSparse(mat, format="csc"), nb_keep=2).toarray()
    array([[ 1. ,  0.2,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0.8],
           [ 0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0.8,  0. ,  1. ]])
    """
    # truncate for RAM security reasons
    if top is not None and cumtop is not None:
        matrix = truncate_by_count(truncate_by_cumulative(matrix, per=cumtop, axis=1),
                                   max_rank=top, axis=1)
        # TODO: we should keep the same argument names: per/cumtop nb/top
    elif top is not None:
        matrix = truncate_by_count(matrix, max_rank=top, axis=1)
    elif cumtop is not None:
        matrix = truncate_by_cumulative(matrix, per=cumtop, axis=1)

    if not is_karmasparse(matrix):
        matrix = KarmaSparse(matrix)
    return matrix.pairwise_min_top(matrix.transpose(), nb_keep=nb_keep, cutoff=cutoff)


def pairwise_buddy(matrix, other, cutoff=0.001, nb_keep=200):
    """
    >>> mat = np.array([[0.4, 0.5, 0, 0.1, 0],
    ...                 [0, 0.1, 0.4, 0.5, 0],
    ...                 [0, 0, 0, 0, 0],
    ...                 [0.1, 0, 0.5, 0.4, 0]])
    >>> pairwise_buddy(mat, mat[:2]).toarray()
    array([[ 1. ,  0.2],
           [ 0.2,  1. ],
           [ 0. ,  0. ],
           [ 0.2,  0.8]])
    >>> pairwise_buddy(KarmaSparse(mat), mat[:2]).toarray()
    array([[ 1. ,  0.2],
           [ 0.2,  1. ],
           [ 0. ,  0. ],
           [ 0.2,  0.8]])
    >>> pairwise_buddy(KarmaSparse(mat, format="csc"), KarmaSparse(mat[1:]), nb_keep=2).toarray()
    array([[ 0.2,  0. ,  0.2],
           [ 1. ,  0. ,  0.8],
           [ 0. ,  0. ,  0. ],
           [ 0.8,  0. ,  1. ]])
    """
    if not is_karmasparse(matrix):
        matrix = KarmaSparse(matrix, format="csr")
    if not is_karmasparse(other):
        other = KarmaSparse(other, format="csr")
    return matrix.pairwise_min_top(other.transpose(), nb_keep=nb_keep, cutoff=cutoff)


def pseudo_element_inverse(matrix, scalar=1.):
    """
    >>> import scipy.sparse as sp
    >>> a = np.array([[1.,2.], [5., 0.]])
    >>> pseudo_element_inverse(a)
    array([[ 1. ,  0.5],
           [ 0.2,  0. ]])
    >>> a = np.random.binomial(3, 0.4, (13, 10))
    >>> np.all(pseudo_element_inverse(sp.csr_matrix(a)).toarray() ==
    ...        pseudo_element_inverse(a))
    True
    >>> np.all(pseudo_element_inverse(KarmaSparse(a)).toarray() ==
    ...        pseudo_element_inverse(a))
    True
    """
    if is_scipysparse(matrix):
        mat = matrix.copy()
        matrix.eliminate_zeros()
        mat.data = scalar / matrix.data
    elif is_karmasparse(matrix):
        return scalar / matrix
    elif np.isscalar(matrix):
        return scalar / matrix
    else:
        mat = np.zeros(matrix.shape)
        mat[matrix != 0] = scalar / matrix[matrix != 0]
    return mat


def safe_compatibility_renormalization(matrix, cat1, cat2, same_factor, not_same_factor):
    arg1 = safe_argmax(cat1, axis=1)
    arg2 = safe_argmax(cat2, axis=1)
    if not is_karmasparse(matrix):
        matrix = KarmaSparse(matrix)
    return matrix.compatibility_renormalization(arg1, arg2, same_factor, not_same_factor)


def safe_mean(matrix, axis=None):
    """
    >>> import scipy.sparse as sp
    >>> a = sp.rand(10, 5, 0.96)
    >>> np.allclose(safe_mean(a.toarray()), safe_mean(a.tocsc()))
    True
    >>> np.allclose(safe_mean(a.toarray()), safe_mean(a.tocsr()))
    True
    >>> np.allclose(a.toarray().mean(), safe_mean(a.tocsr()))
    True
    >>> np.allclose(a.toarray().mean(axis=1), safe_mean(a.tocsr(), axis=1))
    True
    >>> np.allclose(a.toarray().mean(axis=1), safe_mean(a.tocsc(), axis=1))
    True
    >>> np.allclose(a.toarray().mean(axis=0), safe_mean(a.tocsr(), axis=0))
    True
    >>> np.allclose(a.toarray().mean(axis=0), safe_mean(a.tocsc(), axis=0))
    True
    >>> np.allclose(a.toarray().mean(axis=1), safe_mean(KarmaSparse(a), axis=1))
    True
    >>> np.allclose(a.toarray().mean(axis=0), safe_mean(KarmaSparse(a), axis=0))
    True
    >>> np.allclose(a.toarray().mean(), safe_mean(KarmaSparse(a)))
    True
    """
    if is_scipysparse(matrix):
        m = np.array(matrix.mean(axis)).flatten()
        return m[0] if axis is None else m
    else:
        return matrix.mean(axis)


def rank_matrix(matrix, axis, reverse=False):
    """
    Warning : different for sparse and dense matrix
    >>> mat = np.array([[4, 5, 0, -1, 0],
    ...                 [0, 0, 0, 0, 0],
    ...                 [0, 1, 4, 5, 0],
    ...                 [1, 0, 5, 4, 0],
    ...                 [5, 1, 9, 4, 2]])
    >>> ks = KarmaSparse(mat)
    >>> rank_matrix(mat, axis=1)
    array([[3, 4, 1, 0, 2],
           [0, 1, 2, 3, 4],
           [0, 2, 3, 4, 1],
           [2, 0, 4, 3, 1],
           [3, 0, 4, 2, 1]])
    >>> rank_matrix(mat, axis=0)
    array([[3, 4, 0, 0, 0],
           [0, 0, 1, 1, 1],
           [1, 2, 2, 4, 2],
           [2, 1, 3, 2, 3],
           [4, 3, 4, 3, 4]])
    >>> rank_matrix(ks, axis=1).toarray().astype(np.int)
    array([[2, 3, 0, 1, 0],
           [0, 0, 0, 0, 0],
           [0, 1, 2, 3, 0],
           [1, 0, 3, 2, 0],
           [4, 1, 5, 3, 2]])
    >>> rank_matrix(ks, axis=0).toarray().astype(np.int)
    array([[2, 3, 0, 1, 0],
           [0, 0, 0, 0, 0],
           [0, 1, 1, 4, 0],
           [1, 0, 2, 2, 0],
           [3, 2, 3, 3, 1]])
    """
    if is_scipysparse(matrix):
        matrix = KarmaSparse(matrix)
    if is_karmasparse(matrix):
        result = matrix.rank(axis, reverse)
    else:
        if reverse:
            result = matrix.argsort(axis=axis)
            if axis == 0:
                result = result[::-1, :]
            else:
                result = result[:, ::-1]
            result = result.argsort(axis=axis)
        else:
            result = matrix.argsort(axis=axis).argsort(axis=axis)
    return result


def argsort_vector(vec, reverse=False):
    """
    >>> import scipy.sparse as sp
    >>> vec = sp.csc_matrix(np.array([0, 3, 0, 6, -3]))
    >>> argsort_vector(vec)
    (array([4, 1, 3], dtype=int32), array([-3.,  3.,  6.]))
    >>> sp_vec = sp.csc_matrix(np.array([0, 3, 0, 6, -3]))
    >>> argsort_vector(sp_vec)
    (array([4, 1, 3], dtype=int32), array([-3.,  3.,  6.]))
    """
    if vec.shape[0] != 1 and len(vec.shape) != 1:
        raise SparseUtilsException("vec should be of shape (1,n)")
    if is_scipysparse(vec) or is_karmasparse(vec):
        vec = KarmaSparse(vec, format="csr")
        order = np.argsort(vec.data)
        if reverse:
            order = order[::-1]
        return (vec.indices[order], vec.data[order])
    else:
        order = np.argsort(vec)
        if reverse:
            order = order[::-1]
        return (order, vec.flatten()[order])


def safe_max(matrix, axis=None):
    """
    >>> import scipy.sparse as sp
    >>> a = sp.rand(10, 5, 0.96)
    >>> np.allclose(safe_max(a.toarray()), safe_max(a.tocsc()))
    True
    >>> np.allclose(safe_max(a.toarray()), safe_max(a.tocsr()))
    True
    >>> np.allclose(safe_max(a.toarray(), axis=1), safe_max(KarmaSparse(a), axis=1))
    True
    >>> np.allclose(safe_max(a.toarray(), axis=1), safe_max(KarmaSparse(a), axis=1))
    True
    >>> np.allclose(safe_max(a.toarray(), axis=0), safe_max(KarmaSparse(a), axis=0))
    True
    >>> np.allclose(safe_max(a.toarray(), axis=0), safe_max(KarmaSparse(a), axis=0))
    True
    """
    if is_scipysparse(matrix):
        m = matrix.max(axis)
        return m if axis is None else m.toarray().flatten()
    else:
        try:
            return matrix.max(axis)
        except TypeError:
            return max(matrix)


def safe_min(matrix, axis=None):
    """
    >>> import scipy.sparse as sp
    >>> a = np.array([[ 0.284,  0.41 ,  0.11 ,  0.54 ,  0.99 ],
    ...               [ 0.465,  0.265,  0.444,  0.714,  0.383],
    ...               [ 0.   ,  0.625,  0.439,  0.415,  0.06 ],
    ...               [ 0.709,  0.573,  0.743,  0.,  0.5  ],
    ...               [ 0.809,  0.982,  0.779,  0.99 ,  0.995]])
    >>> a = sp.csc_matrix(a) if np.random.rand() > 0.5 else sp.rand(10, 5, 0.9)
    >>> np.allclose(safe_min(a.toarray()), safe_min(a.tocsc()))
    True
    >>> np.allclose(safe_min(a.toarray()), safe_min(a.tocsr()))
    True
    >>> np.allclose(safe_min(a.toarray(), axis=1), safe_min(KarmaSparse(a), axis=1))
    True
    >>> np.allclose(safe_min(a.toarray(), axis=1), safe_min(KarmaSparse(a), axis=1))
    True
    >>> np.allclose(safe_min(a.toarray(), axis=0), safe_min(KarmaSparse(a), axis=0))
    True
    >>> np.allclose(safe_min(a.toarray(), axis=0), safe_min(KarmaSparse(a), axis=0))
    True
    """
    if is_scipysparse(matrix):
        m = matrix.min(axis)
        return m if axis is None else m.toarray().flatten()
    else:
        try:
            return matrix.min(axis)
        except TypeError:
            return min(matrix)

def safe_sum(matrix, axis=None):
    """
    >>> import scipy.sparse as sp
    >>> a = np.array([[ 0.284,  0.41 ,  0.11 ,  0.54 ,  0.99 ],
    ...               [ 0.465,  0.265,  0.444,  0.714,  0.383],
    ...               [ 0.   ,  0.625,  0.439,  0.415,  0.06 ],
    ...               [ 0.709,  0.573,  0.743,  0.,  0.5  ],
    ...               [ 0.809,  0.982,  0.779,  0.99 ,  0.995]])
    >>> a = KarmaSparse(a) if np.random.rand() > 0.5 else sp.rand(10, 5, 0.9)
    >>> np.allclose(safe_sum(a.toarray()), safe_sum(a.tocsc()))
    True
    >>> np.allclose(safe_sum(a.toarray()), safe_sum(a.tocsr()))
    True
    >>> np.allclose(safe_sum(a.toarray(), axis=1), safe_sum(KarmaSparse(a), axis=1))
    True
    >>> np.allclose(safe_sum(a.toarray(), axis=1), safe_sum(KarmaSparse(a), axis=1))
    True
    >>> np.allclose(safe_sum(a.toarray(), axis=0), safe_sum(KarmaSparse(a), axis=0))
    True
    >>> np.allclose(safe_sum(a.toarray(), axis=0), safe_sum(KarmaSparse(a), axis=0))
    True
    """
    if is_scipysparse(matrix):
        m = np.array(matrix.sum(axis)).flatten()
        return m[0] if axis is None else m
    else:
        try:
            return matrix.sum(axis)
        except TypeError:
            return sum(matrix)

def safe_argmax(matrix, axis=None):
    """
    >>> import scipy.sparse as sp
    >>> n = 5
    >>> a = KarmaSparse((sp.rand(n, 2*n, 0.2).toarray()), format="csc")
    >>> b = KarmaSparse((sp.rand(n, 2*n, 0.1).toarray()), format="csc")
    >>> a = a - b
    >>> ac = np.allclose
    >>> a[safe_argmax(a.toarray())] == a[safe_argmax(a.tocsr())]
    True
    >>> a[safe_argmax(a.toarray())] == a[safe_argmax(a.tocsc())]
    True
    >>> ac(safe_max(a.toarray(), axis=1),
    ...    a.toarray()[np.array(range(n)),safe_argmax(a.tocsr(), axis=1)])
    True
    >>> ac(safe_max(a.toarray(), axis=0),
    ...     a.toarray()[safe_argmax(a.tocsr(), axis=0), np.array(range(2*n))])
    True
    >>> ac(safe_max(a.toarray(), axis=0),
    ...     a.toarray()[safe_argmax(a.tocsc(), axis=0), np.array(range(2*n))])
    True
    >>> ac(safe_max(a.toarray(), axis=1),
    ...    a.toarray()[np.array(range(n)), safe_argmax(a.tocsc(), axis=1)])
    True
    >>> m = np.array([[-1., 0, -1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmax(m, axis=0), np.array([1, 1, 1]))
    True
    >>> safe_argmax(m)
    (1, 1)
    >>> m = np.array([[-2, 0, -1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csc")
    >>> ac(safe_argmax(m, axis=0), np.array([1, 1, 1]))
    True
    >>> ac(safe_argmax(m, axis=1), np.array([1, 1]))
    True
    >>> m = np.array([[-2, 0, -1], [0, 2., 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmax(m, axis=0), np.array([1, 1, 1]))
    True
    >>> ac(safe_argmax(m, axis=1), np.array([1, 1, 1]))
    True
    >>> m = np.array([[0, 0, -1], [0, 2, 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmax(m, axis=0), np.array([0, 1, 1]))
    True
    >>> ac(safe_argmax(m, axis=1), np.array([0, 1, 1]))
    True
    >>> m = np.array([[-1, 1, -1], [-1, 2, 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmax(m, axis=0), np.array([2, 1, 1]))
    True
    >>> ac(safe_argmax(m, axis=1), np.array([1, 1, 1]))
    True
    >>> m = np.array([[1, 1, -1], [1, 0., 1], [0, 0, -1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmax(m, axis=0), np.array([0, 0, 1]))
    True
    >>> ac(safe_argmax(m, axis=1), np.array([0, 0, 0]))
    True
    >>> safe_argmax(m)
    (0, 0)
    >>> m = np.array([[1, 1, -1], [1, 0, 1], [0, 0, -1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmax(m, axis=0), np.array([0, 0, 1]))
    True
    >>> ac(safe_argmax(m.tocsc(), axis=0), np.array([0, 0, 1]))
    True
    >>> safe_argmax(m)
    (0, 0)
    >>> m = np.array([[1, 1, 0, 0], [1, 0, 1, 1], [-1, 0, -1, 0]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmax(m, axis=1), np.array([0, 0, 1]))
    True
    >>> m = np.array([[-2, 0, -1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csc")
    >>> ac(safe_argmax(m, axis=0), np.array([1, 1, 1]))
    True
    """
    if is_scipysparse(matrix):
        matrix = KarmaSparse(matrix)
    if is_karmasparse(matrix):
        return matrix.argmax(axis=axis)
    else:
        if axis is None:
            return np.unravel_index(matrix.argmax(), matrix.shape)
        else:
            return matrix.argmax(axis=axis)


def safe_argmin(matrix, axis=None):
    """
    >>> import scipy.sparse as sp
    >>> n = 5
    >>> a = KarmaSparse((sp.rand(n, 2*n, 0.2).toarray()), format="csc")
    >>> b = KarmaSparse((sp.rand(n, 2*n, 0.1).toarray()), format="csc")
    >>> a = a - b
    >>> ac = np.allclose
    >>> a[safe_argmin(a.toarray())] == a[safe_argmin(a.tocsr())]
    True
    >>> a[safe_argmin(a.toarray())] == a[safe_argmin(a.tocsc())]
    True
    >>> ac(safe_min(a.toarray(), axis=1),
    ...    a.toarray()[np.array(range(n)), safe_argmin(a.tocsr(), axis=1)])
    True
    >>> ac(safe_min(a.toarray(), axis=1),
    ...    a.toarray()[np.array(range(n)), safe_argmin(a.tocsc(), axis=1)])
    True
    >>> ac(safe_min(a.toarray(), axis=0),
    ...    a.toarray()[safe_argmin(a.tocsr(), axis=0), np.array(range(2*n))])
    True
    >>> ac(safe_min(a.toarray(), axis=0),
    ...    a.toarray()[safe_argmin(a.tocsc(), axis=0), np.array(range(2*n))])
    True
    >>> m = np.array([[-2, 0., -1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmin(m, axis=0), np.array([0, 0, 0]))
    True
    >>> ac(safe_argmin(m, axis=1), np.array([0, 0]))
    True
    >>> safe_argmin(m)
    (0, 0)
    >>> m = np.array([[-2, 0, -1], [0, 2., 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmin(m, axis=0), np.array([0, 0, 0]))
    True
    >>> ac(safe_argmin(m, axis=1), np.array([0, 0, 0]))
    True
    >>> safe_argmin(m)
    (0, 0)
    >>> m = np.array([[0, 0, -1], [0, 2, 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmin(m, axis=0), np.array([0, 0, 0]))
    True
    >>> ac(safe_argmin(m, axis=1), np.array([2, 0, 0]))
    True
    >>> m = np.array([[-1, 1, -1], [-1, 2, 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmin(m, axis=0), np.array([0, 0, 0]))
    True
    >>> ac(safe_argmin(m, axis=1), np.array([0, 0, 0]))
    True
    >>> m = np.array([[1, 1, -1], [1, 0, 1.], [0, 0, -1]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmin(m, axis=0), np.array([2, 1, 0]))
    True
    >>> ac(safe_argmin(m, axis=1), np.array([2, 1, 2]))
    True
    >>> m = np.array([[1., 1, -1], [1, 0, 1], [0, 0, -1]])
    >>> m = KarmaSparse(m, format="csc")
    >>> ac(safe_argmin(m, axis=0), np.array([2, 1, 0]))
    True
    >>> ac(safe_argmin(m.tocsr(), axis=0), np.array([2, 1, 0]))
    True
    >>> m = np.array([[1, 1, 0, 0], [1, 0, 1, 1], [-1, 0, -1, 0]])
    >>> m = KarmaSparse(m, format="csr")
    >>> ac(safe_argmin(m, axis=1), np.array([2, 1, 0]))
    True
    >>> m = np.array([[-2., 0, -1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csc")
    >>> ac(safe_argmin(m, axis=0), np.array([0, 0, 0]))
    True
    >>> m = np.array([[-2, 0, -1], [0, 2, 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csc")
    >>> ac(safe_argmin(m, axis=0), np.array([0, 0, 0]))
    True
    >>> m = np.array([[0., 0, -1], [0, 2, 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csc")
    >>> ac(safe_argmin(m, axis=0), np.array([0, 0, 0]))
    True
    >>> safe_argmin(m)
    (0, 2)
    >>> m = np.array([[-1., 1, -1], [-1, 2, 1], [0, 2, 1]])
    >>> m = KarmaSparse(m, format="csc")
    >>> ac(safe_argmin(m, axis=0), np.array([0, 0, 0]))
    True
    >>> safe_argmin(m)
    (0, 0)
    """
    if is_scipysparse(matrix):
        return KarmaSparse(matrix).argmin(axis=axis)
    elif is_karmasparse(matrix):
        return matrix.argmin(axis=axis)
    else:
        if axis is None:
            return np.unravel_index(matrix.argmin(), matrix.shape)
        else:
            return matrix.argmin(axis=axis)


def safe_maximum(matrix, other):
    """
    >>> import scipy.sparse as sp
    >>> a = KarmaSparse(sp.rand(10, 5, 0.96), format="csc")
    >>> b = sp.rand(10, 5, 0.96)
    >>> np.allclose(safe_maximum(a, b).toarray(), safe_maximum(b.tocsr(), a).toarray())
    True
    >>> np.allclose(safe_maximum(a.toarray(), b), safe_maximum(a.tocsc(), b).toarray())
    True
    >>> np.allclose(safe_maximum(a.toarray(), b.toarray()), safe_maximum(a.tocsr(), b).toarray())
    True
    """
    if is_karmasparse(matrix) and is_karmasparse(other):
        return matrix.maximum(other)
    elif is_karmasparse(matrix):
        return matrix.maximum(KarmaSparse(other, format=matrix.format))
    elif is_karmasparse(other):
        return other.maximum(KarmaSparse(matrix, format=other.format))
    elif is_scipysparse(matrix) and is_scipysparse(other):
        return matrix.maximum(other)
    elif is_scipysparse(matrix):
        return np.asarray(matrix.maximum(other))
    elif is_scipysparse(other):
        return np.asarray(other.maximum(matrix))
    else:
        return np.maximum(matrix, other)


def safe_minimum(matrix, other):
    """
    >>> import scipy.sparse as sp
    >>> a = KarmaSparse(sp.rand(10, 5, 0.96), format="csc")
    >>> b = sp.rand(10, 5, 0.96)
    >>> np.allclose(safe_minimum(a, b).toarray(), safe_minimum(b.tocsr(), a).toarray())
    True
    >>> np.allclose(safe_minimum(a.toarray(), b), safe_minimum(a.tocsc(), b).toarray())
    True
    >>> np.allclose(safe_minimum(a.toarray(), b.toarray()), safe_minimum(a.tocsr(), b).toarray())
    True
    """
    if is_karmasparse(matrix) and is_karmasparse(other):
        return matrix.minimum(other)
    elif is_karmasparse(matrix):
        return matrix.minimum(KarmaSparse(other, format=matrix.format))
    elif is_karmasparse(other):
        return other.minimum(KarmaSparse(matrix, format=other.format))
    elif is_scipysparse(matrix) and is_scipysparse(other):
        return matrix.minimum(other)
    elif is_scipysparse(matrix):
        return np.asarray(matrix.minimum(other))
    elif is_scipysparse(other):
        return np.asarray(other.minimum(matrix))
    else:
        return np.minimum(matrix, other)


def safe_var(matrix, axis=None):
    """
    >>> import scipy.sparse as sp
    >>> a = np.array([[ 0.284,  0.41 ,  0.11 ,  0.54 ,  0.99 ],
    ...               [ 0.   ,  0.625,  0.439,  0.415,  0.06 ],
    ...               [ 0.709,  0.573,  0.743,  0.,  0.5  ],
    ...               [ 0.809,  0.982,  0.779,  0.99 ,  0.995]])
    >>> a = KarmaSparse(a, format="csc") if np.random.rand() > 0.5 else KarmaSparse(sp.rand(10, 5, 0.7))
    >>> np.allclose(safe_var(a.toarray()), safe_var(a.tocsc()))
    True
    >>> np.allclose(safe_var(a.toarray()), safe_var(a.tocsr()))
    True
    >>> np.allclose(a.toarray().var(axis=0), safe_var(a.tocsr(), axis=0))
    True
    >>> np.allclose(a.toarray().var(axis=1), safe_var(a.tocsr(), axis=1))
    True
    >>> np.allclose(a.toarray().var(axis=1), safe_var(a.tocsc(), axis=1))
    True
    >>> np.allclose(a.toarray().var(axis=0), safe_var(a.tocsc(), axis=0))
    True
    """
    if is_scipysparse(matrix):
        matrix = KarmaSparse(matrix)
    return matrix.var(axis=axis)


def safe_multiply(x, y, dense_output=False):
    """
    >>> import scipy.sparse as sp
    >>> x = np.array(range(6)).reshape((2,3))
    >>> y = np.array(range(2,8)).reshape((2,3))
    >>> ac = np.allclose
    >>> ac(x * y, safe_multiply(x, y))
    True
    >>> ac(safe_multiply(x, y), safe_multiply(sp.csc_matrix(x), y).toarray())
    True
    >>> ac(safe_multiply(x, y), safe_multiply(sp.csr_matrix(x), y).toarray())
    True
    >>> ac(safe_multiply(x, y), safe_multiply(KarmaSparse(x).tocsc(), y).toarray())
    True
    >>> ac(safe_multiply(x, y), safe_multiply(KarmaSparse(x).tocsr(), y).toarray())
    True
    """
    if is_scipysparse(x):
        x = KarmaSparse(x)
    if is_scipysparse(y):
        y = KarmaSparse(y)

    if is_karmasparse(y) and not is_karmasparse(x):
        return y * x
    else:
        return x * y


def kl_div(mat1, mat2):
    """
    >>> import scipy.sparse as sp
    >>> mat1 = np.random.rand(3, 3)
    >>> kl_div(mat1, mat1)
    0.0
    >>> mat1 = sp.eye(4, format="csc")
    >>> mat1[1, 1] = 3
    >>> mat2 = sp.eye(4, format="csc")
    >>> mat2[3, 3] = 2
    >>> np.round(kl_div(mat1, mat2), 4) == 1.6027
    True
    """
    return idiv(mat1, mat2) - mat1.sum() + mat2.sum()


def idiv(mat1, mat2):
    """
    >>> import scipy.sparse as sp
    >>> mat1 = np.random.rand(3, 3)
    >>> idiv(mat1, mat1)
    0.0
    >>> mat1 = sp.eye(4, format="csc")
    >>> mat1[1, 1] = 3
    >>> mat2 = sp.eye(4, format="csc")
    >>> mat2[3, 3] = 2
    >>> np.round(idiv(mat1, mat2), 4) == 2.6027
    True
    >>> mat1 = KarmaSparse(np.random.rand(10,10), format="csc")
    >>> mat2 = KarmaSparse(np.random.rand(10,10), format="csr")
    >>> np.allclose(idiv(mat1, mat2), idiv(mat1.toarray(), mat2.toarray()))
    True
    """
    eps = 10 ** -11
    if is_scipysparse(mat1):
        mat1 = KarmaSparse(mat1)
    if is_scipysparse(mat2):
        mat2 = KarmaSparse(mat2)

    if is_karmasparse(mat1) and is_karmasparse(mat2):
        if mat1.format == mat2.format and np.all(mat1.indices == mat2.indices) \
           and np.all(mat1.indptr == mat2.indptr):  # simpler way on data only
            return idiv_flat(mat1.data, mat2.data, eps)
        else:
            mat1, mat2 = mat1.clip(eps), mat2.clip(eps)
            return (mat1 * (mat1 / mat2).log()).sum()
    elif isinstance(mat1, np.ndarray) and isinstance(mat2, np.ndarray):
        if mat1.ndim == 2:
            return idiv_2d(mat1, mat2, eps)
        elif mat1.ndim == 1:
            return idiv_flat(mat1, mat2, eps)
        else:
            mat1, mat2 = mat1.clip(eps), mat2.clip(eps)
            return (mat1 * np.log(mat1 / mat2)).sum()
    else:
        raise NotImplementedError()


def truncate_with_cutoff(matrix, cutoff):
    """
    >>> mat = np.arange(8).reshape(2,4)
    >>> np.allclose(truncate_with_cutoff(mat, 4), np.array([[0, 0, 0, 0], [4, 5, 6, 7]]))
    True
    >>> np.allclose(truncate_with_cutoff(KarmaSparse(mat), 4).toarray(),
    ...             np.array([[0, 0, 0, 0], [4, 5, 6, 7]]))
    True
    >>> import scipy.sparse as sp
    >>> np.allclose(truncate_with_cutoff(sp.csr_matrix(mat), 4).toarray(),
    ...             np.array([[0, 0, 0, 0], [4, 5, 6, 7]]))
    True
    """
    if is_scipysparse(matrix):
        matrix = KarmaSparse(matrix)
    if is_karmasparse(matrix):
        return matrix.truncate_with_cutoff(cutoff)
    else:
        return threshold(matrix, threshmin=cutoff, threshmax=None, newval=0)


def safe_dot(x, y, mat_mask=None, mask_mode="last", dense_output=None):
    """
    >>> ac = np.allclose
    >>> import scipy.sparse as sp
    >>> x = KarmaSparse(sp.rand(3, 4, 0.5, format="csr"))
    >>> y = KarmaSparse(sp.rand(4, 5, 0.6, format="csc") * 1000)
    >>> ac(safe_dot(x, y, dense_output=True), x.dot(y).toarray())
    True
    >>> ac(safe_dot(x, y, dense_output=True), safe_dot(x.toarray(), y))
    True
    >>> ac(safe_dot(x.toarray(), y), safe_dot(x.toarray(), y.toarray()))
    True
    >>> ac(safe_dot(x.toarray(), y.toarray()), safe_dot(x, y.toarray()))
    True
    >>> ac(safe_dot(x, y.toarray()), safe_dot(x.to_scipy_sparse(), y.toarray()))
    True
    """
    if x.shape[1] != y.shape[0]:
        raise SparseUtilsException("Inner dimensions have to be the same " +
                                   "when multiplying matrices : {} != {}".
                                   format(x.shape[1], y.shape[0]))
    if mat_mask is not None:
        return mask_dot(x, y, mat_mask, mask_mode)

    # z = x * y if isinstance(x, np.ndarray) and is_scipysparse(y) else x.dot(y)
    if is_scipysparse(x):
        x = KarmaSparse(x)
    if is_scipysparse(y):
        y = KarmaSparse(y)

    if is_karmasparse(x) and is_karmasparse(y):
        z = x.dot(y)
    elif is_karmasparse(x):
        z = x.dense_dot_right(y)
    elif is_karmasparse(y):
        z = y.dense_dot_left(x)
    else:  # dense dense dot
        z = x.dot(y)
    return z.toarray() if dense_output and is_karmasparse(z) else z


def mask_dot(x, y, mat_mask, mask_mode="last", dense_output=False):
    """
    This util class computes a masked version of x.dot(y).
    - x, y: dense or sparse matrices
    - mat_mask: dense or sparse (CSR, CSC, COO) matrix, or coordinate
        (rows, cols) tuple.

    - mask_mode is one of (None, 'divide', 'multiply', 'add', 'subtract', 'max', 'min')
        if "last" (default) returns x.dot(y) masked by mat_mask
        if an operation "op" returns pointwise op(mat_mask,  x.dot(y))
        eg. 'divide' mat_mask[i,j] / x.dot(y)[i,j]

    - dense_output: variable to force the output to be sparse or dense.
    Implementation notes :
    - by default, the output format is set according to mat_mask format.
    >>> import scipy.sparse as sp
    >>> x, y = np.random.rand(6).reshape(2,3), np.random.rand(12).reshape(3,4)
    >>> mat_mask = sp.csr_matrix(np.repeat([0, 1, 1, 0], 2).reshape(2,4))
    >>> true_dot = np.zeros((2,4))
    >>> true_dot[mat_mask.nonzero()] = x.dot(y)[mat_mask.nonzero()]
    >>> np.allclose(mask_dot(x, y, mat_mask).toarray(), true_dot)
    True
    >>> x, y = sp.csr_matrix(x), sp.csr_matrix(y)
    >>> np.allclose(mask_dot(x, y, mat_mask).toarray(), true_dot)
    True
    >>> np.allclose(mask_dot(x, y.toarray(), mat_mask).toarray(), true_dot)
    True
    >>> np.allclose(mask_dot(x.toarray(), y, mat_mask).toarray(), true_dot)
    True
    >>> np.allclose(mask_dot(x.toarray(), y, mat_mask.toarray()), true_dot)
    True
    >>> x = sp.rand(10, 10, 0.2, format="csr")
    >>> y = sp.rand(10, 10, 0.2, format="csc")
    >>> np.allclose(mask_dot(x, y, x.dot(y)).toarray(), x.toarray().dot(y.toarray()))
    True
    >>> np.allclose(mask_dot(x, y, x.dot(y).toarray()), x.toarray().dot(y.toarray()))
    True
    >>> x_d = np.array(range(9)).reshape(3,3)
    >>> x = sp.csr_matrix(x_d)
    >>> y_d = np.array([[0, 0, 0], [1, 0., 0], [0., 1, 0]])
    >>> y = sp.csc_matrix(y_d)
    >>> m = np.ones((3, 3))
    >>> np.allclose(x.dot(y).toarray(), mask_dot(x, y, m))
    True
    >>> np.allclose(x.dot(y).toarray(),
    ...     mask_dot(x, y, sp.coo_matrix(m), dense_output=True))
    True
    >>> np.allclose(x.dot(y).toarray(),
    ...     mask_dot(x, y, m.nonzero(), dense_output=True))
    True
    >>> np.allclose(x.dot(y).toarray(), mask_dot(x, y, m, dense_output=True))
    True
    """
    # Check boundaries
    if x.shape[1] != y.shape[0]:
        raise SparseUtilsException("Inner dimensions have to be the same when multiplying matrices : {} != {}".
                                   format(x.shape[1], y.shape[0]))
    output_shape = x.shape[0], y.shape[1]

    is_dense = isinstance(mat_mask, np.ndarray)
    is_matrix_mask = is_scipysparse(mat_mask) or is_dense or is_karmasparse(mat_mask)
    if is_matrix_mask:
        if mat_mask.shape != output_shape:
            raise SparseUtilsException("Wrong shape of mask matrix  {} != {}".
                                       format(mat_mask.shape, output_shape))
    else:
        if (np.max(mat_mask[0]) > x.shape[0]) or (np.max(mat_mask[1]) > y.shape[1]):
            raise SparseUtilsException("Too large mask inputs")

    xx = KarmaSparse(x) if is_scipysparse(x) else x
    yy = KarmaSparse(y) if is_scipysparse(y) else y
    if not is_karmasparse(mat_mask):
        mat_mask = KarmaSparse(mat_mask)
    res = mat_mask.mask_dot(xx, yy, mask_mode)
    if dense_output or is_dense:
        return res.toarray()
    else:
        return res


def normalize(matrix, norm='l1', axis=1, invpow=1., invlog=0., threshold=None, width=1.):
    """
    >>> from sklearn.preprocessing import normalize as sk_normalize
    >>> import scipy.sparse as sp
    >>> matrix = sp.rand(10, 1, 0.3, format="csr")
    >>> np.allclose(sk_normalize(matrix, norm='l1', axis=1).toarray(),
    ...             normalize(KarmaSparse(matrix), norm='l1', axis=1).toarray())
    True
    >>> np.allclose(sk_normalize(matrix.toarray(), norm='l1', axis=1),
    ...             normalize(matrix, norm='l1', axis=1).toarray())
    True
    >>> np.allclose(sk_normalize(matrix.toarray(), norm='l1', axis=1),
    ...             normalize(matrix.toarray(), norm='l1', axis=1))
    True
    >>> np.allclose(sk_normalize(matrix, norm='l1', axis=0).toarray(),
    ...             normalize(KarmaSparse(matrix), norm='l1', axis=0).toarray())
    True
    >>> np.allclose(sk_normalize(matrix, norm='l1', axis=0).toarray(),
    ...             normalize(matrix.toarray(), norm='l1', axis=0))
    True
    >>> np.allclose(sk_normalize(matrix, norm='l2', axis=0).toarray(),
    ...             normalize(matrix.toarray(), norm='l2', axis=0))
    True
    >>> all([x in [0, 1] for x in
    ...      np.round(safe_max(normalize(KarmaSparse(matrix), norm='linf', axis=0), axis=0), 3)])
    True
    >>> all([x in [0, 1] for x in
    ...      np.round(safe_max(normalize(KarmaSparse(matrix), norm='linf', axis=1), axis=1), 3)])
    True
    """
    mapping_name = {'L1': 'l1', 'L2': 'l2', 'Linf': 'linf'}
    norm = mapping_name.get(norm, norm)
    if norm not in ('l1', 'l2', 'linf'):
        raise ValueError("'%s' is not a supported norm" % norm)
    if axis not in [0, 1, None]:
        raise ValueError("'%d' is not a supported axis" % axis)

    if is_scipysparse(matrix):
        matrix = KarmaSparse(matrix)
    if is_karmasparse(matrix):
        return matrix.normalize(norm=norm, axis=axis, invpow=invpow, invlog=invlog,
                                threshold=threshold, width=width)
    # numpy.array
    if norm == 'l1':
        norms = np.abs(matrix).sum(axis=axis)
    elif norm == 'l2':
        norms = np.sqrt(np.sum(matrix ** 2, axis=axis))
    elif norm == 'linf':
        norms = np.max(np.abs(matrix), axis=axis)

    if axis is None:
        return matrix / float(norms) if norms != 0 else matrix.copy()

    norms[norms == 0.0] = 1.0
    factor = np.power(norms, invpow)
    factor *= np.power(np.log1p(norms), invlog)
    if threshold is not None:
        factor /= logit(norms, shift=threshold, width=width)
    if axis == 1:
        return matrix.copy().astype(np.float) / factor[:, np.newaxis]
    elif axis == 0:
        return matrix.copy().astype(np.float) / factor[np.newaxis, :]


def number_nonzero(matrix):
    """
    For a given matrix, returns the number of non-zero elements.
    :param matrix: a matrix, of type dense (numpy) or sparse (scipy.sparse)
    :return: returns the number of non-zero element

    Exemples: ::

        >>> from random import randint
        >>> import scipy.sparse as sp
        >>> n, m = randint(2, 10), randint(2, 10)
        >>> density = 0.5
        >>> matrix = KarmaSparse(sp.rand(n, m, density=density))
        >>> nnz = number_nonzero(matrix)
        >>> n * m * density - 1 <= nnz <= n * m * density + 1
        True
        >>> n, m = randint(2, 10), randint(2, 10)
        >>> density = 0.5
        >>> matrix = sp.rand(n, m, density=density).toarray()
        >>> nnz = number_nonzero(matrix)
        >>> n * m * density - 1 <= nnz <= n * m * density + 1
        True
    """
    return matrix.nnz if is_scipysparse(matrix) or is_karmasparse(matrix)\
        else np.count_nonzero(matrix)


def sparse_quantiles(matrix, nb, axis):
    """
    >>> import scipy.sparse as sp
    >>> mat =  KarmaSparse(sp.csr_matrix(np.arange(20).reshape(2,10)))
    >>> sparse_quantiles(mat, 2, 1).toarray()
    array([[ 0.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.,  2.],
           [ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.,  2.]])
    """
    bins = np.arange(nb, dtype=np.float) / nb

    if axis == 0:
        matrix = matrix.tocsc()
    else:
        matrix = matrix.tocsr()

    data = np.hstack([np.digitize(matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]],
                                  mquantiles(matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]], bins))
                      for i in xrange(matrix.indptr.shape[0] - 1)])
    if axis == 0:
        return KarmaSparse((data, matrix.indices, matrix.indptr),
                           shape=matrix.shape, format="csc")
    else:
        return KarmaSparse((data, matrix.indices, matrix.indptr),
                           shape=matrix.shape, format="csr")


def truncated_dot(mat1, mat2, nb):
    """
    >>> mat1 = np.random.rand(30, 10)
    >>> mat2 = np.random.rand(10, 14)
    >>> truncated_dot(mat1, mat2, nb=15).shape == (30, 14)
    True
    >>> np.allclose(truncated_dot(mat1, mat2, nb=4).toarray(),
    ...             truncate_by_count(mat1.dot(mat2), 4, axis=1))
    True
    >>> np.allclose(truncated_dot(KarmaSparse(mat1).tocsr(), mat2, nb=2).toarray(),
    ...             truncated_dot(mat1, KarmaSparse(mat2).tocsc(), nb=2).toarray())
    True
    """
    if not is_karmasparse(mat1):
        mat1 = KarmaSparse(mat1)
    if not is_karmasparse(mat2):
        mat2 = KarmaSparse(mat2)
    return mat1.sparse_dot_top(mat2, nb)


def align_along_axis(matrix, indices, axis, extend=False):
    """
    >>> from karma.core.labeledmatrix.utils import aeq
    >>> matrix = np.random.rand(10, 5)
    >>> aeq(matrix[[3,2]], align_along_axis(matrix, [3,2], 1))
    True
    >>> aeq(matrix[:, [3,2]], align_along_axis(matrix, [3,2], 0))
    True
    >>> aeq(matrix[[3,2]], align_along_axis(matrix, [3,2, 10], 1, True)[:2])
    True
    >>> aeq(np.zeros((1, 5)), align_along_axis(matrix, [3,2, 10], 1, True)[-1:])
    True
    >>> align_along_axis(matrix, [3, 2, 10], 1, True).shape
    (3, 5)
    >>> aeq(matrix[:,[3,2]], align_along_axis(matrix, [5, 3, 2], 0, True)[:, 1:])
    True
    >>> aeq(np.zeros((10, 1)), align_along_axis(matrix, [5, 3, 2], 0, True)[:,:1])
    True

    >>> ks = KarmaSparse(matrix)
    >>> aeq(ks[[3,2]], align_along_axis(ks, [3, 2], 1))
    True
    >>> aeq(ks[:, [3,2]], align_along_axis(ks, [3, 2], 0))
    True
    >>> aeq(ks[[3,2]], align_along_axis(ks, [3, 2, 10], 1, True)[:2])
    True
    >>> aeq(np.zeros((1, 5)), align_along_axis(ks, [3, 2, 10], 1, True)[-1:])
    True
    >>> align_along_axis(ks, [3,2, 10], 1, True).shape
    (3, 5)
    >>> aeq(ks[:,[3,2]], align_along_axis(ks, [5, 3, 2], 0, True)[:, 1:])
    True
    >>> aeq(np.zeros((10, 1)), align_along_axis(ks, [5, 3, 2], 0, True)[:,:1])
    True

    """
    def _numpy_zeros_extension(matrix, nx, ny):
        if nx > 0 and ny > 0:
            return block_diag(matrix, np.zeros((nx, ny), dtype=matrix.dtype))
        elif nx > 0:
            return np.concatenate((matrix, np.zeros((nx, matrix.shape[1]),
                                                    dtype=matrix.dtype)), axis=0)
        elif ny > 0:
            return np.concatenate((matrix, np.zeros((matrix.shape[0], ny),
                                                    dtype=matrix.dtype)), axis=1)
        else:
            return matrix

    if axis == 1:
        if extend:
            if np.isscalar(indices):  # simple extension by adding zeros rows
                if is_karmasparse(matrix):
                    matrix = matrix.extend((matrix.shape[0] + indices, matrix.shape[1]))
                else:
                    matrix = _numpy_zeros_extension(matrix, indices, 0)
                return matrix
            else:
                if is_karmasparse(matrix):
                    matrix = matrix.extend((matrix.shape[0] + 1, matrix.shape[1]), copy=False)
                else:
                    matrix = _numpy_zeros_extension(matrix, 1, 0)
        return matrix[indices]
    elif axis == 0:
        if extend:
            if np.isscalar(indices):  # simple extension by adding zeros columns
                if is_karmasparse(matrix):
                    matrix = matrix.extend((matrix.shape[0], matrix.shape[1] + indices))
                else:
                    matrix = _numpy_zeros_extension(matrix, 0, indices)
                return matrix
            else:
                if is_karmasparse(matrix):
                    matrix = matrix.extend((matrix.shape[0], matrix.shape[1] + 1), copy=False)
                else:
                    matrix = _numpy_zeros_extension(matrix, 0, 1)
        return matrix[:, indices]
    else:
        raise ValueError("Axis should one of [0,1], got {}".format(axis))


def safe_add(matrix1, matrix2):
    if is_scipysparse(matrix1):
        matrix1 = KarmaSparse(matrix1)
    if is_scipysparse(matrix2):
        matrix2 = KarmaSparse(matrix2)
    if np.isscalar(matrix1) or np.isscalar(matrix2):
        return matrix1 + matrix2
    elif is_karmasparse(matrix1) and not is_karmasparse(matrix2):
        return matrix1 + KarmaSparse(matrix2, format=matrix1.format)
    elif is_karmasparse(matrix1) == is_karmasparse(matrix2):
        return matrix1 + matrix2
    elif is_karmasparse(matrix2) and not is_karmasparse(matrix1):
        return matrix2 + KarmaSparse(matrix1, format=matrix2.format)
    else:
        raise ValueError('Unknown matrix format')


def anomaly(matrix, skepticism):
    """
    >>> matrix = np.array([[2, 2, 1, 1], [3, 2, 0, 0], [3, 0, 1, 1]])
    >>> np.round(anomaly(matrix, skepticism=0.5), 3)
    array([[-0.089,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.045, -0.509, -0.509],
           [ 0.   , -0.974,  0.   ,  0.   ]])
    >>> matrix = np.array([[2, 2, 1, 1, 0], [3, 2, 0, 0, 0], [3, 0, 1, 1, 0]])
    >>> np.round(anomaly(matrix, skepticism=0.01), 3)
    array([[-0.68 ,  0.395,  0.324,  0.324, -0.   ],
           [ 0.391,  0.681, -4.502, -4.502, -0.   ],
           [ 0.391, -5.002,  0.547,  0.547, -0.   ]])
    >>> anomaly(np.random.rand(3, 1), skepticism=0.01)
    array([[ 0.],
           [ 0.],
           [ 0.]])
    """
    if is_karmasparse(matrix) or is_scipysparse(matrix):
        # TODO
        raise NotImplementedError()
    assert matrix.ndim == 2
    assert skepticism > 0
    if matrix.shape[1] == 1:
        return np.zeros_like(matrix, dtype=np.float)
    eps = 10 ** -10
    matrix = 1. * matrix
    total = np.atleast_2d(matrix.sum(axis=0))
    freq = normalize(total, axis=None, norm="l1").clip(eps)
    expected = np.atleast_2d(matrix.sum(axis=1)).transpose().dot(freq)  # warning : dense result
    error = skepticism * safe_maximum(expected, matrix) ** 0.5
    diff = expected - matrix
    error = safe_multiply(safe_minimum(error, np.abs(diff)), np.sign(diff))
    corrected = matrix + error
    frequency = corrected / np.atleast_2d(expected.sum(axis=1)).transpose()
    return np.log(frequency * (1 - freq) / freq / (1 - frequency))


def rank_dispatch(matrix, maximum_pressure, max_rank, max_volume):
    """
    Warning : works as expected only on nonzeros scores
    >>> matrix = np.array([[2, 2], [1, 1.2], [0, 0]])
    >>> rank_dispatch(matrix, 1, max_rank=np.array([3,3]), max_volume=np.array([3,3])).toarray()
    array([[ 2.,  0.],
           [ 1.,  0.],
           [ 0.,  0.]])

    """
    if is_karmasparse(matrix) or is_scipysparse(matrix):  # TODO
        raise NotImplementedError()
    # WARNING: Matrix is currently converted to dense, sparse_rannk_dispatch to be implemented
    return matrix_rank_dispatch(np.asarray(matrix), maximum_pressure, max_rank, max_volume)


def argmax_dispatch(matrix, maximum_pressure, max_rank, max_volume):
    """
    Return KarmaSparse matrix of dispatched rows over columns according to argmax score.

    >>> matrix = np.array([[0.8, 0.3], [0.4, 0.5], [0., 0.1]])
    >>> volumes = np.array([1, 1])
    >>> ranks = np.array([2, 2])
    >>> sparse_argmax_dispatch(KarmaSparse(matrix), maximum_pressure=1, max_rank=ranks, max_volume=volumes).toarray()
    array([[ 0.8,  0. ],
           [ 0. ,  0.5],
           [ 0. ,  0. ]])
    """
    nb_user, nb_topic = matrix.shape
    assert nb_topic == max_volume.shape[0]
    assert nb_topic == max_rank.shape[0]
    assert maximum_pressure > 0

    max_volume = np.minimum(max_rank, max_volume)
    max_rank = np.minimum(max_rank, np.sum(max_volume) * maximum_pressure + 1)

    if 1. * np.sum(max_rank) / nb_topic / nb_user < 0.5:  # arbitrary density constant
        matrix = truncate_by_count(matrix, max_rank, axis=0)

    # WARNING: Matrix is currently converted to sparse by default,
    #          dense_argmax_dispatch to be implemented
    matrix = KarmaSparse(matrix, copy=False)
    return sparse_argmax_dispatch(matrix, maximum_pressure, max_rank, max_volume)


def safe_hstack(args):
    if any(is_karmasparse(x) for x in args):
        return ks_hstack(args)
    else:
        return np.hstack(args)


def safe_vstack(args):
    if any(is_karmasparse(x) for x in args):
        return ks_vstack(args)
    else:
        return np.vstack(args)


def types_promotion(args):
    return [np.asarray(X, dtype=np.promote_types(X.dtype, np.float32)) if isinstance(X, np.ndarray)
            else (np.asarray(X) if not is_karmasparse(X) and not is_scipysparse(X) else X)
            for X in args]

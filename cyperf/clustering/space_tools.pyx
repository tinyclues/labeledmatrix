#cython: embedsignature=True
#cython: nonecheck=True
#cython: overflowcheck=True
#cython: unraisable_tracebacks=True

import numpy as np
from cython.parallel import prange
cimport cython
from cyperf.tools.types import ITYPE, LTYPE, FTYPE as DTYPE
cdef DTYPE_t INF = np.inf

METRIC_LIST = ['euclidean', 'sqeuclidean', 'cityblock', 'chebychev']

METRIC_DICT['euclidean'] = euclidean_distance
METRIC_DICT['sqeuclidean'] = sqeuclidean_distance
METRIC_DICT['cityblock'] = manhattan_distance
METRIC_DICT['chebychev'] = chebychev_distance
METRIC_DICT['idiv'] = idiv_distance
METRIC_DICT['scalar_product'] = scalar_product


cdef metric_func_type get_distance(string metric) except *:
    if METRIC_DICT.count(metric):
        return METRIC_DICT[metric]
    else:
        raise ValueError("unrecognized metric")


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] pairwise_flat(mat, string metric='euclidean'):
    """
    >>> pairwise_flat([[1, 2], [1, 2], [1, 3]])
    array([0., 1., 1.], dtype=float32)
    """
    cdef const DTYPE_t[:, ::1] X = np.asarray(mat, dtype=DTYPE, order="C")
    cdef ITYPE_t n_dim = X.shape[1]
    cdef LTYPE_t n_samples = X.shape[0], i, j, it
    cdef DTYPE_t[::1] D = np.zeros((n_samples - 1) * n_samples // 2, dtype=DTYPE, order="C")
    cdef metric_func_type dist_func = get_distance(metric)

    for i in prange(n_samples, nogil=True, schedule="dynamic"):
        it = i * (n_samples - 1) - i * (i + 1) // 2 - 1
        for j in xrange(i+1, n_samples):
            D[it + j] = dist_func(&X[i, 0], &X[j, 0], n_dim)
    out = np.asarray(D)
    return out


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] pairwise_square(mat, string metric='euclidean'):
    """
    >>> pairwise_square([[1, 2], [1, 2], [1, 3]])
    array([[0., 0., 1.],
           [0., 0., 1.],
           [1., 1., 0.]], dtype=float32)
    """
    cdef const DTYPE_t[:, ::1] X = np.asarray(mat, dtype=DTYPE, order="C")
    cdef ITYPE_t n_dim = X.shape[1], n_samples = X.shape[0], i, j
    cdef DTYPE_t[:, ::1] D = np.zeros((n_samples, n_samples), dtype=DTYPE, order="C")
    cdef DTYPE_t tmp
    cdef metric_func_type dist_func = get_distance(metric)

    for i in prange(n_samples, nogil=True, schedule="dynamic"):
        for j in xrange(i+1, n_samples):
            tmp = dist_func(&X[i, 0], &X[j, 0], n_dim)
            D[i, j] = tmp
            D[j, i] = tmp
    return np.asarray(D)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef DTYPE_t[::1] _ward_pairwise_flat(np.ndarray mat, np.ndarray weight):
    """
    >>> _ward_pairwise_flat(np.array([[1, 2], [1, 2], [1, 3]]), np.array(10, 5, 3))
    """

    cdef const DTYPE_t[:, ::1] X = np.asarray(mat, dtype=DTYPE, order="C")
    cdef const DTYPE_t[::1] W = np.asarray(weight, dtype=DTYPE, order="C")
    cdef ITYPE_t n_dim = X.shape[1],
    cdef LTYPE_t n_samples = X.shape[0]
    cdef LTYPE_t i, j, it
    cdef DTYPE_t[::1] D = np.zeros((n_samples - 1) * n_samples // 2, dtype=DTYPE, order="C")
    cdef DTYPE_t tmp

    assert n_samples == weight.shape[0]

    for i in prange(n_samples, nogil=True, schedule="dynamic"):
        it = i * (n_samples - 1) - i * (i + 1) // 2 - 1
        for j in xrange(i+1, n_samples):
            tmp = sqeuclidean_distance(&X[i, 0], &X[j, 0], n_dim)
            D[it + j] = tmp * 2. * W[i] * W[j] / (W[i] + W[j])
    return np.asarray(D)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[dtype=ITYPE_t, ndim=1] fast_buddies(np.ndarray mat, string metric="sqeuclidean"):
    """
    >>> fast_buddies(np.array([[1, 2], [1, 2], [1, 3], [2, 2]]))
    array([1, 0, 0, 0], dtype=int32)
    """
    cdef:
        const DTYPE_t[:, ::1] X = np.asarray(mat, dtype=DTYPE, order="C")
        ITYPE_t n_samples = X.shape[0]
        ITYPE_t n_dim = X.shape[1]
        ITYPE_t i, j, ind
        DTYPE_t val, val_dist
        ITYPE_t[::1] D = np.empty(n_samples, dtype=ITYPE, order="C")
        metric_func_type dist_func = get_distance(metric)

    for i in prange(n_samples, nogil=True, schedule="dynamic"):
        val = INF
        ind = -1
        for j in xrange(n_samples):
            if j != i:
                val_dist = dist_func(&X[i, 0], &X[j, 0], n_dim)
                if val_dist < val:
                    ind = j
                    val = val_dist
        D[i] = ind
    return np.asarray(D)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] vector_distance(np.ndarray vector,
                                                        np.ndarray mat,
                                                        string metric="sqeuclidean"):
    """
    >>> vector_distance(np.array([1, 2]), np.array([[1, 2], [1, 2], [1, 3], [2, 2]]))
    array([0., 0., 1., 1.], dtype=float32)
    """

    cdef:
        const DTYPE_t[::1] vec = np.asarray(vector, dtype=DTYPE, order="C")
        const DTYPE_t[:, ::1] X = np.asarray(mat, dtype=DTYPE, order="C")
        ITYPE_t n_samples = X.shape[0]
        DTYPE_t[::1] D = np.zeros(n_samples, dtype=DTYPE, order="C")
        metric_func_type dist_func = get_distance(metric)
        ITYPE_t n_dim = X.shape[1]
        ITYPE_t i

    if n_dim != vec.shape[0]:
        raise Exception('Incompatible shapes : {} != {}'.format(n_dim, vec.shape[0]))
    for i in prange(n_samples, nogil=True, schedule="dynamic"):
        D[i] = dist_func(&vec[0], &X[i, 0], n_dim)
    return np.asarray(D)

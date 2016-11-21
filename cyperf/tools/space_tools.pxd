cimport cython
cimport numpy as np
from libc.math cimport sqrt, fabs, fmax
from numpy.math cimport logl

from types cimport ITYPE_t, LTYPE_t, cmap, string, FTYPE_t as DTYPE_t

ctypedef DTYPE_t (*metric_func_type)(DTYPE_t *, DTYPE_t *, ITYPE_t) nogil
cdef cmap[string, metric_func_type] METRIC_DICT


cdef metric_func_type get_distance(string metric) except *
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] pairwise_flat(mat, string metric=*)
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] pairwise_square(mat, string metric=*)
cdef DTYPE_t[::1] _ward_pairwise_flat(np.ndarray mat, np.ndarray weight)
cpdef np.ndarray[dtype=ITYPE_t, ndim=1] fast_buddies(np.ndarray mat, string metric=*)
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] vector_distance(np.ndarray vector, np.ndarray mat, string metric=*)


cdef inline DTYPE_t euclidean_distance(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t N) nogil:
    return sqrt(sqeuclidean_distance(x1, x2, N))


cdef inline DTYPE_t sqeuclidean_distance(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t N) nogil:
    cdef DTYPE_t tmp, d = 0
    cdef ITYPE_t i

    for i in xrange(N):
        tmp = x1[i] - x2[i]
        d = d + tmp * tmp
    return d


cdef inline DTYPE_t scalar_product(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t N) nogil:
    cdef DTYPE_t d = 0
    cdef ITYPE_t i

    for i in xrange(N):
        d = d + x1[i] * x2[i]
    return d


@cython.cdivision(True)
cdef inline DTYPE_t idiv_distance(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t N) nogil:
    cdef DTYPE_t d = 0, mm = 0.000001, m1, m2
    cdef ITYPE_t i

    for i in xrange(N):
        m1 = fmax(x1[i], mm)
        m2 = fmax(x2[i], mm)
        d = d + m1 * logl(m1 / m2)
    return d


cdef inline DTYPE_t manhattan_distance(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t N) nogil:
    cdef DTYPE_t d = 0, tmp
    cdef ITYPE_t i

    for i in xrange(N):
        tmp = x1[i] - x2[i]
        d = d + fabs(tmp)
    return d


cdef inline DTYPE_t chebychev_distance(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t N) nogil:
    cdef DTYPE_t d = 0, tmp, tmp1
    cdef ITYPE_t i

    for i in xrange(N):
        tmp = x1[i] - x2[i]
        tmp1 = fabs(tmp)
        d = fmax(d, tmp1)
    return d

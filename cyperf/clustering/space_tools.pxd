cimport cython
cimport numpy as np
from libc.math cimport sqrt, fabs, fmax
from numpy.math cimport logl

from cyperf.tools.types cimport ITYPE_t, LTYPE_t, FTYPE_t as DTYPE_t

# TODO: support fused type in SpaceTools
ctypedef DTYPE_t (*metric_func_type)(const DTYPE_t *, const DTYPE_t *, const ITYPE_t) nogil


cdef metric_func_type get_distance(str metric) except *
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] pairwise_flat(mat, str metric=*)
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] pairwise_square(mat, str metric=*)
cdef DTYPE_t[::1] _ward_pairwise_flat(np.ndarray mat, np.ndarray weight)
cpdef np.ndarray[dtype=ITYPE_t, ndim=1] fast_buddies(np.ndarray mat, str metric=*)
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] vector_distance(np.ndarray vector, np.ndarray mat, str metric=*)


cdef inline DTYPE_t euclidean_distance(const DTYPE_t* x1, const DTYPE_t* x2, const ITYPE_t N) nogil:
    return sqrt(sqeuclidean_distance(x1, x2, N))


cdef inline DTYPE_t sqeuclidean_distance(const DTYPE_t* x1, const DTYPE_t* x2, const ITYPE_t N) nogil:
    cdef DTYPE_t tmp, d = 0
    cdef ITYPE_t i

    for i in range(N):
        tmp = x1[i] - x2[i]
        d += tmp * tmp
    return d


cdef inline DTYPE_t scalar_product(const DTYPE_t* x1, const DTYPE_t* x2, const ITYPE_t N) nogil:
    cdef DTYPE_t d = 0
    cdef ITYPE_t i

    for i in range(N):
        d += x1[i] * x2[i]
    return d


@cython.cdivision(True)
cdef inline DTYPE_t idiv_distance(const DTYPE_t* x1, const DTYPE_t* x2, const ITYPE_t N) nogil:
    cdef DTYPE_t d = 0, mm = 0.000001, m1, m2
    cdef ITYPE_t i

    for i in range(N):
        m1 = fmax(x1[i], mm)
        m2 = fmax(x2[i], mm)
        d += m1 * logl(m1 / m2)
    return d


cdef inline DTYPE_t manhattan_distance(const DTYPE_t* x1, const DTYPE_t* x2, const ITYPE_t N) nogil:
    cdef DTYPE_t d = 0, tmp
    cdef ITYPE_t i

    for i in range(N):
        tmp = x1[i] - x2[i]
        d += fabs(tmp)
    return d


cdef inline DTYPE_t chebychev_distance(const DTYPE_t* x1, const DTYPE_t* x2, const ITYPE_t N) nogil:
    cdef DTYPE_t d = 0, tmp, tmp1
    cdef ITYPE_t i

    for i in range(N):
        tmp = x1[i] - x2[i]
        tmp1 = fabs(tmp)
        d = fmax(d, tmp1)
    return d

cimport cython
from cyperf.tools.types cimport ITYPE_t, LTYPE_t, cmap, string, bool, A, B, INT1, INT2, INT3, ITER, BOOL_t
from libc.math cimport log


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline bool all_equal(A * old, B * new, long n) nogil:
    cdef long i
    for i in xrange(n):
        if old[i] != new[i]:
            return 0
    return 1

@cython.cdivision(False)
cdef inline double idiv_pointwise(const double x, const double y) nogil:
    return x * log(x / y)

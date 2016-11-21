cimport cython
from cyperf.tools.types cimport DTYPE_t, ITYPE_t, LTYPE_t, cmap, string, binary_func, bool, A, B
from scipy.linalg.cython_blas cimport daxpy, ddot
from libc.math cimport log
# Warning : type of blas routines should be compatible with DTYPE_t
# void daxpy(int *n, d *da, d *dx, int *incx, d *dy, int *incy)
# d ddot(int *n, d *dx, int *incx, d *dy, int *incy)


@cython.cdivision(True)
cdef inline DTYPE_t save_division(DTYPE_t x, DTYPE_t y) nogil:
    if y == 0:
        return 0
    else:
        return x / y

cdef inline DTYPE_t mult(DTYPE_t a, DTYPE_t b) nogil:
    return a * b

cdef inline DTYPE_t mmin(DTYPE_t a, DTYPE_t b) nogil:
    if a > b:
        return b
    else:
        return a

cdef inline DTYPE_t mmax(DTYPE_t a, DTYPE_t b) nogil:
    if a > b:
        return a
    else:
        return b

cdef inline DTYPE_t cadd(DTYPE_t a, DTYPE_t b) nogil:
    return a + b

cdef inline DTYPE_t cminus(DTYPE_t a, DTYPE_t b) nogil:
    return a - b

cdef inline DTYPE_t first(DTYPE_t a, DTYPE_t b) nogil:
    return a

cdef inline DTYPE_t last(DTYPE_t a, DTYPE_t b) nogil:
    return b

cdef inline DTYPE_t complement(DTYPE_t a, DTYPE_t b) nogil:
    if b == 0:
        return a
    else:
        return 0

cdef inline DTYPE_t trunc_diff(DTYPE_t a, DTYPE_t b) nogil:
    if a > b:
        return a - b
    else:
        return 0


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline bool all_equal(A * old, B * new, ITYPE_t n) nogil:
    cdef ITYPE_t i
    for i in xrange(n):
        if old[i] != new[i]:
            return 0
    return 1


cdef inline void axpy(ITYPE_t n, DTYPE_t a, DTYPE_t * x, DTYPE_t * y) nogil:
    cdef int m = 1
    daxpy(<int*>&n, &a, <double*>x, &m, <double*>y, &m)


cdef inline DTYPE_t _scalar_product(ITYPE_t n, DTYPE_t* x, DTYPE_t* y) nogil:
    cdef ITYPE_t k
    cdef DTYPE_t res = 0
    for k in xrange(n):
        res += x[k] * y[k]
    return res

cdef inline double _blas_scalar_product(int n, double* x, double* y) nogil:
    cdef int m = 1
    return ddot(<int*>&n, x, &m, y, &m)


cdef inline DTYPE_t scalar_product(ITYPE_t n, DTYPE_t * x, DTYPE_t * y) nogil:
    if n <= 40:  # turns out to be faster for small n
        return _scalar_product(n, x, y)
    else:
        return _blas_scalar_product(n, x, y)


cpdef DTYPE_t logistic(DTYPE_t x, DTYPE_t shift=*, DTYPE_t width=*) nogil

cdef DTYPE_t computed_quantile(DTYPE_t* data, DTYPE_t quantile, LTYPE_t size, LTYPE_t dim) nogil

cdef cmap[string, binary_func] REDUCERLIST
cdef binary_func get_reducer(string x) nogil except *


@cython.cdivision(False)
cdef inline double idiv_pointwise(const double x, const double y) nogil:
    return x * log(x / y)

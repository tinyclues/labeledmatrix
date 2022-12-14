cimport cython
from cyperf.tools.types cimport A, B
from libc.math cimport log



@cython.cdivision(False)
cdef inline double idiv_pointwise(const double x, const double y) nogil:
    return x * log(x / y)

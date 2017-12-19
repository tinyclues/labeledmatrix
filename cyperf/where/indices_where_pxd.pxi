cimport cython
cimport numpy as np
from libcpp.vector cimport vector

from cyperf.tools.types cimport bool, ITER
from cyperf.tools.getter cimport check_values


cdef class Vector:

    cdef vector[ITYPE_t] vector_buffer
    cdef long size(Vector self) nogil
    cdef inline void append(Vector self, ITYPE_t x) nogil
    cpdef np.ndarray[dtype=ITYPE_t, ndim=1] asarray(Vector self)


ctypedef fused VAL_T:
    int
    long
    float
    double
    object


ctypedef fused ITER_BIS:
    list
    tuple
    np.ndarray[dtype=int, ndim=1]
    np.ndarray[dtype=long, ndim=1]
    np.ndarray[dtype=float, ndim=1]
    np.ndarray[dtype=double, ndim=1]
    np.ndarray[dtype=object, ndim=1]
    object

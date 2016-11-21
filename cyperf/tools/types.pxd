cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t FTYPE_t
ctypedef np.int32_t ITYPE_t
ctypedef np.int64_t LTYPE_t
ctypedef DTYPE_t (*binary_func)(DTYPE_t, DTYPE_t) nogil

cdef DTYPE_t INF = np.inf


from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map as cmap

ctypedef fused A:
    ITYPE_t
    LTYPE_t
    FTYPE_t
    DTYPE_t
    bool

ctypedef fused B:
    ITYPE_t
    LTYPE_t
    FTYPE_t
    DTYPE_t
    bool

ctypedef fused ITER:
    list
    tuple
    np.ndarray[dtype=int, ndim=1]
    np.ndarray[dtype=long, ndim=1]
    np.ndarray[dtype=float, ndim=1]
    np.ndarray[dtype=double, ndim=1]
    np.ndarray[dtype=object, ndim=1]
    object

ctypedef fused INT1:
    ITYPE_t
    LTYPE_t

ctypedef fused INT2:
    ITYPE_t
    LTYPE_t

from openmp cimport omp_set_num_threads

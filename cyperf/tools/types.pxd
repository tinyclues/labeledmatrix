cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t FTYPE_t
ctypedef np.int32_t ITYPE_t
ctypedef np.int64_t LTYPE_t
ctypedef np.uint8_t BOOL_t
ctypedef DTYPE_t (*DTYPE_t_binary_func)(DTYPE_t, DTYPE_t) nogil


from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map as cmap

ctypedef fused A:
    BOOL_t
    ITYPE_t
    LTYPE_t
    FTYPE_t
    DTYPE_t

ctypedef fused B:
    BOOL_t
    ITYPE_t
    LTYPE_t
    FTYPE_t
    DTYPE_t


# Warning : using `char` or `np.int8_t` will lead to bad interpretation of "S1" numpy dtype
ctypedef fused ITER:
    np.ndarray[dtype=BOOL_t, ndim=1]
    np.ndarray[dtype=ITYPE_t, ndim=1]
    np.ndarray[dtype=LTYPE_t, ndim=1]
    np.ndarray[dtype=FTYPE_t, ndim=1]
    np.ndarray[dtype=DTYPE_t, ndim=1]
    np.ndarray[dtype=object, ndim=1]
    list
    tuple
    object

ctypedef fused ITER_BIS:
    np.ndarray[dtype=BOOL_t, ndim=1]
    np.ndarray[dtype=ITYPE_t, ndim=1]
    np.ndarray[dtype=LTYPE_t, ndim=1]
    np.ndarray[dtype=FTYPE_t, ndim=1]
    np.ndarray[dtype=DTYPE_t, ndim=1]
    np.ndarray[dtype=object, ndim=1]
    list
    tuple
    object

ctypedef fused ITER_NP:
    np.ndarray[dtype=BOOL_t, ndim=1]
    np.ndarray[dtype=ITYPE_t, ndim=1]
    np.ndarray[dtype=LTYPE_t, ndim=1]
    np.ndarray[dtype=FTYPE_t, ndim=1]
    np.ndarray[dtype=DTYPE_t, ndim=1]
    np.ndarray[dtype=object, ndim=1]

ctypedef fused INT1:
    ITYPE_t
    LTYPE_t

ctypedef fused INT2:
    ITYPE_t
    LTYPE_t

ctypedef fused INT3:
    ITYPE_t
    LTYPE_t

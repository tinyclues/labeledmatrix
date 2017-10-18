cimport numpy as np
cimport cython
from types cimport ITER, INT1, DTYPE_t, LTYPE_t, bool


ctypedef fused INDICE_t:
    np.ndarray[dtype=int, ndim=1]
    np.ndarray[dtype=long, ndim=1]
    list
    object

ctypedef fused ITER_t:
    np.ndarray[dtype=int, ndim=1]
    np.ndarray[dtype=long, ndim=1]
    np.ndarray[dtype=float, ndim=1]
    np.ndarray[dtype=double, ndim=1]
    np.ndarray[dtype=object, ndim=1]
    list
    object

ctypedef fused ITER_NP_t:
    np.ndarray[dtype=int, ndim=1]
    np.ndarray[dtype=long, ndim=1]
    np.ndarray[dtype=float, ndim=1]
    np.ndarray[dtype=double, ndim=1]
    np.ndarray[dtype=object, ndim=1]

cdef bool check_values(ITER values, dtype=*) except? False
cpdef list take_indices_on_iterable(ITER_t iterable, INDICE_t indices)
cpdef ITER_NP_t take_indices_on_numpy(ITER_NP_t ar, INDICE_t indices)
cpdef list apply_python_dict(dict mapping, ITER indices, object default, bool keep_same)
cpdef np.ndarray[dtype=np.int32_t, ndim=1] python_feature_hasher(ITER inp, int nb_feature)

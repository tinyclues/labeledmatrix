cimport numpy as np
cimport cython
from cyperf.tools.types cimport ITER, INT1, DTYPE_t, LTYPE_t, BOOL_t, bool, ITER_NP


ctypedef fused INDICE_t:
    np.ndarray[dtype=int, ndim=1]
    np.ndarray[dtype=long, ndim=1]
    list
    object


cdef bool check_values(ITER values, dtype=*) except? False
cpdef list take_indices_on_iterable(ITER iterable, INDICE_t indices)
cpdef ITER_NP take_indices_on_numpy(ITER_NP ar, INDICE_t indices)
cpdef list apply_python_dict(dict mapping, ITER indices, object default, bool keep_same)

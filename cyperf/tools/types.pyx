# distutils: language = c++
import numpy as np

ITYPE = np.int32
DTYPE = np.float64
FTYPE = np.float32
LTYPE = np.int64
BOOL = np.uint8


cdef DTYPE_t INF = np.inf


from openmp cimport omp_set_num_threads, omp_get_max_threads


def set_open_mp_num_thread(int n):
    omp_set_num_threads(max(n, 1))


def get_open_mp_num_thread():
    """
    >>> set_open_mp_num_thread(3)
    >>> get_open_mp_num_thread()
    3
    """
    return omp_get_max_threads()


cdef np.ndarray[char, ndim=2, mode="c"] safe_numpy_string_convertor(np.ndarray keys):
    assert keys.dtype.kind == 'S'
    return np.ascontiguousarray(keys).view(np.int8).reshape(len(keys), keys.dtype.itemsize)

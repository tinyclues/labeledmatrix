
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


cdef class FastHash:
        """
            fast support custom hash
            can be used for Exceptional values classes
             - about 4x faster than pure Python code
        """
        cdef long _hash

        def __init__(self):
            """
            >>> fh = FastHash()
            >>> hash(fh)
            557541288731312565
            """
            self._hash = hash(self.__class__.__name__) + 111111

        def __hash__(self):
            return self._hash

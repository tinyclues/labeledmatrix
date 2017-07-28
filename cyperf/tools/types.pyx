
import numpy as np

ITYPE = np.int32
DTYPE = np.float64
FTYPE = np.float32
LTYPE = np.int64


cdef DTYPE_t INF = np.inf


def set_open_mp_num_thread(int n):
    omp_set_num_threads(max(n, 1))


#
# Copyright tinyclues, All rights reserved
#
#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False
#cython: infer_types=True


cimport cython
cimport numpy as np
import numpy as np
from cython.parallel import prange
from cyperf.tools.types import ITYPE, FTYPE as DTYPE
from cyperf.tools import parallel_unique
from cyperf.clustering.heap import ActiveList, MinHeap

cdef DTYPE_t INF = np.inf

include "ward_tree.pxi"
include "tail_tree.pxi"

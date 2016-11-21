
cimport numpy as np
from cyperf.tools.types cimport ITYPE_t, bool, FTYPE_t as DTYPE_t

cdef class MinHeap:
    cdef:
        DTYPE_t[::1] distances
        ITYPE_t[::1] indices
        ITYPE_t[::1] reverse
        ITYPE_t n_nbrs

    cdef inline DTYPE_t dist_ind(self, ITYPE_t i) nogil

    cdef inline ITYPE_t argmin(self) nogil

    cdef inline DTYPE_t smallest(self) nogil

    cdef void heapify(self) nogil

    cdef ITYPE_t pop(self) nogil

    cdef void remove(self, ITYPE_t idx) nogil

    cdef void replace(self, ITYPE_t idxold, ITYPE_t idxnew, DTYPE_t val) nogil

    cdef void update(self, ITYPE_t idx, DTYPE_t val) nogil

    cdef void update_leq(self, ITYPE_t idx, DTYPE_t val) nogil

    cdef void update_geq(self, ITYPE_t idx, DTYPE_t val) nogil

    cdef void update_geq_(self, ITYPE_t i) nogil

    cdef void update_leq_(self, ITYPE_t i) nogil

    cdef void heap_swap(self, ITYPE_t i, ITYPE_t j) nogil


cdef class ActiveList:
    cdef ITYPE_t[::1] pred
    cdef ITYPE_t[::1] succ
    cdef ITYPE_t start
    cdef ITYPE_t size, nb

    cdef void remove(self, ITYPE_t idx) nogil

    cdef inline bool is_inactive(self, ITYPE_t idx) nogil

    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] get_list(self)

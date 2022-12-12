# distutils: language = c++
#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False
#cython: infer_types=True
#cython: embedsignature=True
#cython: overflowcheck=True
#cython: unraisable_tracebacks=True

import numpy as np
from cyperf.tools.types import ITYPE, FTYPE as DTYPE


cdef class MinHeap:

    def __cinit__(self, vector):

        self.n_nbrs = vector.shape[0]
        self.reverse = np.arange(self.n_nbrs, dtype=ITYPE)
        self.indices = np.asarray(self.reverse).copy()
        self.distances = np.array(vector, dtype=DTYPE, order='C')
        self.heapify()

    cdef inline DTYPE_t dist_ind(self, ITYPE_t i) nogil:
        cdef ITYPE_t j = self.indices[i]
        return self.distances[j]

    cdef inline ITYPE_t argmin(self) nogil:
        return self.indices[0]

    cdef inline DTYPE_t smallest(self) nogil:
        return self.dist_ind(0)

    cdef void heapify(self) nogil:
        cdef ITYPE_t idx
        for idx in range(self.n_nbrs // 2 - 1, -1, -1):
            self.update_geq_(idx)

    cdef ITYPE_t pop(self) nogil:
        cdef ITYPE_t ii = self.indices[0]

        if self.n_nbrs > 0:
            self.n_nbrs = self.n_nbrs - 1
            self.indices[0] = self.indices[self.n_nbrs]
            self.reverse[self.indices[0]] = 0
            self.update_geq_(0)
        return ii

    cdef void remove(self, ITYPE_t idx) nogil:
        if self.n_nbrs > 0:
            self.n_nbrs = self.n_nbrs - 1
            self.reverse[self.indices[self.n_nbrs]] = self.reverse[idx]
            self.indices[self.reverse[idx]] = self.indices[self.n_nbrs]
            if self.dist_ind(self.n_nbrs) <= self.distances[idx]:
                self.update_leq_(self.reverse[idx])
            else:
                self.update_geq_(self.reverse[idx])

    cdef void replace(self, ITYPE_t idxold, ITYPE_t idxnew, DTYPE_t val) nogil:
        self.reverse[idxnew] = self.reverse[idxold]
        self.indices[self.reverse[idxnew]] = idxnew
        if (val <= self.distances[idxold]):
            self.update_leq(idxnew, val)
        else:
            self.update_geq(idxnew, val)

    cdef void update(self, ITYPE_t idx, DTYPE_t val) nogil:
        if val <= self.distances[idx]:
            self.update_leq(idx, val)
        else:
            self.update_geq(idx, val)

    cdef void update_leq(self, ITYPE_t idx, DTYPE_t val) nogil:
        self.distances[idx] = val
        self.update_leq_(self.reverse[idx])

    cdef void update_geq(self, ITYPE_t idx, DTYPE_t val) nogil:
        self.distances[idx] = val
        self.update_geq_(self.reverse[idx])

    cdef void update_geq_(self, ITYPE_t i) nogil:
        cdef ITYPE_t j
        while True:
            j = 2 * i + 1
            if j >= self.n_nbrs:
                break
            if self.dist_ind(j) >= self.dist_ind(i):
                j = j + 1
                if (j >= self.n_nbrs) or (self.dist_ind(j) >= self.dist_ind(i)):
                    break
            else:
                if j + 1 < self.n_nbrs and (self.dist_ind(j+1) < self.dist_ind(j)):
                    j = j + 1
            self.heap_swap(i, j)
            i = j

    cdef void update_leq_(self, ITYPE_t i) nogil:
        while i > 0 and self.dist_ind(i) < self.dist_ind((i-1) // 2):
            self.heap_swap(i, (i-1) // 2)
            i = (i-1) // 2

    cdef void heap_swap(self, ITYPE_t i, ITYPE_t j) nogil:
        cdef ITYPE_t tmp
        tmp = self.indices[i]
        self.indices[i] = self.indices[j]
        self.indices[j] = tmp
        self.reverse[self.indices[i]] = i
        self.reverse[self.indices[j]] = j

    # for testing reason
    def get_arrays(self):
        return (np.asarray(self.distances),
                np.asarray(self.indices),
                np.asarray(self.reverse))


cdef class ActiveList:

    def __cinit__(self, ITYPE_t size):
        self.pred = np.zeros(size+1, dtype=ITYPE)
        self.succ = np.zeros(size+1, dtype=ITYPE)
        self.start = 0
        self.size = size
        self.nb = size
        for i in range(size):
            self.pred[i+1] = i
            self.succ[i] = i + 1

    cdef void remove(self, ITYPE_t idx) nogil:
        if not self.is_inactive(idx) and 0 <= idx < self.nb:
            if idx == self.start:
                self.start = self.succ[idx]
            else:
                self.succ[self.pred[idx]] = self.succ[idx]
                self.pred[self.succ[idx]] = self.pred[idx]
            self.succ[idx] = 0
            self.size -= 1

    cdef inline bool is_inactive(self, ITYPE_t idx) nogil:
        return self.succ[idx] == 0

    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] get_list(self):
        cdef ITYPE_t k, i = self.start
        cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] d = np.zeros(self.size, dtype=ITYPE)
        cdef ITYPE_t* dptr = &d[0]
        with nogil:
            for k in range(self.size):
                dptr[k] = i
                i = self.succ[i]
        return d

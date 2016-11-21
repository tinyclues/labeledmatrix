#
# Copyright tinyclues, All rights reserved
#

cdef class TailTree:

    def __cinit__(self, X not None, weights=None, ITYPE_t n_clusters=1,
                  string metric="sqeuclidean"):

        self.X = np.asarray(X, dtype=DTYPE, order='C')  # copy
        self.n = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.n_clusters = min(n_clusters, self.n)
        self.active_indices = ActiveList(self.n)
        self.dist_func = get_distance(metric)
        if weights is not None:
            weights = np.asarray(weights, dtype=DTYPE, order="C")
            assert weights.shape[0] == self.n
        else:
            weights = np.ones(self.n, dtype=DTYPE, order="C")
        self.weights = weights
        self.weights_copy = weights.copy()
        self.heap = MinHeap(self.weights)

    cdef MinVal argmin(self, ITYPE_t i0) nogil:
        cdef:
            ITYPE_t i = self.active_indices.start
            ITYPE_t ind = i
            DTYPE_t val_dist, val = INF
            DTYPE_t * vv = &self.X[i0, 0]
            MinVal mv

        while i < self.n:
            if i != i0:
                val_dist = self.dist_func(&self.X[i, 0], vv, self.dim)
                if val_dist < val:
                    ind = i
                    val = val_dist
            i = self.active_indices.succ[i]
        mv.ind = ind
        mv.val = val
        return mv

    @cython.cdivision(True)
    cdef void merge(self, ITYPE_t i0, ITYPE_t j0) nogil:
        cdef ITYPE_t k
        cdef DTYPE_t wi = self.weights[i0] / (self.weights[i0] + self.weights[j0])
        cdef DTYPE_t wj = self.weights[j0] / (self.weights[i0] + self.weights[j0])

        for k in xrange(self.dim):
            self.X[i0, k] = wi * self.X[i0, k] + wj * self.X[j0, k]
        self.weights[i0] = self.weights[i0] + self.weights[j0]
        self.active_indices.remove(j0)
        self.heap.update(i0, self.weights[i0])

    cpdef np.ndarray[dtype=DTYPE_t, ndim=2] build(self):
        cdef:
            ITYPE_t i0, j0, it = 0
            ITYPE_t size = self.n - self.n_clusters
            MinVal mv
            ITYPE_t[::1] linked_name = np.arange(self.n, dtype=ITYPE)
            DTYPE_t[:, ::1] link = np.empty((size, 4), dtype=DTYPE, order="C")

        with nogil:
            while self.active_indices.size > 1 and it < size:
                j0 = self.heap.pop()
                mv = self.argmin(j0)
                i0 = mv.ind
                self.merge(i0, j0)
                #update link
                link[it, 0] = fmin(linked_name[i0], linked_name[j0])
                link[it, 1] = fmax(linked_name[i0], linked_name[j0])
                link[it, 2], link[it, 3] = sqrt(mv.val), self.weights[i0]
                linked_name[i0] = self.n + it
                it = it + 1
        return np.asarray(link)

    def build_labels(self):
        labels = labels_builder(self.build(), self.n)
        return cluster_center(labels, self.weights_copy)

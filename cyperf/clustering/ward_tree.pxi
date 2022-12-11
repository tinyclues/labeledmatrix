#
# Copyright tinyclues, All rights reserved
#

cdef ITYPE_t MEMORY_DIM = 100000


def labels_builder(np.ndarray full_link not None, ITYPE_t n):
    """
    build labels from (possibly partial) linkage matrix.
    """
    cdef ITYPE_t u
    cdef np.ndarray[ITYPE_t, ndim=1] aux = np.hstack([np.arange(n, dtype=ITYPE), -np.ones(n, dtype=ITYPE)])
    cdef ITYPE_t[:,::1] link = full_link[:, :2].astype(ITYPE)

    if link.shape[0] > 0:
        for u in range(link.shape[0] - 1, -1, -1):
            if aux[u + n] != -1:
                aux[link[u, 0]] = aux[link[u, 1]] = aux[u + n]
            else:
                aux[link[u, 0]] = aux[link[u, 1]] = u + n
        link = None
        return np.searchsorted(parallel_unique(aux[:n]), aux[:n])
    else:
        return np.arange(n)


@cython.wraparound(True)
def cluster_center(labels, weights):
    """
    To choose the represent with the highest weight
    """
    assert labels.shape[0] == weights.shape[0]
    order = np.lexsort((weights, labels))[::-1]
    ordered_labels = labels[order]
    index = np.empty(labels.shape[0], np.bool_)
    index[0] = True
    index[1:] = ordered_labels[1:] != ordered_labels[:-1]
    return order[index][::-1][np.asarray(labels, dtype=ITYPE)]


def huffman_encoding(np.ndarray full_link not None):
    cdef ITYPE_t[:,::1] link = full_link[:, :2].astype(ITYPE)
    cdef ITYPE_t n = full_link.shape[0] + 1
    cdef list res = [""] * (2 * n + 1)
    cdef str one = "1", zero = "0"
    cdef ITYPE_t u

    for u in range(link.shape[0] - 1, -1, -1):
        res[link[u, 0]] += res[u + n] + zero
        res[link[u, 1]] += res[u + n] + one
    link = None
    return dict(enumerate(res[:n]))


cpdef IVEC traversal(ITYPE_t[:,::1] link, ITYPE_t node):
    cdef ITYPE_t d = node - link.shape[0] - 1
    cdef LTYPE_t i
    cdef IVEC vec, vec1, vec2
    if d < 0:
        vec.push_back(node)
    else:
        vec1 = traversal(link, link[d, 1])
        vec2 = traversal(link, link[d, 0])
        vec.reserve(vec1.size() + vec2.size())
        for i in range(<LTYPE_t>vec1.size()):
            vec.push_back(vec1[i])
        for i in range(<LTYPE_t>vec2.size()):
            vec.push_back(vec2[i])
    link = None
    return vec


cdef DTYPE_t simple_dist(const DTYPE_t[::1] matrix, IVEC rows, IVEC cols, ITYPE_t n):
    cdef ITYPE_t pt
    cdef ITYPE_t i, j
    cdef DTYPE_t mm = INF
    cdef DTYPE_t val = 0
    for i in rows:
        pt = pnt(i, n)
        for j in cols:
            if i > j:
                val = matrix[pnt(j, n) + i]
            elif j > i:
                val = matrix[pt + j]
            else:
                return 0.
            if mm > val:
                mm = val
    return mm


def huffman_encoding_reordering(np.ndarray full_link not None, np.ndarray dist_matrix):
    cdef:
        const DTYPE_t[::1] distance_matrix = np.asarray(dist_matrix, dtype=DTYPE, order='C')
        ITYPE_t n = full_link.shape[0] + 1
        np.ndarray[ITYPE_t, ndim=2, mode="c"] link = full_link[:, :2].astype(ITYPE)
        ITYPE_t left, right, uncle, u, f
        IVEC leaf_left, leaf_right, leaf_uncle
        ITYPE_t[::1] father = np.empty(2 * (n - 1), dtype=ITYPE)
        DTYPE_t mr, ml
        bool swap

    father[link[n-2, 0]] = father[link[n-2, 1]] = n - 2

    for u in range(n - 3, -1, -1):
        left, right = link[u, 0], link[u, 1]
        father[left] = father[right] = u

        if link[father[u + n], 0] == u + n:
            uncle = link[father[u + n], 1]
            swap = True
        else:
            uncle = link[father[u + n], 0]
            swap = False

        leaf_left = traversal(link, left)
        leaf_right = traversal(link, right)
        leaf_uncle = traversal(link, uncle)
        ml = simple_dist(distance_matrix, leaf_left, leaf_uncle, n)
        mr = simple_dist(distance_matrix, leaf_right, leaf_uncle, n)
        if (swap and ml < mr) or (ml > mr and not swap):
            link[u, 0], link[u, 1] = right, left
    return huffman_encoding(np.asarray(link))


cdef class WardTree:
    """
    Ward clustering based on feature matrix X.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        feature matrix  representing n_samples samples to be clustered

    weights : array_like object (optional)
        Multiplicity array. Is set to np.ones(n_samples) by default.

    n_clusters : desired number of clusters (Default = 1 - full tree).

    ---------------
    A linkage matrix is a standard form of encoding a hierarchical clustering.
    It has four columns, describing the series of hierarchical merges.
    The first two columns tell which nodes are about to be merged.
    The third is the distance between these two nodes/clusters.
    The fourth is the total size of the newly formed cluster.
    """

    def __cinit__(self, np.ndarray X not None, weights=None, ITYPE_t n_clusters=1):
        cdef ITYPE_t row

        self.X = X  # no copy here!
        self.n = X.shape[0]
        self.n_clusters = min(n_clusters, self.n)
        self.active_indices = ActiveList(self.n)

        if weights is not None:
            weights = np.asarray(weights, dtype=DTYPE, order="C")
            assert weights.shape[0] == self.n
        else:
            weights = np.ones(self.n, dtype=DTYPE, order="C")

        self.weights = weights
        self.weights_copy = np.array(self.weights, copy=True)
        self.distances = _ward_pairwise_flat(X, weights)

        self.min_dist = np.full(self.n, INF, dtype=DTYPE, order="C")
        self.argmin_dist = np.zeros(self.n, dtype=ITYPE, order="C")
        for row in prange(self.n, nogil=True, schedule="guided"):
            self.correct(row)

    cdef void correct(self, ITYPE_t row) nogil:
        cdef:
            ITYPE_t i = self.active_indices.start
            DTYPE_t* values = &self.distances[0]
            ITYPE_t armm = i
            DTYPE_t mm = INF
            DTYPE_t val
            LTYPE_t pt = pnt(row, self.n)

        while i < self.n:
            if i < row:
                val = values[pnt(i, self.n) + row]
                if val < mm:
                    armm = i
                    mm = val
            elif i > row:
                val = values[pt + i]
                if val < mm:
                    armm = i
                    mm = val
            i = self.active_indices.succ[i]

        self.min_dist[row] = mm
        self.argmin_dist[row] = armm

    cdef void ward_update(self, ITYPE_t i0, ITYPE_t j0) nogil:
        # We have to assume that i0 < j0
        cdef:
            DTYPE_t wi0 = self.weights[i0], wj0 = self.weights[j0]
            DTYPE_t ss = wi0 + wj0
            DTYPE_t wi
            ITYPE_t i
            LTYPE_t pti = pnt(i0, self.n)
            LTYPE_t ptj = pnt(j0, self.n)
            DTYPE_t * distances = &self.distances[0]
            DTYPE_t mm = distances[pti + j0]

        self.weights[i0] = ss  # updating weights

        i = self.active_indices.start
        while i < self.n:
            wi = self.weights[i]
            if i0 > i:
                distances[pnt(i, self.n) + i0] = (
                    (wi + wi0) * distances[pnt(i, self.n) + i0] +
                    (wi + wj0) * distances[pnt(i, self.n) + j0] -
                    wi * mm) / (wi + ss)
            elif i > i0 and i < j0:
                distances[pti + i] = ((wi + wi0) * distances[pti + i] +
                                      (wi + wj0) * distances[pnt(i, self.n) + j0] -
                                      wi * mm) / (wi + ss)
            elif i > j0:
                distances[pti + i] = ((wi + wi0) * distances[pti + i] +
                                      (wi + wj0) * distances[ptj + i] -
                                      wi * mm) / (wi + ss)
            i = self.active_indices.succ[i]

    cdef inline ITYPE_t global_argmin(self) nogil:
        cdef ITYPE_t i = self.active_indices.start
        cdef ITYPE_t armm = i
        cdef DTYPE_t val = self.min_dist[i]

        while i < self.n:
            if val > self.min_dist[i]:
                armm = i
                val = self.min_dist[i]
            i = self.active_indices.succ[i]
        return armm

    cdef inline ITYPE_t nn_chain(self) nogil:
        cdef ITYPE_t a, b, c, d

        if self.chain.empty():
            a = self.active_indices.start
        else:
            a = self.chain.back()

        b = self.argmin_dist[a]
        c = self.argmin_dist[b]

        if c == a:
            if not self.chain.empty():
                self.chain.pop_back()
            if a > b:
                return b
            else:
                return a
        while True:
            a = self.argmin_dist[c]
            if b == a:
                return min(a, c)
            self.chain.push_back(b)
            b, c = c, a

    cdef void mindist_update(self, ITYPE_t i0, ITYPE_t j0) nogil:
        cdef ITYPE_t i
        i = self.active_indices.start
        while i < self.n:
            if self.argmin_dist[i] == j0 or self.argmin_dist[i] == i0:
                self.correct(i)
            i = self.active_indices.succ[i]

    cdef void merge(self, ITYPE_t i0, ITYPE_t j0) nogil:
        self.active_indices.remove(j0)
        self.ward_update(i0, j0)
        self.mindist_update(i0, j0)

    cdef np.ndarray[dtype=DTYPE_t, ndim=2] build(self, bool chain=True):
        cdef:
            ITYPE_t iO, j0, it = 0
            DTYPE_t val
            ITYPE_t[::1] linked_name = np.arange(self.n, dtype=ITYPE)
            DTYPE_t[:, ::1] link = np.empty((self.n - self.n_clusters, 4),
                                            dtype=DTYPE, order="C")

        with nogil:
            if chain:
                self.chain.reserve(self.n)
                self.chain.push_back(self.global_argmin())
            while self.active_indices.size > 1 and it < self.n - self.n_clusters:
                if chain:
                    i0 = self.nn_chain()
                else:
                    i0 = self.global_argmin()
                j0, val = self.argmin_dist[i0], sqrt(self.min_dist[i0])
                self.merge(i0, j0)
                #update link
                link[it, 0] = fmin(linked_name[i0], linked_name[j0])
                link[it, 1] = fmax(linked_name[i0], linked_name[j0])
                link[it, 2], link[it, 3] = val, self.weights[i0]
                linked_name[i0] = self.n + it
                it += 1
        return np.asarray(link)

    def build_linkage(self, bool chain=True):
        if self.link is None:
            if self.n_clusters > 1:  # to have correct order w.r.t. cluster size
                chain = False
            self.link = self.build(chain)
        return self.link

    def build_labels(self):
        labels = labels_builder(self.build_linkage(), self.n)
        return cluster_center(labels, self.weights_copy)

    def build_huffman(self):
        if self.n_clusters > 1:
            raise Exception("Complete tree building is " +
                            "impossible when nb_cluster={} > 1".format(self.nb_cluster))
        d = huffman_encoding(self.build_linkage())
        return d

    def build_huffman_ordering(self):
        cdef np.ndarray[dtype=DTYPE_t, ndim=1] distances_copy
        if self.n_clusters > 1:
            raise Exception("Complete tree building is " +
                            "impossible when nb_cluster={} > 1".format(self.nb_cluster))

        if self.link is None and self.n < MEMORY_DIM:
            distances_copy = np.array(self.distances)  # take a copy
            d = huffman_encoding_reordering(self.build_linkage(), distances_copy)
            return d
        else:  # recompute to save memory
            link = self.build_linkage().copy()
            self.distances = None
            self.weights = np.array(self.weights_copy)
            self.distances = _ward_pairwise_flat(self.X, np.asarray(self.weights))
            d = huffman_encoding_reordering(link, np.asarray(self.distances))
            return d

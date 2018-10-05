# cython: embedsignature=True
# cython: nonecheck=True
# cython: overflowcheck=True


import numpy as np
from cython.parallel import prange
from cyperf.matrix.karma_sparse import KarmaSparse, DTYPE, ITYPE, LTYPE

cimport cython
from libc.stdlib cimport RAND_MAX, rand, srand, free, calloc
from libc.string cimport memcpy


cdef class SAFP:
    """
    Sparse Affinity Propagation Algorithm
        :param: matrix:         (np.array, KarmaSparse, scipy.sparse)
        :param: preference :    float (same preference for all samples),
                                np.array (prescribed preferences for each sample),
                                'mean' ( = preference=mean(matrix.data)),
                                'median' ( = preference=median(matrix.data)
                                None (matrix diagonal values will be used as preferences)

    This creates clusters by sending messages between pairs of nodes until convergence.
    The number of cluster is not determined by an input parameter but based on the data provided (matrix and preference)
    Each cluster created this way is described by an exemplars: the most representative of all nodes in the cluster.
    The messages sent between pairs are of two kinds:
        * the responsibility, evidence that a node k must be the exemplar for an other a
        * the availability, evidence that node a should choose node k as examplar
    """

    def __dealloc__(self):
        free(self.is_exemplar)
        free(self.is_old_exemplar)
        free(self.out_indices)
        free(self.row_max)
        free(self.row_second)
        free(self.temp_diag)
        free(self.tranpose_indptr)
        free(self.tranpose_indices)
        free(self.tranpose_data_index)
        free(self.similarity_transpose)

    def __cinit__(self, matrix, preference=None):
        assert matrix.shape[0] == matrix.shape[1], 'Similarity Matrix should be square'
        ksm = KarmaSparse(matrix, format="csc", copy=False)
        self.nrows = ksm.shape[0]

        if preference == "mean":
            preference = np.mean(np.asarray(ksm.data))
        elif preference == "median":
            preference = np.median(np.asarray(ksm.data))
        if isinstance(preference, (np.number, float, int, long)):
            preference = np.full(self.nrows, preference, dtype=DTYPE)

        if preference is not None:
            assert isinstance(preference, np.ndarray)
            if preference.shape[0] != self.nrows:
                raise ValueError("Wrong shape {} != {}".format(self.nrows,
                                                               preference.shape[0]))
            # to protect from "eliminate_zeros" effect
            data = preference - ksm.diagonal() + NOISE * np.random.randn(self.nrows)
        else:  # to activate diagonal connections
            data = NOISE * np.random.randn(self.nrows)

        ksm = ksm + KarmaSparse((data, np.arange(self.nrows), np.arange(self.nrows+1)),
                                       shape=ksm.shape, format="csc", copy=False)

        # getting the attributes
        self.nnz = ksm.nnz
        self.indices = ksm.indices
        self.indptr = ksm.indptr
        self.similarity = ksm.data
        self.no_change = 0
        self.availability = np.zeros(self.nnz, dtype=DTYPE)
        self.responsibility = np.zeros(self.nnz, dtype=DTYPE)

        # Allocate all temporally pointers
        self.is_exemplar = <bool *>calloc(self.nrows, sizeof(bool))
        self.is_old_exemplar = <bool *>calloc(self.nrows, sizeof(bool))

        self.out_indices =  <ITYPE_t *>calloc(self.nrows, sizeof(ITYPE_t))
        self.row_max = <DTYPE_t *>calloc(self.nrows, sizeof(DTYPE_t))
        self.row_second = <DTYPE_t *>calloc(self.nrows, sizeof(DTYPE_t))
        self.temp_diag = <DTYPE_t *>calloc(self.nrows, sizeof(DTYPE_t))

        self.tranpose_indptr = <LTYPE_t *>calloc(self.nrows + 1, sizeof(LTYPE_t))
        self.tranpose_indices = <ITYPE_t *>calloc(self.nnz, sizeof(ITYPE_t))
        self.tranpose_data_index = <LTYPE_t *>calloc(self.nnz, sizeof(LTYPE_t))
        self.similarity_transpose = <DTYPE_t *>calloc(self.nnz, sizeof(DTYPE_t))

        self.add_small_noise()
        self.find_diagonal_indices()
        self.fill_transpose_index()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void fill_transpose_index(self):
        """
        This method computes the mapping of indices and indptr based on data,
        to access data of the sparse backend in a row oriented way.
        """
        cdef LTYPE_t u, last
        cdef ITYPE_t col, ind

        # compute number of occurence per indice
        for u in xrange(self.nnz):
            self.tranpose_indptr[self.indices[u] + 1] += 1

        # cum_sum of number of occurence
        for col in xrange(self.nrows):
            self.tranpose_indptr[col+1] += self.tranpose_indptr[col]


        for col in xrange(self.nrows):
            for u in xrange(self.indptr[col], self.indptr[col+1]):
                ind = self.indices[u]
                self.tranpose_data_index[self.tranpose_indptr[ind]] = u
                self.tranpose_indices[self.tranpose_indptr[ind]] = col
                self.tranpose_indptr[ind] += 1

        # shift all transpose_indptr values, so that the first on is zero
        last = 0
        for col in xrange(self.nrows):
            u = self.tranpose_indptr[col]
            self.tranpose_indptr[col] = last
            last = u

        for col in prange(self.nrows, nogil=True):
            for u in xrange(self.tranpose_indptr[col], self.tranpose_indptr[col+1]):
                self.similarity_transpose[u] = self.similarity[self.tranpose_data_index[u]]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void add_small_noise(self, seed=None):
        cdef LTYPE_t u
        cdef DTYPE_t factor = NOISE / RAND_MAX
        np.random.seed(seed)
        srand(np.random.randint(0, RAND_MAX))
        np.random.seed(None)

        for u in prange(self.nnz, nogil=True, schedule="static"):
            self.similarity[u] *= 1 + factor * rand()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef update_preference(self, preference):
        cdef DTYPE_t[::1] pp
        cdef ITYPE_t u
        if preference is None:
            preference = np.mean(np.asarray(self.similarity))
            # preference = np.median(np.array(self.similarity))
        if isinstance(preference, np.ndarray):
            if preference.shape[0] != self.nrows:
                raise ValueError("Wrong shape {} != {}".format(self.nrows,
                                                               preference.shape[0]))
            pp = np.asarray(preference, dtype=DTYPE)
        else:
            pp = np.full(self.nrows, preference, dtype=DTYPE)

        for u in prange(self.nrows, nogil=True):
            if self.diag_indices[u] != -1:
                self.similarity[u] = pp[self.diag_indices[u]]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void find_diagonal_indices(self):
        """
            self.diag_indices[i] is the index of the i-th diagonal coefficient
            and -1 if there is none.
        """
        cdef LTYPE_t u
        cdef ITYPE_t col
        self.diag_indices = - np.ones(self.nrows, dtype=LTYPE)
        for col in prange(self.nrows, nogil=True, schedule="static"):
                for u in xrange(self.indptr[col], self.indptr[col + 1]):
                    if self.indices[u] == col:
                        self.diag_indices[col] = u  # because it is deduplicated
                        break
                    elif self.indices[u] > col:  # because it is sorted
                        break

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void diagonal_extractor(self, DTYPE_t[::1] array) nogil:
        cdef:
            ITYPE_t i
            LTYPE_t u
        for i in xrange(self.nrows):
            u = self.diag_indices[i]
            if u != -1:
                self.temp_diag[i] = array[u]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void responsibility_update(self, DTYPE_t damping):
        """
        Updates the responsibility matrices. Responsibility can be done independantly on each row.
        It is parallelized thanks to a row oriented mapping of indices of the sparse backend.
        """
        cdef:
            ITYPE_t col, row
            LTYPE_t u
            DTYPE_t a, b, d, newval, co_damping = 1 - damping

        for row in prange(self.nrows, nogil=True, schedule="static"):
            a = MINF
            b = MINF
            col = -1
            for u in xrange(self.tranpose_indptr[row], self.tranpose_indptr[row+1]):
                d = self.similarity_transpose[u] + self.availability[self.tranpose_data_index[u]]
                if d > a:
                    b = a
                    a = d
                    col = self.tranpose_indices[u]
                elif d > b:
                    b = d
            self.out_indices[row] = col
            self.row_max[row] = a
            self.row_second[row] = b

        for col in prange(self.nrows, nogil=True, schedule="static"):
            for u in xrange(self.indptr[col], self.indptr[col + 1]):
                row = self.indices[u]
                if col == self.out_indices[row]:
                    newval = self.similarity[u] - self.row_second[row]
                else:
                    newval = self.similarity[u] - self.row_max[row]
                self.responsibility[u] = damping * self.responsibility[u] + co_damping * newval

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void availability_update(self, DTYPE_t damping):
        """
        Updates the availability matrix. Availability is computed independantly
        on each column of matrices. This can be parallelized thanks to a sparse column oriented backend.
        """
        cdef:
            ITYPE_t col
            LTYPE_t u
            DTYPE_t mx, newval, co_damping = 1 - damping

        with nogil:
            self.diagonal_extractor(self.responsibility)

        for col in prange(self.nrows, nogil=True, schedule="static"):
            mx = 0
            for u in xrange(self.indptr[col], self.indptr[col + 1]):
                if self.responsibility[u] > 0 and self.indices[u] != col:
                    mx = mx + self.responsibility[u]

            for u in xrange(self.indptr[col], self.indptr[col + 1]):
                if self.indices[u] == col:
                    newval = mx
                else:
                    newval = min(mx + self.temp_diag[col] - max(self.responsibility[u], 0), 0)
                self.availability[u] = damping * self.availability[u] + co_damping * newval

    cpdef iterate(self, DTYPE_t damping=0.6):
        self.responsibility_update(damping)
        self.availability_update(damping)

    cpdef bool check_convergence(self, ITYPE_t examplars_stable_criteria=30):
        """
        Convergence is considered as obtained if:
            * the exemplars haven't change since a given number of iterations (examplars_stable_criteria)
        """
        memcpy(self.is_old_exemplar, self.is_exemplar, self.nrows * sizeof(bool))
        self.compute_examplars()
        if all_equal(self.is_old_exemplar, self.is_exemplar, self.nrows):
            self.no_change += 1
        else:
            self.no_change = 0

        if self.no_change >= examplars_stable_criteria:
            return 1
        return 0

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void compute_examplars(self):
        """
        Exemplars are choosen as the cells with the higher result for responsibility and availability:
        they have been choosen by others as examplar and they are available
        """
        cdef:
            ITYPE_t row
            LTYPE_t u
            DTYPE_t val

        for row in xrange(self.nrows):
            u = self.diag_indices[row]
            if u != -1:
                val = self.availability[u] + self.responsibility[u]
                if val > 0:
                    self.is_exemplar[row] = 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef finish(self):
        cdef:
            ITYPE_t row, col
            DTYPE_t mx, val

        self.compute_examplars()
        self.cluster_map = np.arange(self.nrows, dtype=ITYPE)
        for row in xrange(self.nrows):
            mx = MINF
            if self.is_exemplar[row]:
                pass
            else:
                for u in xrange(self.tranpose_indptr[row], self.tranpose_indptr[row+1]):
                    col = self.tranpose_indices[u]
                    if self.is_exemplar[col]:
                        val = self.similarity_transpose[u]
                        if val >= mx:
                            self.cluster_map[row] = col
                            mx = val

    def build(self, ITYPE_t min_iter=30, ITYPE_t max_iter=300,
              DTYPE_t damping=0.6, bool check_conv=False,
              ITYPE_t examplars_stable_criteria=30, bool verbose=False):
        cdef ITYPE_t i, nb = 0

        for i in xrange(max_iter):
            self.iterate(damping)

            if i >= min_iter and check_conv:
                if self.check_convergence(examplars_stable_criteria):
                    if verbose:
                        print("SAFP : Early convergence after {} iterations".format(i))
                    break
        self.finish()
        if verbose:
            for i in xrange(self.nrows):
                if self.is_exemplar[i]:
                    nb += 1
            print("SAFP : {} clusters have been obtained".format(nb))
        return np.asarray(self.cluster_map)

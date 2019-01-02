cimport numpy as np
import numpy as np

from cyperf.matrix.karma_sparse cimport KarmaSparse, DTYPE_t, ITYPE_t, LTYPE_t, BOOL_t, bool
from cyperf.matrix.routine cimport all_equal

cdef DTYPE_t MINF = - np.inf
cdef DTYPE_t NOISE = 0.0000001


cdef class SAFP:
    cdef:
        readonly ITYPE_t nrows
        readonly LTYPE_t nnz
        readonly ITYPE_t no_change
        readonly DTYPE_t[::1] availability
        readonly DTYPE_t[::1] responsibility
        readonly ITYPE_t[::1] cluster_map
        DTYPE_t[::1] similarity
        const ITYPE_t[::1] indices
        const LTYPE_t[::1] indptr
        LTYPE_t[::1] diag_indices

        # temporal pointers
        BOOL_t * is_exemplar
        BOOL_t * is_old_exemplar
        LTYPE_t * tranpose_indptr
        ITYPE_t * tranpose_indices
        LTYPE_t * tranpose_data_index
        DTYPE_t * similarity_transpose
        ITYPE_t * out_indices
        DTYPE_t * row_max
        DTYPE_t * row_second
        DTYPE_t * temp_diag

    cdef void fill_transpose_index(self)

    cdef void add_small_noise(self, seed=*)

    cpdef update_preference(self, preference)

    cdef void find_diagonal_indices(self)

    cdef void diagonal_extractor(self, const DTYPE_t[::1] array) nogil

    cdef void responsibility_update(self, DTYPE_t damping)

    cdef void availability_update(self, DTYPE_t damping)

    cpdef iterate(self, DTYPE_t damping=*)

    cpdef bool check_convergence(self, ITYPE_t examplars_stable_criteria=*)

    cdef void compute_examplars(self)

    cpdef finish(self)

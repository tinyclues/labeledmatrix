
cimport numpy as np
from cyperf.tools.types cimport string, bool, LTYPE_t, ITYPE_t, FTYPE_t as DTYPE_t
from heap cimport ActiveList, MinHeap
from space_tools cimport get_distance, _ward_pairwise_flat, metric_func_type
from libcpp.vector cimport vector
from libc.math cimport sqrt, fmax, fmin


ctypedef vector[ITYPE_t] IVEC

cdef struct MinVal:
    ITYPE_t ind
    DTYPE_t val

cdef class TailTree:
    cdef:
        DTYPE_t[:, ::1] X
        DTYPE_t[::1] weights
        readonly ITYPE_t n_clusters, n, dim
        readonly np.ndarray weights_copy
        readonly np.ndarray link
        ActiveList active_indices
        MinHeap heap
        metric_func_type dist_func

    cdef MinVal argmin(self, ITYPE_t i0) nogil

    cdef void merge(self, ITYPE_t i0, ITYPE_t j0) nogil

    cpdef np.ndarray[dtype=DTYPE_t, ndim=2] build(self)

cpdef IVEC traversal(const ITYPE_t[:,::1] link, ITYPE_t node)

cdef DTYPE_t simple_dist(const DTYPE_t[::1] matrix, IVEC rows, IVEC cols, ITYPE_t n)


cdef inline LTYPE_t pnt(LTYPE_t i, LTYPE_t n) nogil:
    return i * (n - 1) - i * (i + 1) // 2 - 1


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
    cdef:
        readonly np.ndarray X
        readonly ITYPE_t n_clusters, n
        DTYPE_t[::1] weights
        readonly np.ndarray weights_copy
        readonly DTYPE_t[::1] distances
        readonly np.ndarray link
        DTYPE_t[::1] min_dist
        ITYPE_t[::1] argmin_dist
        ActiveList active_indices
        IVEC chain

    cdef void correct(self, ITYPE_t row) nogil

    cdef void ward_update(self, ITYPE_t i0, ITYPE_t j0) nogil

    cdef inline ITYPE_t global_argmin(self) nogil

    cdef inline ITYPE_t nn_chain(self) nogil

    cdef void mindist_update(self, ITYPE_t i0, ITYPE_t j0) nogil

    cdef void merge(self, ITYPE_t i0, ITYPE_t j0) nogil

    cdef np.ndarray[dtype=DTYPE_t, ndim=2] build(self, bool chain=*)

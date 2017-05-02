cimport cython
cimport numpy as np
from cython.parallel cimport parallel, prange
from libc.math cimport pow as _cpow
from cyperf.tools.types cimport DTYPE_t, ITYPE_t, LTYPE_t, bool, string, A, B
from cyperf.tools.sort_tools cimport partial_sort, inplace_reordering, partial_unordered_sort
from routine cimport (logistic, get_reducer, binary_func, mult, axpy, scalar_product, computed_quantile, mmax)


# ctypedef (ITYPE_t, ITYPE_t) Shape_t  # BUG in cython : this type not cimportable
ctypedef tuple Shape_t

cdef Shape_t pair_swap(Shape_t x)
cpdef bool is_karmasparse(mat) except? 0
cdef bool is_shape(tuple shape) except? 0
cdef bool check_acceptable_format(string format) except 0
cdef bool check_nonzero_shape(shape) except 0
cdef bool check_bounds(ITYPE_t row, ITYPE_t upper_bound) except 0
cdef bool check_ordered(ITYPE_t row0, ITYPE_t row1, bool strict) except 0
cdef bool check_shape_comptibility(x1, x2) except 0


cdef inline double cpow(double x, double y) nogil:
    if y == 2:
        return x * x
    elif y == 3:
        return x * x * x
    elif y == 1:
        return x
    else:
        return _cpow(x, y)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _aligned_dense_vector_dot(ITYPE_t nrows, LTYPE_t* indptr, ITYPE_t* indices,
                                           DTYPE_t* data, DTYPE_t* vector, DTYPE_t* out):
    cdef:
        ITYPE_t i
        LTYPE_t j

    with nogil:
        for i in prange(nrows, schedule='guided'):
            for j in xrange(indptr[i], indptr[i + 1]):
                out[i] = out[i] + data[j] * vector[indices[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void kronii_dot(ITYPE_t nrows, LTYPE_t size, LTYPE_t* indptr, ITYPE_t* indices, DTYPE_t* data,
                            cython.floating* matrix, DTYPE_t* factor, DTYPE_t* result):
    cdef LTYPE_t i, j, k
    cdef ITYPE_t ind
    cdef DTYPE_t out, dd

    with nogil:
        for i in prange(nrows, schedule='guided'):
            out = 0
            for j in xrange(indptr[i], indptr[i + 1]):
                ind = indices[j]
                dd = data[j]
                for k in xrange(size):
                    out = out + dd * matrix[i * size + k] * factor[ind * size + k]
            result[i] = out


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _misaligned_dense_vector_dot(ITYPE_t nrows, LTYPE_t* indptr, ITYPE_t* indices,
                                              DTYPE_t* data, DTYPE_t* vector, DTYPE_t* out):
    cdef:
        ITYPE_t i
        LTYPE_t j
        DTYPE_t val

    with nogil:
        for i in xrange(nrows):
            val = vector[i]
            if val != 0:
                for j in xrange(indptr[i], indptr[i + 1]):
                    out[indices[j]] += val * data[j]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void inplace_arange(ITYPE_t * x, int size) nogil:
    for j in xrange(size): x[j] = j


cpdef np.ndarray[dtype=DTYPE_t, ndim=2] dense_pivot(ITYPE_t[::1] rows, ITYPE_t[::1] cols,
                                                    DTYPE_t[::1] values, shape=*,
                                                    string aggregator=*, DTYPE_t default=*)


cdef class KarmaSparse:
    cdef:
        readonly Shape_t shape
        readonly string format
        bool has_sorted_indices
        bool has_canonical_format
        ITYPE_t[::1] indices
        LTYPE_t[::1] indptr
        DTYPE_t[::1] data
        ITYPE_t nrows
        ITYPE_t ncols

    cdef bool aligned_axis(self, axis) except? 0

    cdef bool has_format(self, string my_format) except 0

    cdef inline string swap_format(self) nogil

    cdef bool from_flat_array(self, data, indices, indptr, tuple shape, string format=*, bool copy=*,
                              string aggregator=*) except 0

    cdef bool from_scipy_sparse(self, a, format=*, copy=*, string aggregator=*) except 0

    cdef bool from_zeros(self, tuple shape, format=*) except 0

    cdef bool from_dense(self, np.ndarray a, format=*) except 0

    cdef bool from_coo(self, data, ix, iy, shape=*, format=*, string aggregator=*) except 0

    cdef bool check_internal_structure(self, bool full=*) except 0

    cdef bool check_positive(self) except 0

    cdef bool prune(self) except 0

    cdef LTYPE_t get_nnz(self) nogil except -1

    cpdef bool eliminate_zeros(self, DTYPE_t value=*) except 0

    cdef bool keep_tril(self, ITYPE_t k=*) except 0

    cdef bool keep_triu(self, ITYPE_t k=*) except 0

    cpdef KarmaSparse tril(self, ITYPE_t k=*)

    cpdef KarmaSparse triu(self, ITYPE_t k=*)

    cdef bool _has_sorted_indices(self)

    cdef bool sort_indices(self) except 0

    cdef bool _has_canonical_format(self)

    cdef bool make_canonical(self, string aggregator=*) except 0

    cpdef KarmaSparse copy(self)

    cpdef np.ndarray[dtype=DTYPE_t, ndim=2] toarray(self)

    cpdef KarmaSparse element_sample(self, DTYPE_t proba, seed=*)

    cpdef tuple nonzero(self)

    cpdef np.ndarray[DTYPE_t, ndim=1] diagonal(self)

    cpdef KarmaSparse transpose(self, bool copy=*)

    cpdef KarmaSparse tocsr(self)

    cpdef KarmaSparse tocsc(self)

    cpdef KarmaSparse extend(self, Shape_t shape, bool copy=*)

    cdef KarmaSparse swap_slicing(self)

    cdef bool scale_rows(self, np.ndarray factor) except 0

    cdef bool scale_columns(self, np.ndarray factor) except 0

    cpdef KarmaSparse apply_pointwise_function(self, function, function_args=*)

    cdef KarmaSparse scalar_multiply(self, DTYPE_t factor)

    cdef KarmaSparse scalar_divide(self, DTYPE_t factor)

    cdef KarmaSparse scalar_add(self, DTYPE_t scalar)

    cpdef KarmaSparse nonzero_mask(self)

    cpdef KarmaSparse abs(self)

    cpdef KarmaSparse rint(self)

    cpdef KarmaSparse sign(self)

    cpdef KarmaSparse trunc(self)

    cpdef KarmaSparse sqrt(self)

    cpdef KarmaSparse power(self, DTYPE_t p)

    cpdef KarmaSparse clip(self, lower, upper=*)

    cpdef KarmaSparse log(self)

    cpdef KarmaSparse truncate_with_cutoff(self, DTYPE_t cutoff)

    cdef KarmaSparse aligned_truncate_by_count(self, np.ndarray[ITYPE_t, ndim=1] count)

    cdef KarmaSparse aligned_truncate_by_budget(self, density, DTYPE_t volume)

    cdef KarmaSparse aligned_truncate_cumulative(self, DTYPE_t percentage)

    cdef KarmaSparse global_truncate_cumulative(self, DTYPE_t percentage)

    cpdef KarmaSparse truncate_by_count(self, nb, axis)

    cpdef KarmaSparse truncate_by_budget(self, np.ndarray values, DTYPE_t budget, axis)

    cpdef KarmaSparse truncate_by_cumulative(self, DTYPE_t percentage, axis)

    cdef KarmaSparse csr_generic_dot_top(self, KarmaSparse other, ITYPE_t nb_keep, DTYPE_t cutoff, binary_func op)

    cdef KarmaSparse generic_dot_top(self, KarmaSparse other, ITYPE_t nb_keep, DTYPE_t cutoff, binary_func fn)

    cpdef KarmaSparse sparse_dot_top(self, KarmaSparse other, ITYPE_t nb_keep)

    cpdef KarmaSparse pairwise_min_top(self, KarmaSparse other, ITYPE_t nb_keep, DTYPE_t cutoff)

    cpdef KarmaSparse pairwise_max_top(self, KarmaSparse other, ITYPE_t nb_keep, DTYPE_t cutoff)

    cdef KarmaSparse aligned_sparse_dot(self, KarmaSparse other)

    cdef KarmaSparse sparse_dot(self, KarmaSparse other)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_axis_reduce(self, binary_func fn, bool only_nonzero)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_axis_reduce(self, binary_func fn, bool only_nonzero)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] reducer(self, string name, int axis, bool only_nonzero=*)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum_abs(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_max_abs(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_max_abs(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum_abs(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum_power(self, DTYPE_t p)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum_power(self, DTYPE_t p)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_count_nonzero(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_count_nonzero(self)

    cdef sum_abs(self, axis=*)

    cdef max_abs(self, axis=*)

    cdef KarmaSparse global_rank(self, bool reverse=*)

    cdef KarmaSparse aligned_rank(self, bool reverse=*)

    cpdef KarmaSparse rank(self, axis, bool reverse=*)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_quantile(self, DTYPE_t quantile, bool only_nonzero=*)

    cpdef KarmaSparse scale_along_axis(self, np.ndarray factor, axis)

    cpdef KarmaSparse scale_along_axis_inplace(self, np.ndarray factor, axis)

    cdef KarmaSparse aligned_compatibility_renormalization(self, ITYPE_t[::1] row_gender,
                                                           ITYPE_t[::1] column_gender,
                                                           DTYPE_t homo_factor,
                                                           DTYPE_t hetero_factor)

    cpdef KarmaSparse compatibility_renormalization(self, row_gender, column_gender,
                                                    DTYPE_t homo_factor,
                                                    DTYPE_t hetero_factor)

    cdef bool aligned_truncate_by_count_by_group(self, raw_group, ITYPE_t nb_keep) except 0

    cpdef KarmaSparse truncate_by_count_by_groups(self, group, ITYPE_t nb, axis=*)

    cdef tuple global_argminmax(self, bool reverse, bool only_nonzero=*)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] aligned_argminmax(self, bool reverse, bool only_nonzero=*)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] misaligned_argminmax(self, bool reverse, bool only_nonzero=*)

    cdef tuple global_argmin(self, bool only_nonzero=*)

    cdef tuple global_argmax(self, bool only_nonzero=*)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] axis_argmax(self, axis, bool only_nonzero=*)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] axis_argmin(self, axis, bool only_nonzero=*)

    cdef np.ndarray[DTYPE_t, ndim=2] aligned_dense_dot(self, np.ndarray matrix)

    cdef np.ndarray[DTYPE_t, ndim=2] misaligned_dense_dot(self, np.ndarray matrix)

    cdef np.ndarray[float, ndim=2] aligned_dense_agg(self, np.ndarray matrix, binary_func fn=*)

    cdef np.ndarray[float, ndim=2] misaligned_dense_agg(self, np.ndarray matrix, binary_func fn=*)

    cdef KarmaSparse csr_mask_dense_dense_dot(self, np.ndarray a, np.ndarray b,
                                              binary_func op)

    cdef KarmaSparse csr_mask_sparse_sparse_dot(self, KarmaSparse other_a, KarmaSparse other_b, binary_func op)

    cdef KarmaSparse csr_mask_sparse_dense_dot(self, KarmaSparse other, np.ndarray b, binary_func op)

    cdef inline KarmaSparse generic_restricted_binary_operation(self, KarmaSparse other, binary_func fn)

    cdef inline KarmaSparse generic_binary_operation(self, KarmaSparse other, binary_func fn)

    cpdef KarmaSparse complement(self, other)

    cpdef KarmaSparse maximum(self, KarmaSparse other)

    cpdef KarmaSparse minimum(self, KarmaSparse other)

    cdef KarmaSparse add(self, KarmaSparse other)

    cdef KarmaSparse multiply(self, KarmaSparse other)

    cdef KarmaSparse kronii_align_dense(self, cython.floating[:,:] other)

    cdef KarmaSparse kronii_align_sparse(self, KarmaSparse other)

    cdef KarmaSparse divide(self, KarmaSparse other)

    cdef DTYPE_t aligned_get_single_element(self, ITYPE_t row, ITYPE_t col) nogil

    cdef DTYPE_t get_single_element(self, ITYPE_t row, ITYPE_t col) except? -1

    cdef bool check_arrays(self, np.ndarray rows, np.ndarray cols) except 0

    cdef np.ndarray[DTYPE_t, ndim=1] sample_values(self, row_list, col_list)

    cdef KarmaSparse aligned_get_submatrix(self, ITYPE_t row0, ITYPE_t row1, ITYPE_t col0, ITYPE_t col1)

    cdef KarmaSparse aligned_subinterval(self, ITYPE_t row0, ITYPE_t row1)

    cdef KarmaSparse extractor(self, my_indices, axis)

    cdef KarmaSparse get_submatrix(self, ITYPE_t row0, ITYPE_t row1, ITYPE_t col0, ITYPE_t col1)

    cdef KarmaSparse get_row_slice(self, slice sl)

    cdef KarmaSparse get_column_slice(self, slice sl)

    cdef KarmaSparse get_row(self, ITYPE_t row)

    cdef KarmaSparse get_column(self, ITYPE_t col)

    cdef KarmaSparse restrict_along_row(self, key)

    cdef KarmaSparse restrict_along_column(self, key)

    cdef KarmaSparse aligned_sparse_agg(self, KarmaSparse other, binary_func fn=*)

    cdef np.ndarray[DTYPE_t, ndim=1] aligned_dense_vector_dot(self, DTYPE_t[::1] vector)

    cdef np.ndarray[DTYPE_t, ndim=1] misaligned_dense_vector_dot(self, DTYPE_t[::1] vector)

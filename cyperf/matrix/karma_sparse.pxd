#cython: embedsignature=True
#cython: nonecheck=True
#cython: overflowcheck=True
#cython: unraisable_tracebacks=True

cimport cython
cimport numpy as np
from cython.parallel cimport parallel, prange
from cython cimport floating, integral

from libc.stdlib cimport malloc, free, calloc, realloc
from libc.string cimport memcpy
from libc.math cimport pow as _cpow
from cyperf.tools.types cimport ITYPE_t, LTYPE_t, BOOL_t, bool, A, B, FTYPE_t
from cyperf.tools.sort_tools cimport partial_sort, inplace_reordering, partial_unordered_sort

ctypedef np.float32_t DTYPE_t

ctypedef tuple Shape_t
# ctypedef (ITYPE_t, ITYPE_t) Shape_t  # BUG in cython : this type not cimportable

cdef Shape_t pair_swap(Shape_t x)
cpdef bool is_karmasparse(mat) except? 0
cdef bool is_shape(tuple shape) except? 0
cdef bool check_acceptable_format(str format) except 0
cdef bool check_nonzero_shape(shape) except 0
cdef bool check_bounds(ITYPE_t row, ITYPE_t upper_bound) except 0
cdef bool check_ordered(ITYPE_t row0, ITYPE_t row1, bool strict) except 0
cdef bool check_shape_comptibility(x1, x2) except 0


# cy_syr fused type
from scipy.linalg.cython_blas cimport dsyr, ssyr
# dsyr(char *uplo, int *n, d *alpha, d *x, int *incx, d *a, int *lda)
ctypedef void (*cy_syr_type)(int size, floating alpha, floating * x, floating * target, int dim) nogil


cdef inline void cy_ssyr(int size, float alpha, float * x, float * target, int dim) nogil:
    cdef int one = 1
    cdef char * uplo = 'l'
    ssyr(uplo, &size, &alpha, x, &one, target, &dim)


cdef inline void cy_dsyr(int size, double alpha, double * x, double * target, int dim) nogil:
    cdef int one = 1
    cdef char * uplo = 'l'
    dsyr(uplo, &size, &alpha, x, &one, target, &dim)


cdef inline double cpow(double x, double y) nogil:
    if y == 2:
        return x * x
    elif y == 3:
        return x * x * x
    elif y == 1:
        return x
    else:
        return _cpow(x, y)


#from scipy.linalg.cython_blas cimport daxpy, saxpy
# Warning : type of blas routines should be compatible with DTYPE_t
# void daxpy(int *n, d *da, d *dx, int *incx, d *dy, int *incy)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void axpy(ITYPE_t n, DTYPE_t a, A * x, DTYPE_t * y) nogil:
    cdef int i
    for i in range(n):
        y[i] += a * x[i]
    # daxpy(<int*>&n, &a, <double*>x, &m, <double*>y, &m)
    # saxpy(<int*>&n, &a, <float*>x, &m, <float*>y, &m)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline DTYPE_t _scalar_product(ITYPE_t n, A* x, B* y) nogil:
    cdef ITYPE_t k
    cdef DTYPE_t res = 0
    for k in range(n):
        res += x[k] * y[k]
    return res


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline DTYPE_t _scalar_product_left_squared(ITYPE_t n, A* x, B* y) nogil:
    cdef ITYPE_t k
    cdef DTYPE_t res = 0
    for k in range(n):
        res += x[k] * x[k] * y[k]
    return res


from scipy.linalg.cython_blas cimport sdot
# d ddot(int *n, d *dx, int *incx, d *dy, int *incy)

cdef inline DTYPE_t _blas_scalar_product(int n, DTYPE_t* x, DTYPE_t* y) nogil:
    cdef int m = 1
    #return ddot(<int*>&n, <double*>x, &m, <double*>y, &m)
    return sdot(<int*>&n, <float*>x, &m, <float*>y, &m)


cdef inline DTYPE_t scalar_product(ITYPE_t n, DTYPE_t * x, DTYPE_t * y) nogil:
    if n <= 40:  # turns out to be faster for small n
        return _scalar_product(n, x, y)
    else:
        return _blas_scalar_product(n, x, y)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _linear_error_dense(floating[:,::1] inp, DTYPE_t[:,::1] matrix,
                                     DTYPE_t * tmp, DTYPE_t * out,
                                     DTYPE_t* column, DTYPE_t* row) nogil:
        cdef:
            ITYPE_t n_rows = inp.shape[0], n_inter = inp.shape[1], n_cols = matrix.shape[1]
            LTYPE_t i, k, j
            DTYPE_t mx, val, alpha

        for i in range(n_rows):
            mx = row[i]
            for j in range(n_inter):
                alpha = inp[i, j]
                if alpha != 0:
                    axpy(n_cols, alpha, &matrix[j, 0], tmp)
            for k in range(n_cols):
                val = tmp[k] + mx
                tmp[k] = column[k]
                out[k] += val * val


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _linear_error(LTYPE_t nrows, LTYPE_t n_cols,
                               LTYPE_t * indptr, ITYPE_t * indices, DTYPE_t * data,
                               DTYPE_t[:,::1] matrix, DTYPE_t * tmp, DTYPE_t * out,
                               DTYPE_t* column, DTYPE_t* row) nogil:
        cdef:
            LTYPE_t i, k, l, j
            DTYPE_t mx, val, alpha

        for i in range(nrows):
            mx = row[i]
            for j in range(indptr[i], indptr[i + 1]):
                axpy(n_cols, data[j], &matrix[indices[j], 0], tmp)
            for k in range(n_cols):
                val = tmp[k] + mx
                tmp[k] = column[k]
                out[k] += val * val


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void kronii_dot(ITYPE_t nrows, LTYPE_t size, LTYPE_t* indptr, ITYPE_t* indices, DTYPE_t* data,
                            floating* matrix, DTYPE_t* factor, DTYPE_t* result, double power):
    cdef LTYPE_t i, j, k
    cdef ITYPE_t ind
    cdef DTYPE_t out, dd

    with nogil:
        for i in prange(nrows, schedule='guided'):
            out = 0
            for j in range(indptr[i], indptr[i + 1]):
                ind = indices[j]
                dd = cpow(data[j], power)
                for k in range(size):
                    out = out + dd * cpow(matrix[i * size + k], power) * factor[ind * size + k]
            result[i] = out


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void kronii_dot_transpose(LTYPE_t start, LTYPE_t stop, LTYPE_t size,
                                      LTYPE_t* indptr, ITYPE_t* indices, DTYPE_t* data,
                                      floating* matrix, DTYPE_t* factor, DTYPE_t* result, double power) nogil:
    cdef LTYPE_t i, j, k
    cdef ITYPE_t ind
    cdef DTYPE_t dd, f

    for i in range(start, stop):
        f = factor[i]
        if f == 0:
            continue
        for j in range(indptr[i], indptr[i + 1]):
            ind = indices[j]
            dd = cpow(data[j], power)
            for k in range(size):
                result[ind * size + k] += dd * cpow(matrix[i * size + k], power) * f


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void kronii_sparse_dot_transpose(LTYPE_t start, LTYPE_t stop, ITYPE_t other_ncols,
                                             LTYPE_t* self_indptr, ITYPE_t* self_indices, DTYPE_t* self_data,
                                             LTYPE_t* other_indptr, ITYPE_t* other_indices, DTYPE_t* other_data,
                                             DTYPE_t* factor, DTYPE_t* result, double power) nogil:

    cdef LTYPE_t i, j, ind, begin, end, kk
    cdef DTYPE_t alpha, f

    for i in range(start, stop):
        begin, end = other_indptr[i], other_indptr[i + 1]
        f = factor[i]
        if f == 0:
            continue
        for j in range(self_indptr[i], self_indptr[i + 1]):
            alpha = cpow(self_data[j], power)
            ind = self_indices[j] * other_ncols
            for kk in range(begin, end):
                result[ind + other_indices[kk]] += alpha * cpow(other_data[kk], power) * f


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void kronii_sparse_dot(ITYPE_t nrows, ITYPE_t other_ncols,
                                   LTYPE_t* self_indptr, ITYPE_t* self_indices, DTYPE_t* self_data,
                                   LTYPE_t* other_indptr, ITYPE_t* other_indices, DTYPE_t* other_data,
                                   DTYPE_t* factor, DTYPE_t* result, double power):

    cdef LTYPE_t i, j, ind, start, stop, kk
    cdef DTYPE_t alpha, out

    for i in prange(nrows, nogil=True):
        start, stop = other_indptr[i], other_indptr[i + 1]
        out = 0
        for j in range(self_indptr[i], self_indptr[i + 1]):
            alpha = cpow(self_data[j], power)
            ind = self_indices[j] * other_ncols
            for kk in range(start, stop):
                out = out + alpha * cpow(other_data[kk], power) * factor[ind + other_indices[kk]]

        result[i] = out


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _misaligned_dense_vector_dot(ITYPE_t start, ITYPE_t end, LTYPE_t* indptr, ITYPE_t* indices,
                                              DTYPE_t* data, A* vector, DTYPE_t* out) nogil:
    cdef:
        ITYPE_t i
        LTYPE_t j
        A val

    for i in range(start, end):
        val = vector[i]
        if val != 0:
            for j in range(indptr[i], indptr[i + 1]):
                out[indices[j]] += val * data[j]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _aligned_dense_vector_dot(ITYPE_t start, ITYPE_t stop, LTYPE_t* indptr, ITYPE_t* indices,
                                           DTYPE_t* data, A* vector, DTYPE_t* out) nogil:
    cdef:
        ITYPE_t i
        LTYPE_t j

    for i in range(start, stop):
        for j in range(indptr[i], indptr[i + 1]):
            out[i] = out[i] + data[j] * vector[indices[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void mmax_loop(int n_cols, DTYPE_t* a,  A* b, DTYPE_t alpha) nogil:
    cdef int k
    for k in range(n_cols):
        a[k] = mmax(a[k], alpha * b[k])


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void inplace_arange(ITYPE_t * x, int size) nogil:
    for j in range(size): x[j] = j


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline DTYPE_t computed_quantile(DTYPE_t* data, DTYPE_t quantile, LTYPE_t size, LTYPE_t dim) nogil:
    cdef:
        LTYPE_t last_negative_indice, j, nb_zero = dim - size
        DTYPE_t res, previous
        LTYPE_t indice_quantile = <LTYPE_t>(dim * quantile)
        DTYPE_t* sorted_data = <DTYPE_t *>malloc(size  * sizeof(DTYPE_t))
        DTYPE_t* temp = <DTYPE_t *>malloc(size  * sizeof(DTYPE_t))

    memcpy(sorted_data, data, size * sizeof(DTYPE_t))
    partial_sort(sorted_data, temp, size, size, False)

    # find where is the first zero
    for j in range(size):
        if sorted_data[j] > 0:
            last_negative_indice = j - 1
            break
    else:
        last_negative_indice = size - 1

    if indice_quantile > last_negative_indice:
        if indice_quantile > last_negative_indice + nb_zero:
            res = sorted_data[indice_quantile - nb_zero]
        else:
            res = 0
    else:
        res = sorted_data[indice_quantile]

    if dim % 2 != 1:
        indice_quantile = indice_quantile - 1
        if indice_quantile > last_negative_indice:
            if indice_quantile > last_negative_indice + nb_zero:
                previous = sorted_data[indice_quantile - nb_zero]
            else:
                previous = 0
        else:
            previous = sorted_data[indice_quantile]
        res = (res + previous) / 2
    free(sorted_data)
    free(temp)
    return res

# define function pointers with each signature
ctypedef A (*A_binary_func)(A, A) nogil

ctypedef BOOL_t (*BOOL_t_binary_func)(BOOL_t, BOOL_t) nogil
ctypedef ITYPE_t (*ITYPE_t_binary_func)(ITYPE_t, ITYPE_t) nogil
ctypedef LTYPE_t (*LTYPE_t_binary_func)(LTYPE_t, LTYPE_t) nogil
ctypedef FTYPE_t (*FTYPE_t_binary_func)(FTYPE_t, FTYPE_t) nogil
ctypedef DTYPE_t (*DTYPE_t_binary_func)(DTYPE_t, DTYPE_t) nogil


# defining the function as A(*)(A, A) will allow the compiler to compile 5 add function w/ all the
# different signature automatically.

@cython.cdivision(True)
cdef inline A save_division(A x, A y) nogil:
    if y == 0:
        return 0
    else:
        return x / y

cdef inline A mult(A a, A b) nogil:
    return a * b

cdef inline A mmin(A a, A b) nogil:
    if a > b:
        return b
    else:
        return a

cdef inline A mmax(A a, A b) nogil:
    if a > b:
        return a
    else:
        return b

cdef inline A cadd(A a, A b) nogil:
    return a + b

cdef inline A cminus(A a, A b) nogil:
    return a - b

cdef inline A first(A a, A b) nogil:
    return a

cdef inline A last(A a, A b) nogil:
    return b

cdef inline A complement(A a, A b) nogil:
    if b == 0:
        return a
    else:
        return 0


cpdef np.ndarray[dtype=A, ndim=2] dense_pivot(integral[:] rows, integral[:] cols, A[:] values,
                                              shape=*, str aggregator=*, DTYPE_t default=*)


cdef class KarmaSparse:
    cdef:
        readonly Shape_t shape
        readonly str format
        bool has_unique_indices
        bool has_sorted_indices
        bool has_canonical_format
        ITYPE_t[::1] indices
        LTYPE_t[::1] indptr
        DTYPE_t[::1] data
        ITYPE_t nrows
        ITYPE_t ncols

    cdef void set_canonical_flag(self, bool has_canonical_format, bool has_sorted_indices, bool has_unique_indices) nogil

    cdef bool aligned_axis(self, axis) except? 0

    cdef str swap_format(self)

    cdef void from_flat_array(self, data, indices, indptr, tuple shape, str format=*, bool copy=*,
                              str aggregator=*) except *

    cdef void from_scipy_sparse(self, a, format=*, copy=*, str aggregator=*) except *

    cdef void from_zeros(self, tuple shape, format=*) except *

    cdef void from_dense(self, np.ndarray a, format=*) except *

    cdef void from_coo(self, data, ix, iy, shape=*, format=*, str aggregator=*) except *

    cdef void check_internal_structure(self, bool full=*) except *

    cpdef void check_positive(self) except *

    cdef void prune(self) except *

    cdef LTYPE_t get_nnz(self) nogil except -1

    cpdef void eliminate_zeros(self, DTYPE_t value=*) except *

    cdef void keep_tril(self, ITYPE_t k=*) except *

    cdef void keep_triu(self, ITYPE_t k=*) except *

    cpdef KarmaSparse tril(self, ITYPE_t k=*)

    cpdef KarmaSparse triu(self, ITYPE_t k=*)

    cdef bool _has_sorted_indices(self)

    cdef void sort_indices(self)

    cdef bool _has_canonical_format(self)

    cdef void make_canonical(self, str aggregator=*) except *

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

    cdef void scale_rows(self, np.ndarray factor) except *

    cdef void scale_columns(self, np.ndarray factor) except *

    cpdef KarmaSparse apply_pointwise_function(self, function, function_args=*, function_kwargs=*)

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

    cdef KarmaSparse csr_generic_dot_top(self, KarmaSparse other, ITYPE_t nb_keep, DTYPE_t cutoff, DTYPE_t_binary_func op)

    cdef KarmaSparse generic_dot_top(self, KarmaSparse other, ITYPE_t nb_keep, DTYPE_t cutoff, DTYPE_t_binary_func fn)

    cpdef KarmaSparse sparse_dot_top(self, KarmaSparse other, ITYPE_t nb_keep)

    cpdef KarmaSparse pairwise_min_top(self, KarmaSparse other, ITYPE_t nb_keep, DTYPE_t cutoff)

    cpdef KarmaSparse pairwise_max_top(self, KarmaSparse other, ITYPE_t nb_keep, DTYPE_t cutoff)

    cdef KarmaSparse aligned_sparse_dot(self, KarmaSparse other)

    cdef KarmaSparse sparse_dot(self, KarmaSparse other)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_axis_reduce(self, DTYPE_t_binary_func fn, bool only_nonzero)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_axis_reduce(self, DTYPE_t_binary_func fn, bool only_nonzero)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] reducer(self, str name, int axis, bool only_nonzero=*)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum_abs(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_max_abs(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_max_abs(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum_abs(self)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum_power(self, DTYPE_t p)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum_power(self, DTYPE_t p)

    cdef np.ndarray[dtype=LTYPE_t, ndim=1] aligned_count_nonzero(self)

    cdef np.ndarray[dtype=LTYPE_t, ndim=1] misaligned_count_nonzero(self)

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

    cdef void aligned_truncate_by_count_by_group(self, raw_group, ITYPE_t nb_keep) except *

    cpdef KarmaSparse truncate_by_count_by_groups(self, group, ITYPE_t nb, axis=*)

    cdef tuple global_argminmax(self, bool reverse, bool only_nonzero=*)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] aligned_argminmax(self, bool reverse, bool only_nonzero=*)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] misaligned_argminmax(self, bool reverse, bool only_nonzero=*)

    cdef tuple global_argmin(self, bool only_nonzero=*)

    cdef tuple global_argmax(self, bool only_nonzero=*)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] axis_argmax(self, axis, bool only_nonzero=*)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] axis_argmin(self, axis, bool only_nonzero=*)

    cdef np.ndarray[DTYPE_t, ndim=2] aligned_dense_dot(self, A[:,::1] matrix)

    cdef np.ndarray[DTYPE_t, ndim=2] misaligned_dense_dot(self, A[:,::1] matrix)

    cdef np.ndarray[DTYPE_t, ndim=2] aligned_dense_shadow(self, A[:,::1] matrix)

    cdef np.ndarray[DTYPE_t, ndim=2] misaligned_dense_shadow(self, A[:,::1] matrix)

    cdef np.ndarray[DTYPE_t, ndim=1] aligned_dense_shadow_vector_dot(self, A[:,::1] matrix, B[::1] vector)

    cdef np.ndarray[DTYPE_t, ndim=1] aligned_dense_squared_shadow_vector_dot(self, A[:,::1] matrix, B[::1] vector)

    cdef KarmaSparse aligned_sparse_shadow(self, KarmaSparse other)

    cdef np.ndarray[DTYPE_t, ndim=1] aligned_dense_squared_dot_vector_dot(self, A[:,::1] matrix, B[::1] vector)

    cdef KarmaSparse csr_mask_dense_dense_dot(self, np.ndarray a, np.ndarray b, DTYPE_t_binary_func op)

    cdef KarmaSparse csr_mask_sparse_sparse_dot(self, KarmaSparse other_a, KarmaSparse other_b, DTYPE_t_binary_func op)

    cdef KarmaSparse csr_mask_sparse_dense_dot(self, KarmaSparse other, np.ndarray b, DTYPE_t_binary_func op)

    cdef inline KarmaSparse generic_restricted_binary_operation(self, KarmaSparse other, DTYPE_t_binary_func fn)

    cdef inline KarmaSparse generic_binary_operation(self, KarmaSparse other, DTYPE_t_binary_func fn)

    cpdef KarmaSparse complement(self, other)

    cpdef KarmaSparse maximum(self, KarmaSparse other)

    cpdef KarmaSparse minimum(self, KarmaSparse other)

    cdef KarmaSparse multiply(self, other)

    cdef KarmaSparse divide(self, other)

    cdef KarmaSparse kronii_align_dense(self, floating[:,:] other)

    cdef KarmaSparse kronii_align_sparse(self, KarmaSparse other)

    cdef DTYPE_t aligned_get_single_element(self, ITYPE_t row, ITYPE_t col) nogil

    cdef DTYPE_t get_single_element(self, ITYPE_t row, ITYPE_t col) except? -1

    cdef void check_arrays(self, np.ndarray rows, np.ndarray cols) except *

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

    cdef np.ndarray[DTYPE_t, ndim=1] aligned_dense_vector_dot(self, A[::1] vector)

    cdef np.ndarray[DTYPE_t, ndim=1] misaligned_dense_vector_dot(self, A[::1] vector)

    cdef KarmaSparse generic_dense_restricted_binary_operation(self, floating[:,:] other, DTYPE_t_binary_func fn)

    cdef np.ndarray[DTYPE_t, ndim=2] generic_dense_binary_operation(self, DTYPE_t[:,:] other, DTYPE_t_binary_func fn)

    cdef np.ndarray[DTYPE_t, ndim=2] aligned_quantile_boundaries(self, long nb)

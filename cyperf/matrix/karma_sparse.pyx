#cython: embedsignature=True
#cython: nonecheck=True
#cython: overflowcheck=True
#cython: unraisable_tracebacks=True

cimport cython
import scipy.sparse as sp
import numpy as np
from cython.parallel import parallel, prange
from cyperf.tools.types import ITYPE, DTYPE, LTYPE
from cyperf.tools import logit

from cython.parallel cimport threadid
from openmp cimport omp_get_num_threads
from libc.string cimport memset, memcpy
from libc.stdlib cimport malloc, free, calloc, realloc
from libc.stdlib cimport RAND_MAX, rand, srand
from libc.math cimport fabs

cdef string CSR = 'csr'
cdef string CSC = 'csc'
cdef string DEFAULT_AGG = 'add'


@cython.wraparound(False)
@cython.boundscheck(False)
def linear_error_dense(floating[:,::1] inp, DTYPE_t[:,::1] matrix,
                       DTYPE_t[::1] column, DTYPE_t[::1] row):
    check_shape_comptibility(inp.shape[1], matrix.shape[0])
    check_shape_comptibility(matrix.shape[1], column.shape[0])
    check_shape_comptibility(inp.shape[0], row.shape[0])

    cdef:
        ITYPE_t n_cols = matrix.shape[1]
        DTYPE_t[::1] out = np.zeros(n_cols, dtype=DTYPE)
        DTYPE_t[::1] tmp = column.copy()

    with nogil:
        _linear_error_dense(inp, matrix, &tmp[0], &out[0], &column[0], &row[0])

    return np.asarray(out)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[ndim=1, dtype=DTYPE_t] cython_power(DTYPE_t[::1] a, const double p):
    cdef:
        LTYPE_t i, nb = a.shape[0]
        DTYPE_t[::1] result = np.zeros(nb, dtype=DTYPE)

    with nogil:
        for i in xrange(nb):
            result[i] = cpow(a[i], p)
    return np.asarray(result)


cdef Shape_t pair_swap(Shape_t x):
    return (x[1], x[0])


cpdef bool is_karmasparse(mat) except? 0:
    return isinstance(mat, KarmaSparse)

# TODO : those methods can be rewritten in cython without calling scipy
# scipy.sparse implementation is quite slow for those routines
cpdef KarmaSparse ks_hstack(list_ks):
    return KarmaSparse(sp.hstack([KarmaSparse(x, copy=False)
                                  .to_scipy_sparse(copy=False) for x in list_ks], format='csr'), copy=False)

cpdef KarmaSparse ks_vstack(list_ks):
    return KarmaSparse(sp.vstack([KarmaSparse(x, copy=False, format="csr").to_scipy_sparse(copy=False)
                                  for x in list_ks], format='csr'), copy=False)


cpdef KarmaSparse ks_kron(mat1, mat2, format=None):
    mat1 = KarmaSparse(mat1, copy=False)
    mat2 = KarmaSparse(mat2, copy=False)
    return KarmaSparse(sp.kron(mat1.to_scipy_sparse(copy=False),
                               mat2.to_scipy_sparse(copy=False)), format=format)

cpdef KarmaSparse ks_diag(np.ndarray data, format="csr"):
    shape = data.shape[0]
    return KarmaSparse((data.copy(), np.arange(shape, dtype=ITYPE),
                        np.arange(shape + 1, dtype=LTYPE)),
                       shape=(shape, shape), format=format, copy=False)


cpdef KarmaSparse new_karmasparse(arg, shape, format):
    """
    For pickling protocol 2
    """
    return KarmaSparse(arg, shape=shape, format=format)


cpdef bool is_int(n) except? 0:
    """
    >>> is_int(np.array([1]))
    False
    >>> is_int(1.1)
    False
    >>> is_int(1)
    True
    >>> is_int((1,))
    False
    >>> is_int(np.array([1], dtype=np.int32)[0])
    True
    >>> is_int(np.array([np.nan]))
    False
    """
    if isinstance(n, (int, long, np.int16, np.int32, np.int64)):
        return 1
    try:
        return (<int?>n == n) is 1
    except (TypeError, ValueError):
        return 0


cdef bool is_shape(tuple shape) except? 0:
    if len(shape) != 2:
        return 0
    else:
        n, m = shape
        if is_int(n) and is_int(m):
            return 1
        else:
            return 0


cdef bool check_acceptable_format(string format) except 0:
    if not (format == CSR or format == CSC):
        raise ValueError('Format should be one of [{}, {}], got {}'
                         .format(CSR, CSC, format))
    return 1


cdef bool check_nonzero_shape(shape) except 0:
    if shape is None:
        raise TypeError('Shape should be provided, got {}'.format(shape))
    if not isinstance(shape, (tuple, list)):
        raise TypeError('Shape should be a sequence not {}'.format(type(shape)))
    if not is_shape(shape):
        raise ValueError('Wrong shape parameter, got {}'.format(shape))
    elif min(shape) < 0:
        raise ValueError('Shape values should be > 0, got {}'.format(shape))
    else:
        return 1


cdef bool check_bounds(ITYPE_t row, ITYPE_t upper_bound) except 0:
    if not (0 <= row < upper_bound):
        raise IndexError("index out of bounds, 0 <= {} < {}".format(row, upper_bound))
    else:
        return 1


cdef bool check_ordered(ITYPE_t row0, ITYPE_t row1, bool strict) except 0:
    if (row0 > row1):
        raise IndexError("wrong index, {} <= {}".format(row0, row1))
    if strict and row0 == row1:
        raise IndexError("wrong index, {} < {}".format(row0, row1))
    return 1


cdef bool check_shape_comptibility(x1, x2) except 0:
    if x1 != x2:
        raise ValueError('Incompatible shape, {} != {}'.format(x1, x2))
    else:
        return 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[dtype=floating, ndim=2] dense_pivot(ITYPE_t[::1] rows,
                                                            ITYPE_t[::1] cols,
                                                            floating[::1] values,
                                                            shape=None,
                                                            string aggregator=DEFAULT_AGG,
                                                            DTYPE_t default=0):
    if not (rows.shape[0] == cols.shape[0] == values.shape[0]):
        raise ValueError("Incompatible size of coordinates and/or data : {}, ({}, {})"
                         .format(values.shape[0], rows.shape[0], cols.shape[0]))

    cdef LTYPE_t n, nnz = rows.shape[0]
    cdef ITYPE_t x, y, maxx = 0, maxy = 0

    for n in xrange(nnz):
        x = rows[n]
        y = cols[n]
        if x < 0 or y < 0:
            raise ValueError("Coordinates cannot be negative")
        if x > maxx: maxx = x
        if y > maxy: maxy = y

    if shape is not None:
        check_nonzero_shape(shape)
        if maxx >= shape[0] or maxy >= shape[1]:
            raise ValueError("Coordinates are too large for provided shape")
    else:
        if len(rows) == 0:
            raise ValueError('Cannot derive shape value from empty data')
        shape = (maxx + 1, maxy + 1)

    cdef binary_func reducer = get_reducer(aggregator)
    cdef floating[:,::1] result = np.full(shape, default, dtype=np.array(values).dtype)
    cdef np.uint8_t[:,::1] mask = np.zeros(shape, dtype=np.uint8)
    cdef DTYPE_t val

    with nogil:
        for n in xrange(nnz):
            val = values[n]
            if val == val:  # val is not np.nan
                x, y = rows[n], cols[n]
                if mask[x, y]:
                    result[x, y] = reducer(result[x, y], val)
                else:
                    result[x, y] = val
                    mask[x, y] = 1

    return np.asarray(result)


@cython.wraparound(False)
@cython.boundscheck(False)
def truncate_by_count_axis1_sparse(A[:,::1] matrix, cython.integral[::1] max_counts):
    cdef long nb_row = matrix.shape[0], length = matrix.shape[1]
    assert nb_row == max_counts.shape[0]

    cdef LTYPE_t[::1] indptr = np.zeros(nb_row + 1, dtype=LTYPE)   # CSR output
    cdef long row, rank, j, pos
    cdef ITYPE_t * rank_matrix

    with nogil:
        for row in xrange(nb_row):
            indptr[row + 1] = indptr[row] + max(min(max_counts[row], length), 0)

    cdef ITYPE_t[::1] indices = np.zeros(indptr[nb_row], dtype=ITYPE)
    cdef DTYPE_t[::1] data = np.zeros(indptr[nb_row], dtype=DTYPE)

    with nogil, parallel():
        rank_matrix = <ITYPE_t *>malloc(length * sizeof(ITYPE_t))
        if rank_matrix == NULL:
            with gil: raise MemoryError()

        for row in prange(nb_row):
            inplace_arange(rank_matrix, length)
            rank = indptr[row + 1] - indptr[row]
            partial_unordered_sort(&matrix[row, 0], rank_matrix, length, rank)

            for j in xrange(rank):
                pos = j + indptr[row]
                indices[pos] = rank_matrix[j]
                data[pos] = matrix[row, j]

            inplace_reordering(&matrix[row, 0], rank_matrix, length)

        free(rank_matrix)

    return KarmaSparse((data, indices, indptr), shape=(nb_row, length),
                       format="csr", copy=False)


@cython.wraparound(False)
@cython.boundscheck(False)
def truncate_by_count_axis1_dense(A[:,::1] matrix, cython.integral[::1] max_counts):
    cdef long nb_row = matrix.shape[0], length = matrix.shape[1]
    assert nb_row == max_counts.shape[0]
    cdef A[:,::1] result = np.zeros((nb_row, length), dtype=np.asarray(matrix).dtype)
    cdef long row, rank, j
    cdef ITYPE_t * rank_matrix

    with nogil, parallel():
        rank_matrix = <ITYPE_t *>malloc(length * sizeof(ITYPE_t))
        if rank_matrix == NULL:
            with gil: raise MemoryError()

        for row in prange(nb_row):
            inplace_arange(rank_matrix, length)
            rank = max(min(max_counts[row], length), 0)
            partial_unordered_sort(&matrix[row, 0], rank_matrix, length, rank)

            for j in xrange(rank):
                result[row, rank_matrix[j]] = matrix[row, j]

            inplace_reordering(&matrix[row, 0], rank_matrix, length)

        free(rank_matrix)
    return np.asarray(result)


cdef class KarmaSparse:
    """
    KarmaSparse class implements sparse matrices stored in "CSR" and "CSC" format
    and operations on them.

    __init__ parameters:
    * arg :
        - (data, indices, indptr) - for flat compress form; format should be provided, "csr" or "crc"
        - (data, (x, y)) - for Coordinate-format COO
        - (x, y) with x,y np.ndarray corresponds to COO format with data = 1
        - (a, b) - tuple of integers, this will create a Zeros matrix of the shape=(a, b)
        - KarmaSparse
        - numpy.ndarray of dimension 2
        - scipy.sparse matrix
    * shape - tuple of integers to precise the matrix shape
    * format - either "csr" or "csc"
    * copy - boolean that forces the copy of entry data
    * has_sorted_indices - boolean (need for internal routine to avoid extra-check)
    * has_canonical_format - boolean (need for internal routine to avoid extra-check)
    * aggregator - string, indicates the name of aggregate to use to reduce duplicate values

    Internal data structure::

     * ``shape``: Matrix shape (#line, #col)
     * ``format``: Either CSR or CSC
     * ``indices``: Element positions in the matrix
     * ``indptr``: Element column (if CSC) or row (if CSR)
     * ``data``: Element data
     * ``has_sorted_indices``: Are indices sorted (in an increasing order)
     * ``has_canonical_format``: has_sorted_indices and there're no duplicate

    A valid KarmaSparse always assumed both attributes
    ``has_sorted_indices`` and ``has_canonical_format`` being ``True``.
    If entry does not have this property, KarmaSparse will automatically convert to
    canonical format (sorted and no-duplicated indices).

    Data structure example::

    Take a matrix

        M = | 0 3 4 |
            | 1 0 3 |

    CSR representation in KarmaSparse will be::

     * ``indptr = [0, 2, 4]``
     * ``indices = [1, 2, 0, 2]``
     * ``data = [3, 4, 1, 3]``

    CSC representation in KarmaSparse will be::

     * ``indptr = [0, 1, 3, 5]``
     * ``indices = [1, 0, 0, 1]``
     * ``data = [1, 3, 4, 3]``

    """
    property indices:
        def __get__(self):
            return np.asarray(self.indices)

    property indptr:
        def __get__(self):
            return np.asarray(self.indptr)

    property data:
        def __get__(self):
            return np.asarray(self.data)

    property dtype:
        def __get__(self):
            return np.asarray(self.data).dtype

    property storage_dtype:
        def __get__(self):
            return np.asarray(self.indices).dtype

    property nnz:
        def __get__(self):
            return self.get_nnz()

    property ndim:
        def __get__(self):
            return 2

    property T:
        def __get__(self):
            return self.transpose(copy=False)

    property density:
        def __get__(self):
            if self.shape[0] > 0 and self.shape[1] > 0:
                return 1. * self.nnz / self.shape[0] / self.shape[1]
            else:
                return 0.

    def __cinit__(self, arg, shape=None, format=None, bool copy=True,
                  bool has_sorted_indices=False,
                  bool has_canonical_format=False,
                  string aggregator=DEFAULT_AGG):
        if format is not None :
            check_acceptable_format(format)
        if shape is not None:
            check_nonzero_shape(shape)

        # those settings are useful only for from_flat_array input
        self.has_sorted_indices = has_sorted_indices
        self.has_canonical_format = has_canonical_format

        if sp.issparse(arg):  # ScipySparse matrix
            self.from_scipy_sparse(arg, format, copy=copy, aggregator=aggregator)
        elif isinstance(arg, np.ndarray):
            self.from_dense(arg, format)
        elif is_karmasparse(arg): # KarmaSparse matrix
            if format is not None and arg.format != format:
                arg = (<KarmaSparse?>arg).swap_slicing()
                copy = False
            self.has_sorted_indices = True
            self.has_canonical_format = True
            self.from_flat_array(arg.data, arg.indices, arg.indptr, arg.shape, arg.format,
                                 copy=copy, aggregator=aggregator)
        elif isinstance(arg, tuple):
            if is_shape(arg):  # zeros matrix of the given shape
                self.from_zeros(arg, format)
            elif len(arg) == 2 and isinstance(arg[1], tuple) and len(arg[1]) == 2:
                # data, (x,y) - COO format
                data, (xx, yy) = arg
                self.from_coo(data, xx, yy, shape=shape, format=format, aggregator=aggregator)
            elif len(arg) == 2:  # (x,y) - mask format that provides coordinates of the elements equal to 1
                xx, yy = arg
                data = np.ones(len(xx), dtype=DTYPE)
                self.from_coo(data, xx, yy, shape=shape, format=format, aggregator=aggregator)
            elif len(arg) == 3:  # flat compressed format (CSR or CSC)
                data, indices, indptr = arg
                if format is None:
                    raise ValueError('format should be specified')
                self.from_flat_array(data, indices, indptr, shape, format, copy=copy, aggregator=aggregator)
            else:
                raise ValueError('Cannot cast to KarmaSparse')
        else:
            try:
                self.from_dense(np.asarray(arg))
            except:
                raise TypeError('Cannot cast to KarmaSparse')

    def __repr__(self):
        repr_ = "<KarmaSparse matrix with properties :"
        repr_ += "\n * shape = {}".format(self.shape)
        repr_ += "\n * format '{}'".format(self.format)
        repr_ += "\n * data type {}".format(self.dtype)
        repr_ += "\n * storage type {}".format(self.storage_dtype)
        repr_ += "\n * number of non-zeros elements {}".format(self.nnz)
        repr_ += "\n * density of non-zeros elements {}".format(round(self.density,6))
        return repr_ + ">"

    def __str__(self):
        cdef LTYPE_t i
        cdef LTYPE_t max_len = 2
        cdef list row = [], col = [], data = []

        for i in xrange(min(max_len, self.nnz)):
            row.append(np.searchsorted(self.indptr, i, side='right') - 1)
            col.append(self.indices[i])
            data.append(self.data[i])

        for i in xrange(max(self.nnz - i - 1, max_len), self.nnz):
            row.append(np.searchsorted(self.indptr, i, side='right') - 1)
            col.append(self.indices[i])
            data.append(self.data[i])
        if self.format == CSC:
            row, col = col, row
        return ', '.join([('%s : %s' % t) for t in zip(list(zip(row, col)), data)])

    def __reduce__(self):
        return new_karmasparse, ((np.asarray(self.data), np.asarray(self.indices),
                                 np.asarray(self.indptr)), self.shape, self.format)

    def __len__(self):
        return self.shape[0]

    cdef bool aligned_axis(self, axis) except? 0:
        if not is_int(axis):
            raise TypeError("Axis should be of integer type, got {}".format(type(axis)))
        if axis not in [0,1]:
            raise ValueError("Axis value must be in [0,1], got {}".format(axis))
        return (axis == 1 and self.format == CSR) or (axis == 0 and self.format == CSC)

    cdef bool has_format(self, string my_format) except 0:
        if self.format != my_format:
            raise ValueError('This method accepts only format {}'.format(my_format))
        else:
            return 1

    cdef inline string swap_format(self) nogil:
        return CSC if self.format == CSR else CSR

    cdef bool from_flat_array(self, data, indices, indptr, tuple shape,
                              string format=CSR, bool copy=True, string aggregator=DEFAULT_AGG) except 0:
        check_nonzero_shape(shape)
        self.shape = shape
        check_acceptable_format(format)
        self.format = format
        if self.format == CSR:
            self.nrows, self.ncols = shape
        elif self.format == CSC:
            self.nrows, self.ncols = pair_swap(shape)

        self.indices = np.array(indices, dtype=ITYPE, order="C", copy=copy)
        self.indptr = np.array(indptr, dtype=LTYPE, order="C", copy=copy)
        self.data = np.array(data, dtype=DTYPE, order="C", copy=copy)

        self.check_internal_structure()
        self.make_canonical(aggregator=aggregator)
        return 1

    cdef bool from_scipy_sparse(self, a, format=None, copy=True, string aggregator=DEFAULT_AGG) except 0:
        assert sp.issparse(a), "Argument should be scipy.sparse matrix"
        check_nonzero_shape(a.shape)
        if format is None:
            if a.format in ['csr', 'csc']:
                format = a.format
            else:
                format = CSR
        if format != a.format:
            a = getattr(sp, format + "_matrix")(a)
        self.has_sorted_indices = a.has_sorted_indices
        self.has_canonical_format = a.has_canonical_format
        self.from_flat_array(a.data, a.indices, a.indptr, a.shape, a.format, copy, aggregator)
        return 1

    cdef bool from_zeros(self, tuple shape, format=None) except 0:
        check_nonzero_shape(shape)
        if format is None:
            format = CSR
        else:
            check_acceptable_format(format)
        length = shape[0] + 1 if <string>format == CSR else shape[1] + 1

        data = np.zeros(0, dtype=DTYPE)
        indices = np.zeros(0, dtype=ITYPE)
        indptr = np.zeros(length, dtype=LTYPE)
        self.from_flat_array(data, indices, indptr, shape, format, copy=False)
        return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool from_dense(self, np.ndarray a, format=None) except 0:
        if a.ndim <= 1:
            a = np.atleast_2d(a)  # embedding of 1-dim vector
        shape = tuple([a.shape[l] for l in xrange(a.ndim)])  # convert *int to tuple
        check_nonzero_shape(shape)
        if format is not None:
            check_acceptable_format(format)
        self.shape = shape
        if (format is not None and <string?>format == CSC) or \
           (format is None and np.isfortran(a)):
            a = a.T
            self.format = CSC
            self.nrows, self.ncols = pair_swap(self.shape)
        else:
            self.format = CSR
            self.nrows, self.ncols = self.shape
        cdef:
            DTYPE_t[:,:] aa = np.asarray(a, dtype=DTYPE)
            ITYPE_t i, j
            LTYPE_t n, nnz_max = (<LTYPE_t>self.nrows) * self.ncols

        self.indptr = np.zeros(self.nrows + 1, dtype=LTYPE, order="C")
        self.indices = np.zeros(nnz_max, dtype=ITYPE, order="C")
        self.data = np.zeros(nnz_max, dtype=DTYPE, order="C")

        with nogil:
            n = 0
            for i in xrange(self.nrows):
                for j in xrange(self.ncols):
                    if aa[i, j] != 0:
                        self.data[n] = aa[i, j]
                        self.indices[n] = j
                        n += 1
                self.indptr[i + 1] = n
        self.prune()
        self.check_internal_structure()
        self.has_sorted_indices = True
        self.has_canonical_format = True
        return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool from_coo(self, data, ix, iy,
                       shape=None, format=None, string aggregator=DEFAULT_AGG) except 0:
        if not (len(ix) == len(iy) == len(data)):
            raise ValueError("Incompatible size of coordinates and/or data : {}, ({}, {})"
                             .format(len(data), len(ix), len(iy)))
        if format is None:
            self.format = CSR
        else:
            check_acceptable_format(format)
            self.format = format

        cdef:
            ITYPE_t[::1] xx = np.asarray(ix, dtype=ITYPE, order="C")
            ITYPE_t[::1] yy = np.asarray(iy, dtype=ITYPE, order="C")
            LTYPE_t dest, last, temp, n, nnz = len(xx)
            ITYPE_t row
            ITYPE_t x, y, maxx = 0, maxy = 0

        for n in xrange(nnz):
            x = xx[n]
            y = yy[n]
            if x < 0 or y < 0:
                raise ValueError("Coordinates cannot be negative")
            if x > maxx: maxx = x
            if y > maxy: maxy = y

        if shape is not None:
            check_nonzero_shape(shape)
            if maxx >= shape[0] or maxy >= shape[1]:
                raise ValueError("Coordinates are too large for provided shape")
            self.shape = shape
        else:
            if len(ix) == 0:
                raise ValueError('Cannot derive shape value from empty data')
            self.shape = (maxx + 1, maxy + 1)
        self.nrows, self.ncols = self.shape if self.format == CSR else pair_swap(self.shape)

        if self.format == CSC:
            xx, yy = yy, xx

        cdef DTYPE_t[::1] dd = np.asarray(data, dtype=DTYPE, order="C")
        cdef DTYPE_t val

        self.indptr = np.zeros(self.nrows + 1, dtype=LTYPE, order="C")
        with nogil:
            for n in xrange(nnz):
                val = dd[n]
                if val == val:  # val is not np.nan
                    self.indptr[xx[n] + 1] += 1

            for row in xrange(self.nrows):
                self.indptr[row + 1] += self.indptr[row]

        self.indices = np.zeros(self.indptr[self.nrows], dtype=ITYPE, order="C")
        self.data = np.zeros(self.indptr[self.nrows], dtype=DTYPE, order="C")

        with nogil:
            for n in xrange(nnz):
                val = dd[n]
                if val == val:  # val is not np.nan
                    row  = xx[n]
                    dest = self.indptr[row]
                    self.indices[dest] = yy[n]
                    self.data[dest] = val
                    self.indptr[row] += 1

            last = 0
            for n in xrange(self.nrows + 1):
                temp = self.indptr[n]
                self.indptr[n] = last
                last = temp

        self.check_internal_structure()
        self.make_canonical(aggregator=aggregator)
        return 1

    cdef bool check_internal_structure(self, bool full=False) except 0:
        cdef Shape_t shape

        check_acceptable_format(self.format)
        check_nonzero_shape(self.shape)
        if self.format == CSR:
            shape = (self.nrows, self.ncols)
        else:
            shape = (self.ncols, self.nrows)
        if self.shape != shape:
            raise ValueError('Internal structure is not the one expected, got {}'.
                            format((self.nrows, self.ncols)))
        if self.nrows + 1 != self.indptr.shape[0]:
            raise ValueError('Wrong indptr shape: should be {}, got {}'.
                            format(self.nrows + 1, self.indptr.shape[0]))
        if self.indptr[0] != 0:
            raise ValueError('First element of indptr should be == 0')
        if self.indptr[self.nrows] < 0:
            raise ValueError('Last element of indptr should be >= 0, is {}'.
                            format(self.indptr[self.nrows]))
        if self.data.shape[0] != self.indices.shape[0]:
            raise ValueError('data and indices shape should be equal, {} != {}'.
                format(self.data.shape[0], self.indices.shape[0]))
        if self.indices.shape[0] < self.indptr[self.nrows]:
            raise ValueError('Last value of indptr should be <= the size of indices and data.')
        if full:
            if self.indptr[self.nrows] > 0:
                if self.ncols <= np.max(np.asarray(self.indices)):
                    raise ValueError('indices values should be < {}'.format(self.ncols))
                if np.min(np.asarray(self.indices)) < 0:
                    raise ValueError('indices values should be >= than 0')
                if np.min(np.diff(np.asarray(self.indptr))) < 0:
                    raise ValueError('indices values must form a non decreasing sequence.')
            if not self._has_sorted_indices():
                raise ValueError('indices should be sorted.')
            if not self._has_canonical_format():
                raise ValueError('indices have duplicate values.')
        return 1

    def check_format(self):
        if not (self.has_canonical_format and self.has_sorted_indices):
            raise ValueError("KarmaSparse should have canonical format")
        self.check_internal_structure(1)

    cpdef bool check_positive(self) except 0:
        cdef LTYPE_t total_nnz = self.data.shape[0]
        cdef LTYPE_t j

        for j in xrange(total_nnz):
            if self.data[j] < 0:
                raise ValueError('KarmaSparse contains negative values while only positive are expected')

        return 1

    def repair_format(self):
        self.has_sorted_indices = 0
        self.has_canonical_format = 0
        self.make_canonical()
        self.check_format()

    cdef bool prune(self) except 0:
        self.check_internal_structure()
        # TODO : resize inplace if possible
        if max(self.data.shape[0], self.indices.shape[0]) >= 2 * self.get_nnz():
            # copy to free the memory
            self.data = np.asarray(self.data)[:self.get_nnz()].copy()
            self.indices = np.asarray(self.indices)[:self.get_nnz()].copy()
        elif max(self.data.shape[0], self.indices.shape[0]) > self.get_nnz():
            # without coping
            self.data = np.asarray(self.data)[:self.get_nnz()]
            self.indices = np.asarray(self.indices)[:self.get_nnz()]
        return 1

    cdef LTYPE_t get_nnz(self) nogil except -1:
        return self.indptr[self.nrows]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef bool eliminate_zeros(self, DTYPE_t value=0.) except 0:
        cdef LTYPE_t nnz, row_end, j
        cdef ITYPE_t i
        cdef bool has_zero = 0

        self.prune()
        with nogil:
            row_end = 0
            for i in xrange(self.nrows):
                for j in xrange(row_end, self.indptr[i + 1]):
                    if not has_zero and self.data[j] == value:
                        has_zero = 1
                        nnz = j
                    if has_zero and self.data[j] != value:
                        self.indices[nnz] = self.indices[j]
                        self.data[nnz] = self.data[j]
                        nnz += 1
                row_end = self.indptr[i + 1]
                if has_zero:
                    self.indptr[i + 1] = nnz
        self.prune()
        return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool keep_tril(self, ITYPE_t k=0) except 0:
        """
        Inplace method for CSR format
        """
        cdef LTYPE_t nnz, row_end, j
        cdef ITYPE_t i

        with nogil:
            row_end, nnz = 0, 0
            for i in xrange(self.nrows):
                for j in xrange(row_end, self.indptr[i + 1]):
                    if self.indices[j] <= i + k:
                        self.indices[nnz] = self.indices[j]
                        self.data[nnz] = self.data[j]
                        nnz += 1
                row_end = self.indptr[i + 1]
                self.indptr[i + 1] = nnz
        self.prune()
        return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool keep_triu(self, ITYPE_t k=0) except 0:
        """
        Inplace method for CSR format
        """
        cdef LTYPE_t nnz, row_end, j
        cdef ITYPE_t i

        with nogil:
            row_end, nnz = 0, 0
            for i in xrange(self.nrows):
                for j in xrange(row_end, self.indptr[i + 1]):
                    if self.indices[j] >= i + k:
                        self.indices[nnz] = self.indices[j]
                        self.data[nnz] = self.data[j]
                        nnz += 1
                row_end = self.indptr[i + 1]
                self.indptr[i + 1] = nnz
        self.prune()
        return 1

    cpdef KarmaSparse tril(self, ITYPE_t k=0):
        """
        Extract lower triangle matrix from KarmaSparse.
        Return a copy of KarmaSparse with elements above the k-th diagonal zeroed.
        See: numpy.tril, scipy.sparse.tril
        """
        cdef KarmaSparse res = self.copy()
        if self.format == CSR:
            res.keep_tril(k)
        else:
            res.keep_triu(-k)
        return res

    cpdef KarmaSparse triu(self, ITYPE_t k=0):
        """
        Extract upper triangle matrix from KarmaSparse.
        Return a copy of KarmaSparse with elements below the k-th diagonal zeroed.
        See: numpy.triu, scipy.sparse.triu
        """
        cdef KarmaSparse res = self.copy()
        if self.format == CSR:
            res.keep_triu(k)
        else:
            res.keep_tril(-k)
        return res

    def check_sorted(self):
        return self._has_canonical_format()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool _has_sorted_indices(self):
        cdef ITYPE_t i
        cdef LTYPE_t jj

        with nogil:
            for i in xrange(self.nrows):
                for jj in xrange(self.indptr[i], self.indptr[i + 1] - 1):
                    if self.indices[jj] > self.indices[jj + 1]:
                        self.has_sorted_indices = 0
                        return 0
            self.has_sorted_indices = 1
            return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool sort_indices(self) except 0:
        cdef ITYPE_t i, nn

        for i in prange(self.nrows, nogil=True):
            nn = self.indptr[i + 1] - self.indptr[i]
            partial_sort(&self.indices[self.indptr[i]], &self.data[self.indptr[i]], nn, nn, 0)

        self.has_sorted_indices = 1
        return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool _has_canonical_format(self):
        cdef ITYPE_t i
        cdef LTYPE_t jj

        with nogil:
            for i in xrange(self.nrows):
                if self.has_canonical_format == 0 or self.indptr[i + 1] < self.indptr[i]:
                    self.has_canonical_format = 0
                    return 0
                for jj in xrange(self.indptr[i], self.indptr[i + 1] - 1):
                    if self.indices[jj] >= self.indices[jj + 1]:
                        self.has_canonical_format = 0
                        return 0
            self.has_canonical_format = 1
            self.has_sorted_indices = 1
            return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool make_canonical(self, string aggregator=DEFAULT_AGG) except 0:

        cdef:
            LTYPE_t nnz, row_end, jj, temp
            ITYPE_t i, j, pos
            DTYPE_t x
            ITYPE_t * address
            binary_func op = get_reducer(aggregator)

        self.prune()
        if self.has_canonical_format or self._has_canonical_format():
            return 1

        if self.has_sorted_indices or self._has_sorted_indices():
            # Case 1: already sorted indices
            with nogil:
                row_end, nnz = 0, 0
                for i in xrange(self.nrows):
                    jj = row_end
                    row_end = self.indptr[i + 1]
                    while jj < row_end:
                        j = self.indices[jj]
                        x = self.data[jj]
                        jj += 1
                        while jj < row_end and self.indices[jj] == j:
                            x = op(x, self.data[jj])
                            jj += 1
                        if x != 0:
                            self.indices[nnz] = j
                            self.data[nnz] = x
                            nnz += 1
                    self.indptr[i + 1] = nnz
        else:
            # Case 2: unsorted indices : we reduce deduplicate indices first, then sort
            with nogil:
                address = <ITYPE_t*>malloc(self.ncols * sizeof(ITYPE_t))
                memset(address, -1, self.ncols * sizeof(ITYPE_t))

                row_end, nnz, temp = -1, 0, 0
                for i in xrange(self.nrows):
                    for jj in xrange(temp, self.indptr[i + 1]):
                        j = self.indices[jj]
                        pos = address[j]
                        if pos > row_end:
                            self.data[pos] = op(self.data[pos], self.data[jj])
                        else:
                            address[j] = nnz
                            if nnz != jj:
                                self.indices[nnz] = j
                                self.data[nnz] = self.data[jj]
                            nnz += 1
                    temp = self.indptr[i + 1]
                    self.indptr[i + 1] = nnz
                    row_end = nnz - 1
                free(address)
            self.eliminate_zeros()
            self.sort_indices()

        self.prune()
        self.has_canonical_format = 1
        return 1

    def to_scipy_sparse(self, dtype=None, copy=True):
        return getattr(sp, self.format + "_matrix")((np.array(self.data, dtype=dtype, copy=copy),
                                                     np.array(self.indices, copy=copy),
                                                     np.array(self.indptr, copy=copy)),
                                                    shape=self.shape)

    cpdef KarmaSparse copy(self):
        cdef KarmaSparse res = KarmaSparse((self.data, self.indices, self.indptr),
                                           self.shape, self.format, copy=True,
                                           has_sorted_indices=1, has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef np.ndarray[dtype=DTYPE_t, ndim=2] toarray(self):
        if self.nnz == 0:
            return np.zeros(self.shape, dtype=DTYPE, order="C")

        cdef:
            DTYPE_t[:,::1] res = np.zeros((self.nrows, self.ncols), dtype=DTYPE, order="C")
            ITYPE_t i
            LTYPE_t j

        self.check_internal_structure(1)
        for i in prange(self.nrows, nogil=True):
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                res[i, self.indices[j]] += self.data[j]

        if self.format == CSR:
            return np.asarray(res)
        else:
            return np.asarray(res).transpose()

    def __array__(self, dtype=None):
        return self.toarray().astype(dtype=dtype, copy=False)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef KarmaSparse element_sample(self, DTYPE_t proba, seed=None):
        assert 0 <= proba <= 1., "Probability should be between 0 and 1: {}".format(proba)
        cdef:
            KarmaSparse res = self.copy()
            LTYPE_t i
        np.random.seed(seed)
        srand(np.random.randint(0, RAND_MAX))
        np.random.seed(None)
        with nogil:
            for i in xrange(res.get_nnz()):
                if proba * RAND_MAX < rand():
                    res.data[i] = 0
        res.eliminate_zeros()
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef tuple nonzero(self):
        self.eliminate_zeros()
        col = np.asarray(self.indices).copy()
        cdef:
            ITYPE_t[::1] row = np.zeros(self.get_nnz(), dtype=ITYPE)
            ITYPE_t i
            LTYPE_t j

        with nogil:
            for i in xrange(self.nrows):
                for j in xrange(self.indptr[i], self.indptr[i + 1]):
                    row[j] = i

        if self.format == CSR:
            res = (np.asarray(row), np.asarray(col))
        else:
            res = (np.asarray(col), np.asarray(row))
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] diagonal(self):
        cdef ITYPE_t n = min(self.nrows, self.ncols)
        cdef ITYPE_t i
        cdef LTYPE_t j
        cdef DTYPE_t diag
        cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] res = np.zeros(n, dtype=DTYPE)

        for i in prange(n, nogil=True, schedule='static'):
            diag = 0
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                if self.indices[j] == i:
                    diag = diag + self.data[j]
            res[i] = diag
        return res

    cpdef KarmaSparse transpose(self, bool copy=True):
        cdef KarmaSparse res = KarmaSparse((self.data, self.indices, self.indptr),
                                           pair_swap(self.shape), self.swap_format(),
                                           copy=copy,
                                           has_sorted_indices=1, has_canonical_format=1)
        return res

    cpdef KarmaSparse tocsr(self):
        if self.format == CSR:
            return self.copy()
        else:
            return self.swap_slicing()

    cpdef KarmaSparse tocsc(self):
        if self.format == CSC:
            return self.copy()
        else:
            return self.swap_slicing()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef KarmaSparse extend(self, Shape_t shape, bool copy=True):
        cdef ITYPE_t nb_new_rows

        check_nonzero_shape(shape)
        assert (self.shape[0] <= shape[0]) & (self.shape[1] <= shape[1]), \
               'The new shape should be greated than original shape.'
        if self.shape == shape:
            return self.copy()
        if self.format == CSR:
            nb_new_rows = shape[0] - self.shape[0]
        else:
            nb_new_rows = shape[1] - self.shape[1]
        cdef KarmaSparse res = KarmaSparse((self.data, self.indices,
                                            np.hstack([np.asarray(self.indptr),
                                                       self.get_nnz() * np.ones(nb_new_rows,
                                                                                dtype=LTYPE)])),
                                            shape, self.format, copy=copy,
                                            has_sorted_indices=1, has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse swap_slicing(self):
        cdef:
            DTYPE_t[::1] data = np.zeros(self.get_nnz(), dtype=DTYPE, order="C")
            ITYPE_t[::1] indices = np.zeros(self.get_nnz(), dtype=ITYPE, order="C")
            LTYPE_t[::1] indptr = np.zeros(self.ncols + 1, dtype=LTYPE, order="C")
            ITYPE_t col, row
            LTYPE_t i, cumsum, temp, jj, dest, last

        with nogil:
            for i in xrange(self.get_nnz()):
               indptr[self.indices[i]] += 1

            cumsum = 0
            for col in xrange(self.ncols):
                temp = indptr[col]
                indptr[col] = cumsum
                cumsum += temp
            indptr[self.ncols] = cumsum

            for row in xrange(self.nrows):
                for jj in xrange(self.indptr[row], self.indptr[row + 1]):
                    col = self.indices[jj]
                    dest = indptr[col]
                    indices[dest] = row
                    data[dest] = self.data[jj]
                    indptr[col] += 1

            last = 0
            for col in xrange(self.ncols):
                temp = indptr[col]
                indptr[col] = last
                last = temp
        cdef KarmaSparse res = KarmaSparse((data, indices, indptr), self.shape,
                                           self.swap_format(), copy=False,
                                           has_sorted_indices=1, has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool scale_rows(self, np.ndarray factor) except 0:
        check_shape_comptibility(factor.shape[0], self.nrows)
        cdef:
            ITYPE_t i
            LTYPE_t jj
            DTYPE_t[::1] fw = np.asarray(factor, dtype=DTYPE, order="C")

        for i in prange(self.nrows, nogil=True, schedule='static'):
            for jj in xrange(self.indptr[i], self.indptr[i + 1]):
                self.data[jj] *= fw[i]
        return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool scale_columns(self, np.ndarray factor) except 0:
        check_shape_comptibility(factor.shape[0], self.ncols)
        cdef:
            LTYPE_t i
            DTYPE_t[::1] fw = np.asarray(factor, dtype=DTYPE, order="C")

        with nogil:
            for i in xrange(self.get_nnz()):
                self.data[i] *= fw[self.indices[i]]
        return 1

    cpdef KarmaSparse scale_along_axis(self, np.ndarray factor, axis):
        cdef KarmaSparse res = self.copy()
        if res.aligned_axis(axis):
            res.scale_rows(factor)
        else:
            res.scale_columns(factor)
        res.eliminate_zeros()
        return res

    cpdef KarmaSparse scale_along_axis_inplace(self, np.ndarray factor, axis):
        if self.aligned_axis(axis):
            self.scale_rows(factor)
        else:
            self.scale_columns(factor)
        self.eliminate_zeros()
        return self

    cpdef KarmaSparse apply_pointwise_function(self, function, function_args=[], function_kwargs={}):
        cdef KarmaSparse res = KarmaSparse((function(np.asarray(self.data), *function_args, **function_kwargs),
                                            np.array(self.indices, copy=True),
                                            np.array(self.indptr, copy=True)),
                                           self.shape, self.format, copy=False,
                                           has_sorted_indices=1, has_canonical_format=1)
        res.eliminate_zeros()
        return res

    cdef KarmaSparse scalar_multiply(self, DTYPE_t factor):
        cdef KarmaSparse res
        if factor == 0:
            return KarmaSparse(self.shape, format=self.format)
        else:
            return self.apply_pointwise_function(np.multiply, [factor])

    cdef KarmaSparse scalar_divide(self, DTYPE_t factor):
        if factor == 0:
            raise ZeroDivisionError()
        return self.apply_pointwise_function(np.divide, [factor])

    cdef KarmaSparse scalar_add(self, DTYPE_t scalar):
        self.eliminate_zeros()
        return self.apply_pointwise_function(np.add, [scalar])

    cpdef KarmaSparse nonzero_mask(self):
        self.eliminate_zeros()
        return self.apply_pointwise_function(np.ones_like)

    cpdef KarmaSparse abs(self):
        return self.apply_pointwise_function(np.abs)

    cpdef KarmaSparse rint(self):
        return self.apply_pointwise_function(np.rint)

    cpdef KarmaSparse sign(self):
        return self.apply_pointwise_function(np.sign)

    cpdef KarmaSparse trunc(self):
        return self.apply_pointwise_function(np.trunc)

    cpdef KarmaSparse sqrt(self):
        return self.apply_pointwise_function(np.sqrt)

    cpdef KarmaSparse power(self, DTYPE_t p):
        return self.apply_pointwise_function(cython_power, [p])

    cpdef KarmaSparse clip(self, lower, upper=None):
        return self.apply_pointwise_function(np.clip, [lower, upper])

    cpdef KarmaSparse log(self):
        return self.apply_pointwise_function(np.log)

    cpdef KarmaSparse truncate_with_cutoff(self, DTYPE_t cutoff):
        data = np.array(self.data, copy=True)
        data[data < cutoff] = 0.
        cdef KarmaSparse res = KarmaSparse((data, np.array(self.indices, copy=True),
                                           np.array(self.indptr, copy=True)),
                                           self.shape, self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        res.eliminate_zeros()
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_truncate_by_count(self, np.ndarray[ITYPE_t, ndim=1] count):
        assert count.shape[0] + 1== self.indptr.shape[0]
        mm = count.min()
        if mm < 0:
            raise ValueError("nb should be positive, got {}".format(mm))

        if self.get_nnz() == 0 or np.all(np.max(self.aligned_count_nonzero()) <= count):
            return self.copy()
        if np.all(count == 0):
            return 0 * self
        cdef:
            ITYPE_t i, size
            DTYPE_t[::1] new_data = np.zeros(self.get_nnz(), dtype=DTYPE)
            ITYPE_t[::1] new_indices = np.zeros(self.get_nnz(), dtype=ITYPE)
            LTYPE_t[::1] new_indptr = np.zeros(self.nrows + 1, dtype=LTYPE)

        for i in prange(self.nrows, nogil=True, schedule='static'):
            size = self.indptr[i + 1] - self.indptr[i]
            partial_unordered_sort(&self.data[self.indptr[i]], &self.indices[self.indptr[i]], size, min(count[i], size))

        with nogil:
            for i in xrange(self.nrows):
                size = min(count[i], self.indptr[i + 1] - self.indptr[i])
                if size > 0:
                    memcpy(&new_indices[new_indptr[i]], &self.indices[self.indptr[i]], size * sizeof(ITYPE_t))
                    memcpy(&new_data[new_indptr[i]], &self.data[self.indptr[i]], size * sizeof(DTYPE_t))
                    partial_sort(&new_indices[new_indptr[i]], &new_data[new_indptr[i]], size, size, False)
                new_indptr[i + 1] = new_indptr[i] + size
        self.sort_indices()
        cdef KarmaSparse res = KarmaSparse((new_data, new_indices, new_indptr),
                                           self.shape, self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_truncate_by_budget(self, density, DTYPE_t volume):
        check_shape_comptibility(len(density), self.ncols)
        if self.get_nnz() == 0:
            return self.copy()
        cdef:
            ITYPE_t i, j, size
            DTYPE_t vol
            DTYPE_t[::1] densi = np.asarray(density, dtype=DTYPE, order="C")
            DTYPE_t[::1] new_data = np.zeros(self.get_nnz(), dtype=DTYPE)
            ITYPE_t[::1] new_indices = np.zeros(self.get_nnz(), dtype=ITYPE)
            LTYPE_t[::1] new_indptr = np.zeros(self.nrows + 1, dtype=LTYPE)

        with nogil:
            for i in prange(self.nrows, schedule='static'):
                size = self.indptr[i + 1] - self.indptr[i]
                partial_sort(&self.data[self.indptr[i]], &self.indices[self.indptr[i]], size, size)

            for i in xrange(self.nrows):
                size = self.indptr[i + 1] - self.indptr[i]
                if size > 0:
                    vol = 0
                    for j in xrange(size):
                        vol += densi[self.indices[self.indptr[i] + j]]
                        if vol >= volume:
                            break
                    size = j + 1
                new_indptr[i + 1] = new_indptr[i] + size

            for i in prange(self.nrows, schedule='static'):
                size = new_indptr[i + 1] - new_indptr[i]
                memcpy(&new_data[new_indptr[i]], &self.data[self.indptr[i]], size * sizeof(DTYPE_t))
                memcpy(&new_indices[new_indptr[i]], &self.indices[self.indptr[i]], size * sizeof(ITYPE_t))
                partial_sort(&new_indices[new_indptr[i]], &new_data[new_indptr[i]], size, size, False)

        self.sort_indices() # TODO avoid sorting by values and indices in place

        cdef KarmaSparse res = KarmaSparse((new_data, new_indices, new_indptr),
                                           self.shape, self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_truncate_cumulative(self, DTYPE_t percentage):
        if self.get_nnz() == 0:
            return self.copy()
        cdef:
            ITYPE_t i, j, size
            DTYPE_t[::1] max_volume = (1 - percentage) * self.aligned_sum()
            DTYPE_t current_volume
            DTYPE_t[::1] new_data = np.zeros(self.get_nnz(), dtype=DTYPE)
            ITYPE_t[::1] new_indices = np.zeros(self.get_nnz(), dtype=ITYPE)
            LTYPE_t[::1] new_indptr = np.zeros(self.nrows + 1, dtype=LTYPE)

        with nogil:
            for i in prange(self.nrows, schedule='static'):
                size = self.indptr[i + 1] - self.indptr[i]
                partial_sort(&self.data[self.indptr[i]], &self.indices[self.indptr[i]],
                             size, size, reverse=True)

            for i in xrange(self.nrows):
                size = self.indptr[i + 1] - self.indptr[i]
                if size > 0:
                    current_volume = 0
                    for j in xrange(size):
                        current_volume += self.data[self.indptr[i] + j]
                        if current_volume >= max_volume[i]:
                            break
                    size = j + 1
                new_indptr[i + 1] = new_indptr[i] + size

            for i in prange(self.nrows, schedule='static'):
                size = new_indptr[i + 1] - new_indptr[i]
                memcpy(&new_data[new_indptr[i]], &self.data[self.indptr[i]], size * sizeof(DTYPE_t))
                memcpy(&new_indices[new_indptr[i]], &self.indices[self.indptr[i]], size * sizeof(ITYPE_t))
                partial_sort(&new_indices[new_indptr[i]], &new_data[new_indptr[i]], size, size, False)

        self.sort_indices()  # TODO avoid sorting by values and indices in place

        cdef KarmaSparse res = KarmaSparse((new_data, new_indices, new_indptr),
                                           shape=self.shape, format=self.format,
                                           copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse global_truncate_cumulative(self, DTYPE_t percentage):
        """This method is more relaxed than the directed one"""
        cdef:
            DTYPE_t[::1] data = np.array(self.data, copy=True)
            DTYPE_t[::1] new_data = np.zeros(self.get_nnz(), dtype=DTYPE)
            LTYPE_t[::1] order = np.arange(self.get_nnz(), dtype=LTYPE)
            DTYPE_t max_volume = self.sum(axis=None) * (1 - percentage)
            LTYPE_t i, size = self.get_nnz()
            DTYPE_t last, volume = 0
        with nogil:
            partial_sort(&data[0], &order[0], size, size, reverse=True)
            for i in xrange(size):
                volume += data[i]
                if volume >= max_volume:
                    last = data[i]
                    break
            size = i + 1
            for i in prange(size, schedule="static"):
                new_data[order[i]] = data[i]

        cdef KarmaSparse res = KarmaSparse((new_data, np.array(self.indices, copy=True),
                                            np.array(self.indptr, copy=True)),
                                           shape=self.shape, format=self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        res.eliminate_zeros()
        return res

    cpdef KarmaSparse truncate_by_count(self, nb, axis):
        """
        Keep only `nb` largest elements along `axis`.
        This takes into account only on nonzero elements, so it will truncate
        negative elements as well even if many zeros elements are present.
        Thus, this behaviour is different from the dense (and it returns the same result
        if all matrix elements are positive).

        If axis=None, perform it will keep nb largest elements among all nonzero ones.

        Example::
            >>> ks = KarmaSparse(np.array([[-1, 2, 0, -1.1], [3, 4, -3, -1]]))
            >>> ks.truncate_by_count(nb=1, axis=1).toarray()
            array([[0., 2., 0., 0.],
                   [0., 4., 0., 0.]])
            >>> ks.truncate_by_count(nb=1, axis=0).toarray()
            array([[ 0.,  0.,  0.,  0.],
                   [ 3.,  4., -3., -1.]])
            >>> ks.truncate_by_count(nb=2, axis=1).toarray()
            array([[-1.,  2.,  0.,  0.],
                   [ 3.,  4.,  0.,  0.]])
        """
        assert np.all(nb >= 0), 'nb values should be non-negative'
        if axis is None:
            res = self.copy()
            data = np.asarray(res.data)
            indices_to_remove = np.argpartition(-data, nb)[nb:]
            data[indices_to_remove] = 0.
            res.data = data
            res.eliminate_zeros()
            return res

        if is_int(nb):
            count = np.full(self.shape[1 - axis], nb, dtype=ITYPE)
        else:
            count = np.asarray(nb, dtype=ITYPE)
        if self.aligned_axis(axis):
            return self.aligned_truncate_by_count(count)
        else:
            return self.swap_slicing().aligned_truncate_by_count(count).swap_slicing()

    cpdef KarmaSparse truncate_by_budget(self, np.ndarray values, DTYPE_t budget, axis):
        """
        Truncate KarmaSparse along a given `axis` to reach a total budget.
        `values` represents the value of each row (axis=1) or col (axis=0)
        This will keep the minimal number of largest elements along axis under condition that
        the sum of their values is greater than `budget`.

        Takes into account only nonzero elements.

        Example::
            >>> ks = KarmaSparse(np.array([[-1, 2, 0, -1], [3, 4, -3, -1]]))
            >>> values = np.array([1, -1, 2, 3])
            >>> ks.truncate_by_budget(values, budget=0., axis=1).toarray()
            array([[-1.,  2.,  0.,  0.],
                   [ 3.,  4.,  0.,  0.]])
            >>> ks.truncate_by_budget(values, budget=0.1, axis=1).toarray()
            array([[-1.,  2.,  0., -1.],
                   [ 3.,  4.,  0., -1.]])
            >>> values = np.array([1, 1, 1, 1])  # same as truncate_by_count
            >>> ks.truncate_by_budget(values, budget=2, axis=1).toarray()
            array([[-1.,  2.,  0.,  0.],
                   [ 3.,  4.,  0.,  0.]])
        """
        if self.aligned_axis(axis):
            return self.aligned_truncate_by_budget(values, budget)
        else:
            return self.swap_slicing().aligned_truncate_by_budget(values,
                                                                  budget).swap_slicing()

    cpdef KarmaSparse truncate_by_cumulative(self, DTYPE_t percentage, axis):
        assert 0 <= percentage <= 1, "percentage should be between 0 and 1, got {}".format(percentage)
        if axis is None:
            return self.global_truncate_cumulative(percentage)
        if self.aligned_axis(axis):
            return self.aligned_truncate_cumulative(percentage)
        else:
            return self.swap_slicing().aligned_truncate_cumulative(percentage).swap_slicing()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse csr_generic_dot_top(self, KarmaSparse other,
                                         ITYPE_t nb_keep, DTYPE_t cutoff, binary_func op):
        check_shape_comptibility(self.ncols, other.nrows)
        assert nb_keep > 0, "Parameter nb_keep should be positive, got {}".format(nb_keep)
        cdef:
            ITYPE_t nrows = self.nrows
            ITYPE_t ncols = other.ncols
            DTYPE_t * ires
            DTYPE_t * icol
            ITYPE_t * locind
            DTYPE_t ival
            LTYPE_t j, k
            ITYPE_t i, nn, ind
            ITYPE_t[::1] count = np.zeros(nrows, dtype=ITYPE, order="C")
            ITYPE_t nb = min(nb_keep, ncols)
            DTYPE_t[:, ::1] sort_values = np.zeros((nrows, nb), dtype=DTYPE, order="C")
            ITYPE_t[:, ::1] sort_index = np.zeros((nrows, nb), dtype=ITYPE, order="C")
            DTYPE_t[::1] data
            ITYPE_t[::1] indices
            LTYPE_t[::1] indptr

        with nogil, parallel():
            ires = <DTYPE_t *>malloc(ncols * sizeof(DTYPE_t))
            icol = <DTYPE_t *>malloc(ncols * sizeof(DTYPE_t))
            locind = <ITYPE_t *>malloc(ncols * sizeof(ITYPE_t))
            if ires == NULL or icol == NULL or locind == NULL:
                with gil:
                    raise MemoryError()
            for i in prange(nrows, schedule='guided'):
                memset(ires, 0, ncols * sizeof(DTYPE_t))
                for k in xrange(self.indptr[i], self.indptr[i + 1]):
                    ind, ival = self.indices[k], self.data[k]
                    for j in xrange(other.indptr[ind], other.indptr[ind + 1]):
                        ires[other.indices[j]] += op(ival, other.data[j])
                # set index
                nn = 0
                for k in xrange(ncols):
                    if ires[k] > cutoff and ires[k] != 0:
                        locind[nn] = k
                        icol[nn] = ires[k]
                        nn = nn + 1
                count[i] = min(nn, nb)

                if nn > nb:  # sort them
                    partial_sort(icol, locind, nn, count[i])
                memcpy(&sort_values[i, 0], icol, count[i] * sizeof(DTYPE_t))
                memcpy(&sort_index[i, 0], locind, count[i] * sizeof(ITYPE_t))
            free(ires)
            free(icol)
            free(locind)

        indptr = np.zeros(nrows + 1, dtype=LTYPE, order="C")
        with nogil:
            for i in xrange(nrows):
                indptr[i + 1] = indptr[i] + count[i]

        data = np.zeros(indptr[nrows], dtype=DTYPE, order="C")
        indices = np.zeros(indptr[nrows], dtype=ITYPE, order="C")

        for i in prange(nrows, nogil=True, schedule='static'):
            memcpy(&data[indptr[i]], &sort_values[i, 0], count[i] * sizeof(DTYPE_t))
            memcpy(&indices[indptr[i]], &sort_index[i, 0], count[i] * sizeof(ITYPE_t))
            partial_sort(&indices[indptr[i]], &data[indptr[i]], count[i], count[i], 0)
        return KarmaSparse((data, indices, indptr),
                           (nrows, ncols), self.format, copy=False,
                           has_sorted_indices=1, has_canonical_format=1)

    cdef KarmaSparse generic_dot_top(self, KarmaSparse other,
                                     ITYPE_t nb_keep, DTYPE_t cutoff, binary_func op):
        if (self.format == CSR) and (other.format == CSR):
            return self.csr_generic_dot_top(other, nb_keep, cutoff, op)
        elif self.format == CSR:
            return self.csr_generic_dot_top(other.tocsr(), nb_keep, cutoff, op)
        elif other.format == CSR:
            return self.tocsr().csr_generic_dot_top(other, nb_keep, cutoff, op)
        else:
            return self.tocsr().csr_generic_dot_top(other.tocsr(), nb_keep, cutoff, op)

    cpdef KarmaSparse sparse_dot_top(self, KarmaSparse other, ITYPE_t nb_keep):
        return self.generic_dot_top(other, nb_keep, -np.inf, get_reducer('multiply'))

    cpdef KarmaSparse pairwise_min_top(self, KarmaSparse other,
                                       ITYPE_t nb_keep, DTYPE_t cutoff):
        return self.generic_dot_top(other, nb_keep, cutoff, get_reducer('min'))

    cpdef KarmaSparse pairwise_max_top(self, KarmaSparse other,
                                       ITYPE_t nb_keep, DTYPE_t cutoff):
        return self.generic_dot_top(other, nb_keep, cutoff, get_reducer('max'))

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_sparse_dot(self, KarmaSparse other):
        check_shape_comptibility(self.ncols, other.nrows)
        cdef:
            ITYPE_t nrows = self.nrows
            ITYPE_t ncols = other.ncols
            LTYPE_t jj, kk
            ITYPE_t row_nnz, j, i, k
            ITYPE_t ind, nn, head, tmp_head
            DTYPE_t ival
            ITYPE_t * mask
            DTYPE_t * ires
            ITYPE_t[::1] count = np.zeros(nrows, dtype=ITYPE)
            LTYPE_t[::1] indptr = np.zeros(nrows + 1, dtype=LTYPE)
            ITYPE_t[::1] indices
            DTYPE_t[::1] data
            double density

        with nogil, parallel():
            mask = <ITYPE_t *>malloc(ncols * sizeof(ITYPE_t))
            if mask == NULL:
                with gil: raise MemoryError()
            memset(mask, -1, ncols * sizeof(ITYPE_t))
            for i in prange(nrows, schedule='static'):
                row_nnz = 0
                for jj in xrange(self.indptr[i], self.indptr[i + 1]):
                    j = self.indices[jj]
                    for kk in xrange(other.indptr[j], other.indptr[j + 1]):
                        k = other.indices[kk]
                        if mask[k] != i:
                            mask[k] = i
                            row_nnz = row_nnz + 1
                count[i] = row_nnz
            free(mask)
        with nogil:
            for i in xrange(nrows):
                indptr[i + 1] = indptr[i] + count[i]

        data = np.zeros(indptr[nrows], dtype=DTYPE)
        indices = np.zeros(indptr[nrows], dtype=ITYPE)

        density = 1. * indptr[nrows] / max(ncols, 1) / max(nrows, 1)
        if density < 0.02:  # experimental constant: TODO improve it
            with nogil, parallel():
                ires = <DTYPE_t *>calloc(ncols, sizeof(DTYPE_t))
                mask = <ITYPE_t *>malloc(ncols * sizeof(ITYPE_t))
                if ires == NULL or mask == NULL:
                    with gil: raise MemoryError()
                memset(mask, -1, ncols * sizeof(ITYPE_t))
                for i in prange(nrows, schedule='guided'):
                    head = - 2
                    for kk in xrange(self.indptr[i], self.indptr[i + 1]):
                        ind = self.indices[kk]
                        ival = self.data[kk]
                        for jj in xrange(other.indptr[ind], other.indptr[ind + 1]):
                            j = other.indices[jj]
                            ires[j] += mult(other.data[jj], ival)
                            if mask[j] == -1:
                                mask[j] = head
                                head = j
                    nn = 0
                    for k in xrange(count[i]):
                        if ires[head] != 0:
                            indices[indptr[i] + nn] = head
                            data[indptr[i] + nn] = ires[head]
                            nn = nn + 1
                        tmp_head = head
                        head = mask[head]
                        mask[tmp_head] = -1
                        ires[tmp_head] = 0
                free(ires)
                free(mask)
        else:
            with nogil, parallel():
                ires = <DTYPE_t *>calloc(ncols, sizeof(DTYPE_t))
                if ires == NULL:
                    with gil: raise MemoryError()
                for i in prange(nrows, schedule='guided'):
                    for kk in xrange(self.indptr[i], self.indptr[i + 1]):
                        ind = self.indices[kk]
                        ival = self.data[kk]
                        for jj in xrange(other.indptr[ind], other.indptr[ind + 1]):
                            j = other.indices[jj]
                            ires[j] += mult(other.data[jj], ival)
                    nn = 0
                    if count[i] > 0:
                        for k in xrange(ncols):
                            if ires[k] != 0:
                                indices[indptr[i] + nn] = k
                                data[indptr[i] + nn] = ires[k]
                                ires[k] = 0
                                nn = nn + 1
                free(ires)
        res = KarmaSparse((np.asarray(data), np.asarray(indices), np.asarray(indptr)),
                          (nrows, ncols), self.format, copy=False)
        return res

    cdef KarmaSparse sparse_dot(self, KarmaSparse other):
        if self.format == CSR and other.format == CSR:
            return self.aligned_sparse_dot(other)
        elif self.format == CSR:
            return self.aligned_sparse_dot(other.tocsr())
        elif other.format == CSR:
            return self.tocsr().aligned_sparse_dot(other)
        else:
            return other.transpose(copy=False)\
                        .aligned_sparse_dot(self.transpose(copy=False))\
                        .transpose(copy=False)

    def dot(self, other):
        if is_karmasparse(other):
            return self.sparse_dot(other)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                return self.dense_vector_dot_right(other)
            else:
                return self.dense_dot_right(other)
        elif sp.issparse(other):
            return self.sparse_dot(KarmaSparse(other, copy=False))
        else:
            raise NotImplementedError()

    def pairwise_l2square_dense_distance(self):
        norm = np.atleast_2d(self.sum_power(axis=1, power=2))
        return norm + norm.transpose() - 2 * self.dot(self.transpose(copy=False)).toarray()

    def kronii(self, other):
        check_shape_comptibility(self.shape[0], other.shape[0])
        if self.format == 'csr':
            if isinstance(other, np.ndarray):
                if other.dtype == np.float32:
                    return self.kronii_align_dense[float](other)
                else:
                    return self.kronii_align_dense[double](other.astype(DTYPE, copy=False))
            else:
                return self.kronii_align_sparse(KarmaSparse(other, format='csr', copy=False))
        else:
            if isinstance(other, np.ndarray):
                self_transp = self.swap_slicing()
                if other.dtype == np.float32:
                    return self_transp.kronii_align_dense[float](other)
                else:
                    return self_transp.kronii_align_dense[double](other.astype(DTYPE, copy=False))
            else:
                return self.swap_slicing().kronii_align_sparse(KarmaSparse(other, format='csr', copy=False))

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse kronii_align_dense(self, floating[:,:] other):
        cdef LTYPE_t ncols = other.shape[1]
        cdef LTYPE_t nnz = self.nnz * ncols
        cdef LTYPE_t[::1] indptr = np.asarray(self.indptr) * ncols
        cdef ITYPE_t[::1] indices = np.zeros(nnz, dtype=ITYPE)
        cdef DTYPE_t[::1] data = np.zeros(nnz, dtype=DTYPE)
        cdef LTYPE_t i, j, k, ind, pos, ind_pos, ii
        cdef DTYPE_t alpha

        for i in prange(self.nrows, nogil=True):
            for j in xrange(self.indptr[i], self.indptr[i+1]):
                alpha = self.data[j]
                ind = self.indices[j]
                pos = j * ncols
                ind_pos = ind * ncols
                for k in xrange(ncols):
                    ii = pos + k
                    data[ii] = alpha * other[i, k]
                    indices[ii] = ind_pos + k

        shape = (self.nrows, self.ncols * ncols)
        ks = KarmaSparse((data, indices, indptr), shape=shape, format="csr",
                           copy=False, has_sorted_indices=True, has_canonical_format=True)
        ks.eliminate_zeros()
        return ks

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse kronii_align_sparse(self, KarmaSparse other):
        cdef LTYPE_t[::1] indptr = np.zeros(self.nrows + 1, dtype=LTYPE)
        cdef LTYPE_t i, j, k, ind, pos, start, stop, kk, size, ncols = other.ncols
        cdef DTYPE_t alpha

        with nogil:
            for i in xrange(self.nrows):
                indptr[i + 1] = indptr[i] + \
                    (self.indptr[i + 1] - self.indptr[i]) * (other.indptr[i + 1] - other.indptr[i])

        cdef ITYPE_t[::1] indices = np.zeros(indptr[self.nrows], dtype=ITYPE)
        cdef DTYPE_t[::1] data = np.zeros(indptr[self.nrows], dtype=DTYPE)

        for i in prange(self.nrows, nogil=True):
            start, stop = other.indptr[i], other.indptr[i + 1]
            size = stop - start
            pos = indptr[i]
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                alpha = self.data[j]
                ind = self.indices[j] * ncols
                for kk in xrange(start, stop):
                    data[pos] = alpha * other.data[kk]
                    indices[pos] = ind + other.indices[kk]
                    pos = pos + 1

        shape = (self.nrows, self.ncols * other.ncols)
        return KarmaSparse((data, indices, indptr), shape=shape, format="csr",
                           copy=False, has_sorted_indices=True, has_canonical_format=True)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_axis_reduce(self, binary_func fn, bool only_nonzero):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.nrows, dtype=DTYPE)
            ITYPE_t i
            LTYPE_t j
            DTYPE_t mx

        for i in prange(self.nrows, nogil=True, schedule='static'):
            if self.indptr[i] < self.indptr[i + 1]:
                mx = self.data[self.indptr[i]]
                for j in xrange(self.indptr[i] + 1, self.indptr[i + 1]):
                    mx = fn(mx, self.data[j])
                if not only_nonzero and (self.indptr[i + 1] - self.indptr[i] < self.ncols):
                    res[i] = fn(mx, 0)
                else:
                    res[i] = mx
            else:
                res[i] = 0
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_axis_reduce(self, binary_func fn,
                                                                  bool only_nonzero):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.ncols, dtype=DTYPE)
            ITYPE_t[::1] count = None if only_nonzero else np.zeros(self.ncols, dtype=ITYPE)
            bool * first = <bool *>malloc(self.ncols * sizeof(bool))
            ITYPE_t i
            LTYPE_t j

        with nogil:
            memset(first, 1, self.ncols * sizeof(bool))
            for j in xrange(self.get_nnz()):
                i = self.indices[j]
                if first[i]:
                    res[i] = self.data[j]
                    first[i] = 0
                else:
                    res[i] = fn(res[i], self.data[j])
                if not only_nonzero:
                    count[i] += 1
            if not only_nonzero:
                for i in xrange(self.ncols):
                    if count[i] < self.nrows:
                        res[i] = fn(res[i], 0)
            free(first)
        return res

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] reducer(self, string name, int axis,
                                                   bool only_nonzero=False):
        if self.aligned_axis(axis):
            return self.aligned_axis_reduce(get_reducer(name), only_nonzero)
        else:
            return self.misaligned_axis_reduce(get_reducer(name), only_nonzero)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum(self):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.nrows, dtype=DTYPE)
            ITYPE_t i
            LTYPE_t j
            DTYPE_t mx

        for i in prange(self.nrows, nogil=True, schedule='static'):
            mx = 0
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                mx = mx + self.data[j]
            res[i] = mx
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum_abs(self):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.nrows, dtype=DTYPE)
            ITYPE_t i
            LTYPE_t j
            DTYPE_t mx

        for i in prange(self.nrows, nogil=True, schedule='static'):
            mx = 0
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                mx = mx + fabs(self.data[j])
            res[i] = mx
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_max_abs(self):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.nrows, dtype=DTYPE)
            ITYPE_t i
            LTYPE_t j
            DTYPE_t mx

        for i in prange(self.nrows, nogil=True, schedule='static'):
            mx = 0
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                mx = max(mx, fabs(self.data[j]))
            res[i] = mx
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_max_abs(self):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.ncols, dtype=DTYPE)
            LTYPE_t i
        with nogil:
            for i in xrange(self.get_nnz()):
                res[self.indices[i]] = max(fabs(self.data[i]), res[self.indices[i]])
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum(self):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.ncols, dtype=DTYPE)
            LTYPE_t i
        with nogil:
            for i in xrange(self.get_nnz()):
                res[self.indices[i]] += self.data[i]
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum_abs(self):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.ncols, dtype=DTYPE)
            LTYPE_t i
        with nogil:
            for i in xrange(self.get_nnz()):
                res[self.indices[i]] += fabs(self.data[i])
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_sum_power(self, DTYPE_t p):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.nrows, dtype=DTYPE)
            ITYPE_t i
            LTYPE_t j
            DTYPE_t mx

        for i in prange(self.nrows, nogil=True, schedule='static'):
            mx = 0
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                mx = mx + cpow(self.data[j], p)
            res[i] = mx
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_sum_power(self, DTYPE_t p):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.ncols, dtype=DTYPE)
            LTYPE_t i
        with nogil:
            for i in xrange(self.get_nnz()):
                res[self.indices[i]] += cpow(self.data[i], p)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_count_nonzero(self):
        cdef:
            np.ndarray[dtype=ITYPE_t, ndim=1, mode="c"] res = np.zeros(self.nrows, dtype=ITYPE)
            ITYPE_t i

        self.make_canonical()
        for i in prange(self.nrows, nogil=True, schedule='static'):
            res[i] = self.indptr[i + 1] - self.indptr[i]
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] misaligned_count_nonzero(self):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.ncols, dtype=DTYPE)
            LTYPE_t i

        self.eliminate_zeros()
        with nogil:
            for i in xrange(self.get_nnz()):
                res[self.indices[i]] += 1
        return res

    cdef sum_abs(self, axis=None):
        if axis is None:
            return np.sum(np.abs(self.data))
        else:
            if self.aligned_axis(axis):
                return self.aligned_sum_abs()
            else:
                return self.misaligned_sum_abs()

    cdef max_abs(self, axis=None):
        if axis is None:
            if self.get_nnz():
                return np.absolute(np.asarray(self.data)).max()
            else:
                return 0.0
        if self.aligned_axis(axis):
            return self.aligned_max_abs()
        else:
            return self.misaligned_max_abs()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse global_rank(self, bool reverse=False):
        cdef:
            DTYPE_t[::1] data = np.array(self.data, order="C", dtype=DTYPE, copy=True)
            LTYPE_t[::1] order = np.arange(1, self.get_nnz() + 1, dtype=LTYPE)
            LTYPE_t[::1] rank = np.array(order, order="C", dtype=LTYPE, copy=True)
            LTYPE_t nn = self.get_nnz()
        with nogil:
            partial_sort(&data[0], &order[0], nn, nn, reverse=reverse)
            partial_sort(&order[0], &rank[0], nn, nn, reverse=False)
        cdef KarmaSparse res = KarmaSparse((rank, np.array(self.indices, copy=True),
                                            np.array(self.indptr, copy=True)),
                                           shape=self.shape, format=self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_rank(self, bool reverse=False):
        cdef:
            DTYPE_t[::1] data = np.array(self.data, copy=True)
            ITYPE_t[::1] rank = np.zeros(self.get_nnz(), dtype=ITYPE, order="C")
            ITYPE_t[::1] order = np.zeros(self.get_nnz(), dtype=ITYPE, order="C")
            ITYPE_t* range_mask
            LTYPE_t i, size

        with nogil:
            size = min(self.get_nnz(), self.ncols)
            range_mask = <ITYPE_t*>malloc(size * sizeof(ITYPE_t))
            for i in prange(size, schedule='static'):
                range_mask[i] = i + 1

            for i in prange(self.nrows, schedule='static'):
                size = self.indptr[i + 1] - self.indptr[i]
                memcpy(&rank[self.indptr[i]], range_mask, size * sizeof(ITYPE_t))
            free(range_mask)

            memcpy(&order[0], &rank[0], self.get_nnz() * sizeof(ITYPE_t))
            for i in prange(self.nrows, schedule='static'):
                size = self.indptr[i + 1] - self.indptr[i]
                partial_sort(&data[self.indptr[i]],
                             &order[self.indptr[i]],
                             size, size, reverse=reverse)
                partial_sort(&order[self.indptr[i]],
                             &rank[self.indptr[i]],
                             size, size, reverse=False)

        cdef KarmaSparse res = KarmaSparse((rank, np.array(self.indices, copy=True),
                                            np.array(self.indptr, copy=True)),
                                           shape=self.shape, format=self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        return res

    cpdef KarmaSparse rank(self, axis, bool reverse=False):
        if axis is None:
            return self.global_rank(reverse=reverse)
        if self.aligned_axis(axis):
            return self.aligned_rank(reverse=reverse)
        else:
            return self.swap_slicing().aligned_rank(reverse=reverse).swap_slicing()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] aligned_quantile(self, DTYPE_t quantile,
                                                            bool only_nonzero=False):
        cdef:
            np.ndarray[dtype=DTYPE_t, ndim=1, mode="c"] res = np.zeros(self.nrows, dtype=DTYPE)
            ITYPE_t i, size, dim

        for i in prange(self.nrows, nogil=True, schedule='static'):
            size = self.indptr[i + 1] - self.indptr[i]
            dim = size if only_nonzero else self.ncols
            if size:
                res[i] = computed_quantile(&self.data[self.indptr[i]], quantile, size, dim)
        return res

    def quantile(self, DTYPE_t q, axis, bool only_nonzero=False):
        cdef LTYPE_t dim, size
        if not 0 <= q <= 1:
            raise ValueError("quantile should be in [0,1] interval, got {}"
                             .format(q))
        self.eliminate_zeros()
        if axis is None:
            size = self.get_nnz()
            if size > 0:
                dim = size if only_nonzero else (<LTYPE_t>self.nrows) * self.ncols
                return computed_quantile(&self.data[0], q, size, dim)
            else:
                return 0.
        if self.aligned_axis(axis):
            return self.aligned_quantile(quantile=q, only_nonzero=only_nonzero)
        else:
            return self.swap_slicing().aligned_quantile(quantile=q,
                                                        only_nonzero=only_nonzero)

    def median(self, axis=None, bool only_nonzero=False):
        return self.quantile(axis=axis, q=0.5, only_nonzero=only_nonzero)

    def sum(self, axis=None):
        if axis is None:
            return np.sum(np.asarray(self.data))
        else:
            if self.aligned_axis(axis):
                return self.aligned_sum()
            else:
                return self.misaligned_sum()

    def mean(self, axis=None):
        if axis is None:
            return 1. * np.sum(np.asarray(self.data)) \
                      / max(self.shape[0], 1) / max(self.shape[1], 1)
        else:
            return 1. * self.sum(axis) / max(self.shape[axis], 1)

    def sum_power(self, power, axis=None):
        if axis is None:
            return np.sum(cython_power(np.asarray(self.data), power))
        else:
            if self.aligned_axis(axis):
                return self.aligned_sum_power(power)
            else:
                return self.misaligned_sum_power(power)

    def norm(self, axis=None, norm="l2"):
        if norm == 'l1':
            return self.sum_abs(axis=axis)
        elif norm == 'l2':
            return np.sqrt(self.sum_power(power=2, axis=axis))
        elif norm == 'linf':
            return self.max_abs(axis=axis)
        else:
            raise Exception('{} rejected as norm. accepted values : "l1", "l2" or "linf"'.format(norm))

    def normalize(self, axis=None, norm="l2",
                  invpow=1., invlog=0., threshold=None, width=1.):
        vnorm = self.norm(axis=axis, norm=norm)
        if axis is None and vnorm == 0:  # zero matrix
            return self.copy()
        elif axis is not None:
            vnorm[vnorm != 0] = 1.0 / vnorm[vnorm != 0]
        factor = np.power(vnorm, invpow)
        factor *= np.power(np.log1p(vnorm), invlog)
        if threshold is not None:
            factor /= logit(vnorm, threshold, width)
        if axis is None:
            return self.scalar_divide(factor)
        else:
            return self.scale_along_axis(vnorm, axis)

    def count_nonzero(self, axis=None):
        if axis is None:
            return np.count_nonzero(np.asarray(self.data))
        else:
            if self.aligned_axis(axis):
                return self.aligned_count_nonzero()
            else:
                return self.misaligned_count_nonzero()

    def var(self, axis=None):
        if axis is None:
            nb = 1. / max(self.shape[0], 1) / max(self.shape[1], 1)
        else:
            nb = 1. / max(self.shape[axis], 1)
        mean = self.mean(axis)
        sq_sum = self.sum_power(2, axis)
        return sq_sum * nb - np.power(mean, 2)

    def std(self, axis=None):
        return np.sqrt(self.var(axis))

    def min(self, axis=None):
        if axis is None:
            if self.get_nnz() == 0:
                return 0.
            if self.density < 1:
                return min(np.min(np.asarray(self.data)), 0)
            else:
                return np.min(np.asarray(self.data))
        else:
            return self.reducer('min', axis, False)

    def max(self, axis=None):
        if axis is None:
            if self.get_nnz() == 0:
                return 0.
            if self.density < 1:
                return max(np.max(np.asarray(self.data)), 0)
            else:
                return np.max(np.asarray(self.data))
        else:
            return self.reducer('max', axis, False)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_compatibility_renormalization(self, ITYPE_t[::1] row_gender,
                                                           ITYPE_t[::1] column_gender,
                                                           DTYPE_t homo_factor,
                                                           DTYPE_t hetero_factor):
        check_shape_comptibility(row_gender.shape[0], self.nrows)
        check_shape_comptibility(column_gender.shape[0], self.ncols)
        cdef:
            ITYPE_t i
            LTYPE_t j
            KarmaSparse res = self.copy()
        for i in prange(res.nrows, nogil=True, schedule='static'):
            for j in xrange(res.indptr[i], res.indptr[i + 1]):
                if row_gender[i] == column_gender[self.indices[j]]:
                    res.data[j] *= homo_factor
                else:
                    res.data[j] *= hetero_factor
        res.eliminate_zeros()
        return res

    cpdef KarmaSparse compatibility_renormalization(self, row_gender, column_gender,
                                                    DTYPE_t homo_factor,
                                                    DTYPE_t hetero_factor):
        """
        Any element KS[i,j] of KarmaSparse is multiplied by
            A) homo_factor if `row_gender`[i] == `column_gender`[j]
            B) hetero_factor if `row_gender`[i] != `column_gender`[j]

        Copy (not inplace) method.
        """
        # first convert to numpy array of right type
        _, merged = np.unique(np.hstack([row_gender, column_gender]), return_inverse=True)
        row_gender = np.asarray(merged[:len(row_gender)], dtype=ITYPE)
        column_gender = np.asarray(merged[len(row_gender):], dtype=ITYPE)
        if self.format == CSR:
            return self.aligned_compatibility_renormalization(row_gender, column_gender,
                                                              homo_factor, hetero_factor)
        else:
            return self.aligned_compatibility_renormalization(column_gender, row_gender,
                                                              homo_factor, hetero_factor)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bool aligned_truncate_by_count_by_group(self, raw_group, ITYPE_t nb_keep) except 0:
        """
        Inplace method
        """
        check_shape_comptibility(len(raw_group), self.ncols)
        cat_label, _group = np.unique(np.asarray(raw_group), return_inverse=True)
        if -1 in cat_label:
            _group[raw_group == -1] = -1
        cdef:
            ITYPE_t i, cat, size
            LTYPE_t k
            ITYPE_t nb_cat = cat_label.shape[0]
            ITYPE_t[::1] group = np.asarray(_group, dtype=ITYPE)
            LTYPE_t * cat_count

        with nogil, parallel():
            cat_count = <LTYPE_t *>calloc(nb_cat, sizeof(LTYPE_t))
            if cat_count == NULL:
                with gil: raise MemoryError()
            for i in prange(self.nrows, schedule='static'):
                memset(cat_count, 0, nb_cat * sizeof(LTYPE_t))
                size = self.indptr[i + 1] - self.indptr[i]
                # ordering by data
                partial_sort(&self.data[self.indptr[i]],
                             &self.indices[self.indptr[i]], size, size)
                for k in xrange(self.indptr[i], self.indptr[i + 1]):
                    cat = group[self.indices[k]]
                    if cat != -1:
                        if  cat_count[cat] >= nb_keep:
                            self.data[k] = 0
                        else:
                            cat_count[cat] += 1
            free(cat_count)

        self.sort_indices()
        self.eliminate_zeros()
        return 1

    cpdef KarmaSparse truncate_by_count_by_groups(self, group, ITYPE_t nb, axis=1):
        """
        Along given axis this will keep "nb" largest elements in each group.
        Other elements will be set to 0.

        Takes into account only nonzero elements.

        Parameters:
            group - list or numpy array of length self.shape[1]
            nb - integer
            axis - integer (0 or 1)

        Note: -1 as a group value  means non-existing group,
        and no truncation will be performed for this group.
        This is needed for compatibility with argmin/argmax method outputs.
        """
        cdef KarmaSparse res
        if self.aligned_axis(axis):
            res = self.copy()
        else:
            res = self.swap_slicing()
        res.aligned_truncate_by_count_by_group(group, nb)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)  # turn of bounds-checking for entire function
    cdef tuple global_argminmax(self, bool reverse, bool only_nonzero=False):
        cdef:
            ITYPE_t col, row, i
            LTYPE_t j
            DTYPE_t tampon

        with nogil:
            row = -1
            for i in xrange(self.nrows):
                if row == -1 and self.indptr[i + 1] > self.indptr[i]:
                    row = i
                    col = self.indices[self.indptr[i]]
                    tampon = self.data[self.indptr[i]]
                for j in xrange(self.indptr[i], self.indptr[i + 1]):
                    if (reverse and tampon < self.data[j]) \
                        or (not reverse and tampon > self.data[j]):
                        tampon = self.data[j]
                        row = i
                        col = self.indices[j]
            if not only_nonzero and \
                ((reverse and tampon < 0) or (not reverse and tampon > 0)):
                for i in xrange(self.nrows):
                    if self.indptr[i + 1] - self.indptr[i] >= self.ncols:
                        continue
                    row = i
                    col = 0
                    for j in xrange(self.indptr[i], self.indptr[i + 1]):
                        if self.indices[j] > col:
                            break
                        else:
                            col += 1
                    break
        return (row, col)

    @cython.wraparound(False)
    @cython.boundscheck(False)  # turn of bounds-checking for entire function
    cdef np.ndarray[dtype=ITYPE_t, ndim=1] aligned_argminmax(self, bool reverse,
                                                             bool only_nonzero=False):
        cdef:
            np.ndarray[ITYPE_t, ndim=1, mode="c"] out = - np.ones(self.nrows, dtype=ITYPE)
            np.ndarray[DTYPE_t, ndim=1, mode="c"] value = np.zeros(self.nrows, dtype=DTYPE)
            ITYPE_t i
            LTYPE_t j

        with nogil, parallel():
            for i in prange(self.nrows, schedule='static'):
                if self.indptr[i] < self.indptr[i + 1]:
                    value[i] = self.data[self.indptr[i]]
                    out[i] = self.indices[self.indptr[i]]
                    for j in xrange(self.indptr[i] + 1, self.indptr[i + 1]):
                        if (reverse and (value[i] < self.data[j])) or \
                           (not reverse and value[i] > self.data[j]):
                            value[i] = self.data[j]
                            out[i] = self.indices[j]
            if not only_nonzero:
                for i in prange(self.nrows, schedule='static'):
                    if (self.indptr[i + 1] - self.indptr[i] < self.ncols and \
                        ((reverse and value[i] < 0) or (not reverse and value[i] > 0))) \
                       or out[i] == -1:
                        out[i] = 0
                        for j in xrange(self.indptr[i], self.indptr[i + 1]):
                            if self.indices[j] > out[i]:
                                break
                            else:
                                out[i] += 1
        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)  # turn of bounds-checking for entire function
    cdef np.ndarray[dtype=ITYPE_t, ndim=1] misaligned_argminmax(self, bool reverse,
                                                                bool only_nonzero=False):
        cdef:
            np.ndarray[ITYPE_t, ndim=1, mode="c"] out = - np.ones(self.ncols, dtype=ITYPE)
            np.ndarray[DTYPE_t, ndim=1, mode="c"] value = np.zeros(self.ncols, dtype=DTYPE)
            np.ndarray[ITYPE_t, ndim=1, mode="c"] count
            ITYPE_t i, prev_j, axis_ind, k, number_to_fill, tot_count
            LTYPE_t j

        if not only_nonzero:
            count = np.zeros(self.ncols, dtype=ITYPE)
        with nogil:
            for i in xrange(self.nrows):
                for j in xrange(self.indptr[i], self.indptr[i + 1]):
                    axis_ind = self.indices[j]
                    if (out[axis_ind] == -1) or \
                       (reverse and value[axis_ind] < self.data[j]) or \
                       (not reverse and value[axis_ind] > self.data[j]):
                        value[axis_ind] = self.data[j]
                        out[axis_ind] = i
                    if not only_nonzero:
                        count[axis_ind] += 1
            if not only_nonzero:
                number_to_fill = 0
                for i in prange(self.ncols, schedule='static'):
                    if out[i] == -1:
                       count[i] = 0
                       out[i] = 0
                    elif count[i] < self.nrows and ((reverse and value[i] < 0) or
                                                   (not reverse and value[i] > 0)):
                        count[i] = 1
                        number_to_fill += 1
                    else:
                        count[i] = 0
                tot_count = 0
                for i in xrange(self.nrows):
                    if self.indptr[i + 1] == self.indptr[i]:
                        for j in xrange(self.ncols):
                            if count[j] == 1:
                                out[j] = i
                        break
                    else:
                        prev_j = 0
                        for j in xrange(self.indptr[i], self.indptr[i + 1]):
                            for k in xrange(prev_j, self.indices[j]):
                                if count[k] == 1:
                                    count[k] = 0
                                    tot_count += 1
                                    out[k] = i
                            prev_j = self.indices[j] + 1
                        for k in xrange(prev_j, self.ncols):
                                if count[k] == 1:
                                    count[k] = 0
                                    tot_count += 1
                                    out[k] = i
                        if tot_count >= number_to_fill:
                            break
        return out

    cdef tuple global_argmin(self, bool only_nonzero=False):
        res = self.global_argminmax(False, only_nonzero)
        if self.format == CSR:
            return res
        else:
            return pair_swap(res)

    cdef tuple global_argmax(self, bool only_nonzero=False):
        res = self.global_argminmax(True, only_nonzero)
        if self.format == CSR:
            return res
        else:
            return pair_swap(res)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] axis_argmax(self, axis,
                                                       bool only_nonzero=False):
        if self.aligned_axis(axis):
            return self.aligned_argminmax(True, only_nonzero)
        else:
            return self.misaligned_argminmax(True, only_nonzero)

    cdef np.ndarray[dtype=ITYPE_t, ndim=1] axis_argmin(self, axis,
                                                       bool only_nonzero=False):
        if self.aligned_axis(axis):
            return self.aligned_argminmax(False, only_nonzero)
        else:
            return self.misaligned_argminmax(False, only_nonzero)

    def argmin(self, axis=None, bool only_nonzero=False):
        if axis is None:
            return self.global_argmin(only_nonzero)
        else:
            return self.axis_argmin(axis, only_nonzero)

    def argmax(self, axis=None, bool only_nonzero=False):
        if axis is None:
            return self.global_argmax(only_nonzero)
        else:
            return self.axis_argmax(axis, only_nonzero)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] aligned_dense_dot(self, np.ndarray matrix):
        check_shape_comptibility(self.ncols, matrix.shape[0])
        cdef:
            ITYPE_t n_cols = matrix.shape[1]
            np.ndarray[DTYPE_t, ndim=2, mode="c"] mat = np.asarray(matrix, dtype=DTYPE, order='C')
            np.ndarray[DTYPE_t, ndim=2, mode="c"] out = np.zeros((self.nrows, n_cols), dtype=DTYPE, order='C')
            ITYPE_t i
            LTYPE_t j

        for i in prange(self.nrows, nogil=True, schedule='static'):
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                axpy(n_cols, self.data[j], &mat[self.indices[j], 0], &out[i, 0])
        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def linear_error(self, DTYPE_t[:,::1] matrix, DTYPE_t[::1] column, DTYPE_t[::1] row):
        if self.format != CSR:
            return self.tocsr().linear_error(matrix, column, row)

        check_shape_comptibility(self.ncols, matrix.shape[0])
        check_shape_comptibility(matrix.shape[1], column.shape[0])
        check_shape_comptibility(self.nrows, row.shape[0])

        cdef:
            ITYPE_t n_cols = matrix.shape[1]
            DTYPE_t[::1] out = np.zeros(n_cols, dtype=DTYPE)
            DTYPE_t[::1] tmp = column.copy()

        with nogil:
            _linear_error(self.nrows, n_cols,
                          &self.indptr[0], &self.indices[0], &self.data[0],
                          matrix, &tmp[0], &out[0],
                          &column[0], &row[0])

        return np.asarray(out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_sparse_agg(self, KarmaSparse other, binary_func fn=mmax):
        check_shape_comptibility(self.ncols, other.nrows)
        cdef:
            ITYPE_t nrows = self.nrows
            ITYPE_t ncols = other.ncols
            LTYPE_t jj, kk
            ITYPE_t row_nnz, j, i, k
            ITYPE_t ind, nn, head, tmp_head
            DTYPE_t ival
            ITYPE_t * mask
            DTYPE_t * ires
            ITYPE_t[::1] count = np.zeros(nrows, dtype=ITYPE)
            LTYPE_t[::1] indptr = np.zeros(nrows + 1, dtype=LTYPE)
            ITYPE_t[::1] indices
            DTYPE_t[::1] data
            double density

        with nogil, parallel():
            mask = <ITYPE_t *>malloc(ncols * sizeof(ITYPE_t))
            if mask == NULL:
                with gil: raise MemoryError()
            memset(mask, -1, ncols * sizeof(ITYPE_t))
            for i in prange(nrows, schedule='static'):
                row_nnz = 0
                for jj in xrange(self.indptr[i], self.indptr[i + 1]):
                    j = self.indices[jj]
                    for kk in xrange(other.indptr[j], other.indptr[j + 1]):
                        k = other.indices[kk]
                        if mask[k] != i:
                            mask[k] = i
                            row_nnz = row_nnz + 1
                count[i] = row_nnz
            free(mask)
        with nogil:
            for i in xrange(nrows):
                indptr[i + 1] = indptr[i] + count[i]

        data = np.zeros(indptr[nrows], dtype=DTYPE)
        indices = np.zeros(indptr[nrows], dtype=ITYPE)

        density = 1. * indptr[nrows] / max(ncols, 1) / max(nrows, 1)
        if density < 0.02:  # experimental constant: TODO improve it
            with nogil, parallel():
                ires = <DTYPE_t *>calloc(ncols, sizeof(DTYPE_t))
                mask = <ITYPE_t *>malloc(ncols * sizeof(ITYPE_t))
                if ires == NULL or mask == NULL:
                    with gil: raise MemoryError()
                memset(mask, -1, ncols * sizeof(ITYPE_t))
                for i in prange(nrows, schedule='guided'):
                    head = - 2
                    for kk in xrange(self.indptr[i], self.indptr[i + 1]):
                        ind = self.indices[kk]
                        ival = self.data[kk]
                        for jj in xrange(other.indptr[ind], other.indptr[ind + 1]):
                            j = other.indices[jj]
                            ires[j] = fn(ires[j], mult(other.data[jj], ival))
                            if mask[j] == -1:
                                mask[j] = head
                                head = j
                    nn = 0
                    for k in xrange(count[i]):
                        if ires[head] != 0:
                            indices[indptr[i] + nn] = head
                            data[indptr[i] + nn] = ires[head]
                            nn = nn + 1
                        tmp_head = head
                        head = mask[head]
                        mask[tmp_head] = -1
                        ires[tmp_head] = 0
                free(ires)
                free(mask)
        else:
            with nogil, parallel():
                ires = <DTYPE_t *>calloc(ncols, sizeof(DTYPE_t))
                if ires == NULL:
                    with gil: raise MemoryError()
                for i in prange(nrows, schedule='guided'):
                    for kk in xrange(self.indptr[i], self.indptr[i + 1]):
                        ind = self.indices[kk]
                        ival = self.data[kk]
                        for jj in xrange(other.indptr[ind], other.indptr[ind + 1]):
                            j = other.indices[jj]
                            ires[j] = fn(ires[j], mult(other.data[jj], ival))
                    nn = 0
                    if count[i] > 0:
                        for k in xrange(ncols):
                            if ires[k] != 0:
                                indices[indptr[i] + nn] = k
                                data[indptr[i] + nn] = ires[k]
                                ires[k] = 0
                                nn = nn + 1
                free(ires)
        res = KarmaSparse((np.asarray(data), np.asarray(indices), np.asarray(indptr)),
                          (nrows, ncols), self.format, copy=False)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[float, ndim=2] aligned_dense_agg(self, np.ndarray matrix, binary_func fn=mmax):
        check_shape_comptibility(self.ncols, matrix.shape[0])
        cdef:
            ITYPE_t n_cols = matrix.shape[1]
            np.ndarray[DTYPE_t, ndim=2, mode="c"] mat = np.asarray(matrix, dtype=DTYPE, order='C')
            np.ndarray[float, ndim=2, mode="c"] out = np.zeros((self.nrows, n_cols), dtype=np.float32, order='C')
            ITYPE_t i, k, ind
            LTYPE_t j
            DTYPE_t alpha

        for i in prange(self.nrows, nogil=True, schedule='static'):
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                alpha = self.data[j]
                ind = self.indices[j]
                for k in xrange(n_cols):
                    out[i, k] = fn(out[i, k], alpha * mat[ind, k])
        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[float, ndim=2] misaligned_dense_agg(self, np.ndarray matrix, binary_func fn=mmax):
        check_shape_comptibility(self.nrows, matrix.shape[0])
        cdef:
            ITYPE_t n_cols = matrix.shape[1]
            np.ndarray[DTYPE_t, ndim=2, mode="c"] mat = np.asarray(matrix, dtype=DTYPE, order='C')
            np.ndarray[float, ndim=2, mode="c"] out = np.zeros((self.ncols, n_cols), dtype=np.float32, order='C')
            ITYPE_t i, k
            LTYPE_t j, ind
            DTYPE_t alpha

        with nogil:
            for i in xrange(self.nrows):
                for j in xrange(self.indptr[i], self.indptr[i + 1]):
                    ind = self.indices[j]
                    alpha = self.data[j]
                    for k in xrange(n_cols):
                        out[ind, k] = fn(out[ind, k], alpha * mat[i, k])
        return out

    def dense_shadow(self, np.ndarray matrix, reducer="max"):
        supported_reducers = ['max', 'add']
        if reducer not in supported_reducers:
            raise ValueError('Unsupported reducer "{}", choose one from {}'.format(reducer,
                                                                                   ', '.join(supported_reducers)))
        cdef binary_func fn = get_reducer(<string?>reducer)

        if reducer == "max":
            self.check_positive()
            if np.any(matrix < 0):
                raise ValueError('Numpy matrix contains negative values while only positive are expected')

        if self.format == CSR:
            return self.aligned_dense_agg(matrix, fn)
        else:
            if matrix.shape[1] > 60:  # 60 is an experimentally found constant
                return self.swap_slicing().aligned_dense_agg(matrix, fn)
            else:
                return self.misaligned_dense_agg(matrix, fn)

    def sparse_shadow(self, KarmaSparse other, reducer="max"):
        supported_reducers = ['max', 'add']
        if reducer not in supported_reducers:
            raise ValueError('Unsupported reducer "{}", choose one from {}'.format(reducer,
                                                                                   ', '.join(supported_reducers)))
        cdef binary_func fn = get_reducer(<string?>reducer)

        if reducer == "max":
            self.check_positive()
            other.check_positive()

        if self.format == CSR and other.format == CSR:
            return self.aligned_sparse_agg(other, fn)
        elif self.format == CSR:
            return self.aligned_sparse_agg(other.tocsr(), fn)
        elif other.format == CSR:
            return self.tocsr().aligned_sparse_agg(other, fn)
        else:
            ### Q: is it correct in this case ?
            return other.transpose(copy=False)\
                        .aligned_sparse_agg(self.transpose(copy=False), fn)\
                        .transpose(copy=False)

    def shadow(self, other, reducer="max"):
        if isinstance(other, np.ndarray):
            return self.dense_shadow(other, reducer)
        elif is_karmasparse(other):
            return self.sparse_shadow(other, reducer)
        else:
            raise ValueError(other)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] misaligned_dense_dot(self, np.ndarray matrix):
        check_shape_comptibility(self.nrows, matrix.shape[0])
        cdef:
            ITYPE_t n_cols = matrix.shape[1]
            np.ndarray[DTYPE_t, ndim=2, mode="c"] mat = np.asarray(matrix, dtype=DTYPE, order='C')
            np.ndarray[DTYPE_t, ndim=2, mode="c"] out = np.zeros((self.ncols, n_cols), dtype=DTYPE, order='C')
            ITYPE_t i
            LTYPE_t j

        with nogil:
            for i in xrange(self.nrows):
                for j in xrange(self.indptr[i], self.indptr[i + 1]):
                    axpy(n_cols, self.data[j], &mat[i, 0], &out[self.indices[j], 0])
        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[DTYPE_t, ndim=1] aligned_dense_vector_dot(self, A[::1] vector):
        check_shape_comptibility(self.ncols, vector.shape[0])
        cdef DTYPE_t[::1] out = np.zeros(self.nrows, dtype=DTYPE, order='C')

        cdef int ti, n_th

        for ti in prange(1, nogil=True):
            n_th = omp_get_num_threads()  # workaround to detect threads number

        cdef int size = self.nrows / n_th, start, stop

        if self.nnz:
            with nogil, parallel(num_threads=n_th):
                ti = threadid()
                start = size * ti
                stop = self.nrows if ti + 1 == n_th else size * (ti + 1)
                _aligned_dense_vector_dot(start, stop, &self.indptr[0], &self.indices[0],
                                          &self.data[0], &vector[0], &out[0])

        return np.asarray(out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[DTYPE_t, ndim=1] misaligned_dense_vector_dot(self, A[::1] vector):
        check_shape_comptibility(self.nrows, vector.shape[0])

        cdef int ti, n_th

        for ti in prange(1, nogil=True):
            n_th = min(omp_get_num_threads(), 8)  # workaround to detect threads number

        cdef int size = self.nrows / n_th, start, stop
        cdef DTYPE_t[:, ::1] out_tmp = np.zeros((n_th, self.ncols), dtype=DTYPE)

        if self.nnz:
            with nogil, parallel(num_threads=n_th):
                ti = threadid()
                start = size * ti
                stop = self.nrows if ti + 1 == n_th else size * (ti + 1)
                _misaligned_dense_vector_dot(start, stop, &self.indptr[0], &self.indices[0],
                                             &self.data[0], &vector[0], &out_tmp[ti, 0])

        return np.asarray(out_tmp).sum(axis=0)

    def dense_vector_dot_right(self, A[::1] vector):
        if self.format == CSR:
            return self.aligned_dense_vector_dot(vector)
        else:
            return self.misaligned_dense_vector_dot(vector)

    def dense_vector_dot_left(self, A[::1] vector):
        if self.format == CSR:
            return self.misaligned_dense_vector_dot(vector)
        else:
            return self.aligned_dense_vector_dot(vector)

    def dense_dot_right(self, np.ndarray matrix):
        if self.format == CSR:
            return self.aligned_dense_dot(matrix)
        else:
            if matrix.shape[1] > 60:  # 60 is an experimentally found constant
                return self.swap_slicing().aligned_dense_dot(matrix)
            else:
                return self.misaligned_dense_dot(matrix)

    def dense_dot_left(self, np.ndarray matrix):
        if self.format == CSC:
            return self.aligned_dense_dot(matrix.transpose()).transpose()
        else:
            if matrix.shape[0] > 60:  # 60 is an experimentally found constant
                return self.swap_slicing().aligned_dense_dot(matrix.transpose()).transpose()
            else:
                return self.misaligned_dense_dot(matrix.transpose()).transpose()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse csr_mask_dense_dense_dot(self, np.ndarray a, np.ndarray b,
                                              binary_func op):
        check_shape_comptibility(a.shape[1], b.shape[0])
        check_shape_comptibility(a.shape[0], self.nrows)
        check_shape_comptibility(b.shape[1], self.ncols)

        cdef:
            ITYPE_t inner_dim = a.shape[1]
            ITYPE_t i, ii
            LTYPE_t j
            DTYPE_t mx
            DTYPE_t[:, ::1] aa = np.asarray(a, order='C', dtype=DTYPE)
            DTYPE_t[:, ::1] bb = np.asarray(b.transpose(), order='C', dtype=DTYPE)
            DTYPE_t[::1] data = np.zeros(self.get_nnz(), dtype=DTYPE)

        for i in prange(self.nrows, nogil=True, schedule='static'):
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                ii = self.indices[j]
                mx = scalar_product(inner_dim, &aa[i, 0], &bb[ii, 0])
                data[j] = op(self.data[j], mx)

        res = KarmaSparse((data,
                           np.array(self.indices, copy=True),
                           np.array(self.indptr, copy=True)),
                          self.shape, self.format, copy=False,
                          has_sorted_indices=1, has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse csr_mask_sparse_sparse_dot(self, KarmaSparse other_a, KarmaSparse other_b,
                                                binary_func op):
        check_shape_comptibility(self.nrows, other_a.shape[0])
        check_shape_comptibility(self.ncols, other_b.shape[1])
        check_shape_comptibility(other_a.shape[1], other_b.shape[0])

        if other_a.format != CSR:
            other_a = other_a.swap_slicing()
        if other_b.format != CSC:
            other_b = other_b.swap_slicing()

        cdef:
            DTYPE_t[::1] data = np.zeros(self.get_nnz(), dtype=DTYPE)
            ITYPE_t i, j, ind_a, ind_b
            LTYPE_t jj, stop_a, stop_b, pos_a, pos_b, start_a
            DTYPE_t mx, fac

        for i in prange(self.nrows, nogil=True, schedule='static'):
            for jj in xrange(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[jj]
                mx = 0
                pos_a = other_a.indptr[i]
                stop_a = other_a.indptr[i + 1]
                pos_b = other_b.indptr[j]
                stop_b = other_b.indptr[j + 1]
                while pos_a < stop_a and pos_b < stop_b:
                    ind_a = other_a.indices[pos_a]
                    ind_b = other_b.indices[pos_b]
                    if ind_a == ind_b:
                        mx = mx + other_a.data[pos_a] * other_b.data[pos_b]
                        pos_a = pos_a + 1
                        pos_b = pos_b + 1
                    elif ind_a < ind_b:
                        pos_a = pos_a + 1
                    else: # ind_b < ind_a:
                        pos_b = pos_b + 1
                data[jj] = op(self.data[jj], mx)
        res = KarmaSparse((data,
                          np.array(self.indices, copy=True),
                          np.array(self.indptr, copy=True)),
                          self.shape, self.format, copy=False,
                          has_sorted_indices=1,
                          has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse csr_mask_sparse_dense_dot(self, KarmaSparse other, np.ndarray b,
                                               binary_func op):
        check_shape_comptibility(other.shape[1], b.shape[0])
        check_shape_comptibility(other.shape[0], self.nrows)
        check_shape_comptibility(b.shape[1], self.ncols)

        if other.format != CSR:
            other = other.swap_slicing()
        cdef:
            ITYPE_t i, ii
            LTYPE_t j, k
            DTYPE_t mx
            DTYPE_t[::1, :] bb = np.asarray(b, order='F', dtype=DTYPE)
            DTYPE_t[::1] data = np.zeros(self.get_nnz(), dtype=DTYPE)

        for i in prange(self.nrows, nogil=True, schedule='static'):
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                ii = self.indices[j]
                mx = 0
                for k in xrange(other.indptr[i], other.indptr[i + 1]):
                    mx = mx + other.data[k] * bb[other.indices[k], ii]
                data[j] = op(self.data[j], mx)
        res = KarmaSparse((data,
                           np.array(self.indices, copy=True),
                           np.array(self.indptr, copy=True)),
                           self.shape, self.format, copy=False,
                           has_sorted_indices=1,
                           has_canonical_format=1)
        return res

    def mask_dot(self, a, b, string mask_mode="last"):
        cdef binary_func op = get_reducer(mask_mode)
        check_shape_comptibility(a.shape[1], b.shape[0])
        check_shape_comptibility(a.shape[0], self.shape[0])
        check_shape_comptibility(b.shape[1], self.shape[1])

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if self.format == CSR:
                return self.csr_mask_dense_dense_dot(a, b, op)
            else:
                return self.transpose(copy=False).csr_mask_dense_dense_dot(b.T, a.T, op).transpose(copy=False)
        elif is_karmasparse(a) and isinstance(b, np.ndarray):
            if self.format == CSR:
                return self.csr_mask_sparse_dense_dot(a, b, op)
            else:
                return self.swap_slicing().csr_mask_sparse_dense_dot(a, b, op).swap_slicing()
        elif isinstance(a, np.ndarray) and is_karmasparse(b):
            if self.format == CSC:
                return self.transpose(copy=False)\
                           .csr_mask_sparse_dense_dot(b.transpose(copy=False), a.T, op)\
                           .transpose(copy=False)
            else:
                return self.swap_slicing().transpose(copy=False)\
                           .csr_mask_sparse_dense_dot(b.transpose(copy=False), a.T, op)\
                           .transpose(copy=False).swap_slicing()
        elif is_karmasparse(a) and is_karmasparse(b):
            if self.format == CSR:
                return self.csr_mask_sparse_sparse_dot(a, b, op)
            else:
                return self.transpose(copy=False)\
                           .csr_mask_sparse_sparse_dot(b.transpose(copy=False),
                                                       a.transpose(copy=False), op)\
                           .transpose(copy=False)
        else:
            raise NotImplementedError('Unknown format')

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse generic_dense_restricted_binary_operation(self, floating[:,:] other,
                                                               binary_func fn):
        check_shape_comptibility(self.shape, np.asarray(other).shape)

        if self.format != CSR:
            other = np.asarray(other).transpose()

        cdef:
            LTYPE_t[::1] indptr = np.asarray(self.indptr).copy()
            ITYPE_t[::1] indices = np.asarray(self.indices).copy()
            DTYPE_t[::1] data = np.zeros(self.get_nnz(), dtype=DTYPE)
            ITYPE_t i
            LTYPE_t j

        for i in prange(self.nrows, nogil=True, schedule='static'):
            for j in xrange(self.indptr[i], self.indptr[i + 1]):
                data[j] = fn(self.data[j], other[i, self.indices[j]])

        cdef KarmaSparse res = KarmaSparse((data, indices, indptr),
                                           self.shape, self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        res.eliminate_zeros()
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] generic_dense_binary_operation(self, DTYPE_t[:,:] other, binary_func fn):
        check_shape_comptibility(self.shape, np.asarray(other).shape)

        cdef:
            DTYPE_t[:,::1] result = other.copy()
            ITYPE_t i, ind
            LTYPE_t j

        if self.format == CSR:
            for i in prange(self.nrows, nogil=True, schedule='static'):
                for j in xrange(self.indptr[i], self.indptr[i + 1]):
                    ind = self.indices[j]
                    result[i, ind] = fn(self.data[j], result[i, ind])
        else:
            for i in prange(self.nrows, nogil=True, schedule='static'):
                for j in xrange(self.indptr[i], self.indptr[i + 1]):
                    ind = self.indices[j]
                    result[ind, i] = fn(self.data[j], result[ind, i])

        return np.asarray(result)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef inline KarmaSparse generic_restricted_binary_operation(self, KarmaSparse other,
                                                                binary_func fn):
        check_shape_comptibility(self.shape, other.shape)
        if self.format != other.format:
            other = other.swap_slicing()
        cdef:
            LTYPE_t[::1] indptr = np.minimum(np.asarray(self.indptr), np.asarray(other.indptr))
            ITYPE_t[::1] indices = np.zeros(min(self.get_nnz(), other.get_nnz()), dtype=ITYPE)
            DTYPE_t[::1] data = np.zeros(min(self.get_nnz(), other.get_nnz()), dtype=DTYPE)
            ITYPE_t i, ind_a, ind_b
            LTYPE_t nn, stop_a, stop_b, pos_a, pos_b, start_a

        for i in prange(self.nrows, nogil=True, schedule='static'):
            pos_a = self.indptr[i]
            stop_a = self.indptr[i + 1]
            pos_b = other.indptr[i]
            stop_b = other.indptr[i + 1]
            nn = indptr[i]
            while pos_a < stop_a and pos_b < stop_b:
                ind_a = self.indices[pos_a]
                ind_b = other.indices[pos_b]
                if ind_a == ind_b:
                    data[nn] = fn(self.data[pos_a], other.data[pos_b])
                    indices[nn] = ind_a
                    nn = nn + 1
                    pos_a = pos_a + 1
                    pos_b = pos_b + 1
                elif ind_a < ind_b:
                    pos_a = pos_a + 1
                else: # ind_b < ind_a:
                    pos_b = pos_b + 1
        cdef KarmaSparse res = KarmaSparse((data, indices, indptr),
                                           self.shape, self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        res.eliminate_zeros()
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef inline KarmaSparse generic_binary_operation(self, KarmaSparse other, binary_func fn):
        check_shape_comptibility(self.shape, other.shape)
        if self.format != other.format:
            other = other.swap_slicing()

        cdef:
            LTYPE_t[::1] indptr = np.asarray(self.indptr) + np.asarray(other.indptr)
            ITYPE_t[::1] indices = np.zeros(self.get_nnz() + other.get_nnz(), dtype=ITYPE)
            DTYPE_t[::1] data = np.zeros(self.get_nnz() + other.get_nnz(), dtype=DTYPE)
            ITYPE_t i, ind_a, ind_b
            LTYPE_t nn, stop_a, stop_b, pos_a, pos_b, start_a
            DTYPE_t x

        for i in prange(self.nrows, nogil=True, schedule='static'):
            pos_a = self.indptr[i]
            stop_a = self.indptr[i + 1]
            pos_b = other.indptr[i]
            stop_b = other.indptr[i + 1]
            nn = indptr[i]
            while pos_a < stop_a and pos_b < stop_b:
                ind_a = self.indices[pos_a]
                ind_b = other.indices[pos_b]
                if ind_a == ind_b:
                    x = fn(self.data[pos_a], other.data[pos_b])
                    if x != 0.:
                        data[nn] = x
                        indices[nn] = ind_a
                        nn = nn + 1
                    pos_a = pos_a + 1
                    pos_b = pos_b + 1
                elif ind_a < ind_b:
                    x = fn(self.data[pos_a], 0.)
                    if x != 0.:
                        data[nn] = x
                        indices[nn] = ind_a
                        nn = nn + 1
                    pos_a = pos_a + 1
                else: # ind_b < ind_a:
                    x = fn(0., other.data[pos_b])  # we should respect order of argument
                    if x != 0.:
                        data[nn] = x
                        indices[nn] = ind_b
                        nn = nn + 1
                    pos_b = pos_b + 1
            while pos_a < stop_a:
                x = fn(self.data[pos_a], 0.)
                if x != 0.:
                    data[nn] = x
                    indices[nn] = self.indices[pos_a]
                    nn = nn + 1
                pos_a = pos_a + 1
            while pos_b < stop_b:
                x = fn(0., other.data[pos_b])
                if x != 0.:
                    data[nn] = x
                    indices[nn] = other.indices[pos_b]
                    nn = nn + 1
                pos_b = pos_b + 1
        cdef KarmaSparse res = KarmaSparse((data, indices, indptr),
                                           self.shape, self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        res.eliminate_zeros()
        return res

    cpdef KarmaSparse complement(self, other):
        """
        Returns a copy of the KarmaSparse where all locations from
        other.nonzero() are set to zero.

        :param other:  * either matrix (KarmaSparse, np.ndarray)
                       * either tuple of indices (row, col) to exclude

        """
        cdef KarmaSparse other_ks = KarmaSparse(other, shape=self.shape,
                                                format=self.format, copy=False)
        check_shape_comptibility(self.shape, other_ks.shape)
        return self.generic_binary_operation(other_ks, get_reducer('complement'))

    cpdef KarmaSparse maximum(self, KarmaSparse other):
        return self.generic_binary_operation(other, get_reducer('max'))

    cpdef KarmaSparse minimum(self, KarmaSparse other):
        return self.generic_binary_operation(other, get_reducer('min'))

    cdef KarmaSparse multiply(self, other):
        cdef binary_func fn = get_reducer('multiply')
        if is_karmasparse(other):
            return self.generic_restricted_binary_operation(other, fn)
        elif isinstance(other, np.ndarray):
            if other.dtype == np.float32:
                return self.generic_dense_restricted_binary_operation[float](other, fn)
            else:
                return self.generic_dense_restricted_binary_operation[double](other.astype(np.float, copy=False),
                                                                              fn)
        else:
            raise ValueError(other)

    cdef KarmaSparse divide(self, other):
        cdef binary_func fn = get_reducer('divide')

        if is_karmasparse(other):
            return self.generic_restricted_binary_operation(other, fn)
        elif isinstance(other, np.ndarray):
            if other.dtype == np.float32:
                return self.generic_dense_restricted_binary_operation[float](other, fn)
            else:
                return self.generic_dense_restricted_binary_operation[double](other.astype(np.float, copy=False),
                                                                              fn)
        else:
            raise ValueError(other)

    def __add__(self, other):
        if is_karmasparse(other) and is_karmasparse(self):
            return (<KarmaSparse?>self).generic_binary_operation(other, get_reducer('add'))
        elif is_karmasparse(self) and isinstance(other, np.ndarray):
            if other.ndim == 0:
                return (<KarmaSparse?>self).scalar_add(float(other))
            if other.ndim == 1:
                check_shape_comptibility(self.shape[1], other.shape[0])
                other = np.repeat(np.atleast_2d(other), self.shape[0], axis=0)
            if other.ndim == 2:
                if other.shape != self.shape:
                    if other.shape == (self.shape[0], 1):
                        other = np.repeat(other, self.shape[1], axis=1)
                    if other.shape == (1, self.shape[1]):
                        other = np.repeat(other, self.shape[0], axis=0)
                return (<KarmaSparse?>self).generic_dense_binary_operation(other.astype(np.float, copy=False),
                                                                           get_reducer('add'))
            else:
                raise ValueError('operands could not be broadcast together with shapes {} {}'
                                 .format(self.shape, other.shape))

        elif is_karmasparse(self) and np.isscalar(other):
            return (<KarmaSparse?>self).scalar_add(other)  # XXX: it will make an addition only on nonzeros cells
        elif is_karmasparse(other) and np.isscalar(self):
            return (<KarmaSparse?>other).scalar_add(self)
        else:
            raise NotImplementedError()

    __iadd__ = __add__

    def __mul__(self, other):
        if np.isscalar(other) and is_karmasparse(self):
            return (<KarmaSparse?>self).scalar_multiply(other)
        elif np.isscalar(self) and is_karmasparse(other):
            return (<KarmaSparse?>other).scalar_multiply(self)
        elif is_karmasparse(other) and is_karmasparse(self):
            if self.shape == other.shape:
                return (<KarmaSparse?>self).multiply(other)
            elif self.shape[0] == other.shape[0] and other.shape[1] == 1:
                return (<KarmaSparse?>self).scale_along_axis(other.toarray()[:, 0], axis=1)
            elif self.shape[1] == other.shape[1] and other.shape[0] == 1:
                return (<KarmaSparse?>self).scale_along_axis(other.toarray()[0, :], axis=0)
            elif self.shape[0] == other.shape[0] and self.shape[1] == 1:
                return (<KarmaSparse?>other).scale_along_axis(self.toarray()[:, 0], axis=1)
            elif self.shape[1] == other.shape[1] and self.shape[0] == 1:
                return (<KarmaSparse?>other).scale_along_axis(self.toarray()[0, :], axis=0)
            else:
                raise ValueError('operands could not be broadcast together with shapes {} {}'
                                 .format(self.shape, other.shape))
        elif is_karmasparse(self) and isinstance(other, np.ndarray):
            if other.ndim == 0:
                return (<KarmaSparse?>self).scalar_multiply(float(other))
            if other.ndim == 1:
                if other.shape[0] == self.shape[1]:
                    return (<KarmaSparse?>self).scale_along_axis(other, axis=0)
                else:
                    raise ValueError('operands could not be broadcast together with shapes {} {}'
                                     .format(self.shape, other.shape))
            if other.ndim == 2:
                if self.shape == other.shape:
                    return (<KarmaSparse?>self).multiply(other)
                elif self.shape[0] == other.shape[0] and other.shape[1] == 1:
                    return (<KarmaSparse?>self).scale_along_axis(other[:, 0], axis=1)
                elif self.shape[1] == other.shape[1] and other.shape[0] == 1:
                    return (<KarmaSparse?>self).scale_along_axis(other[0, :], axis=0)
                elif self.shape[0] == other.shape[0] and self.shape[1] == 1:
                    return KarmaSparse(other, format=self.format).scale_along_axis(self.toarray()[:, 0], axis=1)
                elif self.shape[1] == other.shape[1] and self.shape[0] == 1:
                    return KarmaSparse(other, format=self.format).scale_along_axis(self.toarray()[0, :], axis=0)
                else:
                    raise ValueError('operands could not be broadcast together with shapes {} {}'
                                     .format(self.shape, other.shape))
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __div__(self, other):
        if np.isscalar(other) and is_karmasparse(self):
            return (<KarmaSparse?>self).scalar_divide(other)
        elif np.isscalar(self) and is_karmasparse(other):
            return (<KarmaSparse?>other).power(-1).scalar_multiply(self)
        elif is_karmasparse(other) and is_karmasparse(self):
            return (<KarmaSparse?>self).divide(other)
        elif is_karmasparse(self) and isinstance(other, np.ndarray):
            factor = 1. / other
            return self.__mul__(factor)
        else:
            raise NotImplementedError()

    def __idiv__(self, other):
        return self.__div__(other)

    def __abs__(self):
        return self.abs()

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self + (- other)

    __isub__ = __sub__

    def __pow__(self, p, _):
        if is_karmasparse(self):
            return self.power(p)
        else:
            raise NotImplementedError()

    __ipow__ = __pow__

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef DTYPE_t aligned_get_single_element(self, ITYPE_t row, ITYPE_t col) nogil:
        """
        Take element M[row, col] for "CSR" format without bounds check
        """
        cdef DTYPE_t mx = 0
        cdef LTYPE_t j
        for j in xrange(self.indptr[row], self.indptr[row + 1]):
            if self.indices[j] == col:
                mx += self.data[j]
        return mx

    cdef DTYPE_t get_single_element(self, ITYPE_t row, ITYPE_t col) except? -1:
        """
        Take element M[row, col]
        """
        if self.format != CSR:
            row, col = col, row
        if row < 0:
            row += self.nrows
        if col < 0:
            col += self.ncols
        check_bounds(row, self.nrows)
        check_bounds(col, self.ncols)
        return self.aligned_get_single_element(row, col)

    cdef bool check_arrays(self, np.ndarray rows, np.ndarray cols) except 0:
        assert rows.ndim == 1
        assert cols.ndim == 1
        check_shape_comptibility(rows.shape[0], cols.shape[0])
        minx, miny = np.min(rows), np.min(cols)
        if minx < 0:
            rows[rows < 0] += self.shape[0]
            minx += self.shape[0]
        if miny < 0:
            cols[cols < 0] += self.shape[1]
            miny += self.shape[1]
        maxx, maxy = np.max(rows), np.max(cols)
        check_bounds(minx, self.shape[0])
        check_bounds(maxx, self.shape[0])
        check_bounds(miny, self.shape[1])
        check_bounds(maxy, self.shape[1])
        return 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.ndarray[DTYPE_t, ndim=1] sample_values(self, row_list, col_list):
        if len(row_list) == 0:
            return np.array([], dtype=ITYPE)
        rows = np.asarray(row_list, dtype=ITYPE)
        cols = np.asarray(col_list, dtype=ITYPE)

        self.check_arrays(rows, cols)

        cdef:
            ITYPE_t[::1] view_rows, view_cols
            LTYPE_t i, size = len(rows)
            DTYPE_t[::1] result = np.zeros(size, dtype=DTYPE)
        if self.format == CSR:
            view_rows, view_cols = rows, cols
        else:
            view_rows, view_cols = cols, rows
        # not optimize way of doing things
        for i in prange(size, nogil=True):
            result[i] = self.aligned_get_single_element(view_rows[i], view_cols[i])
        # TODO: for large values of size we should use another strategy
        return np.asarray(result)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_get_submatrix(self, ITYPE_t row0, ITYPE_t row1,
                                           ITYPE_t col0, ITYPE_t col1):
        """
        This corresponds to submatrix M[row0:row1, col0:col1]
        for CSR format without bounds check
        """
        cdef:
            ITYPE_t nrows = row1 - row0
            LTYPE_t[::1] indptr = np.zeros(nrows + 1, dtype=LTYPE)
            ITYPE_t[::1] indices
            DTYPE_t[::1] data
            ITYPE_t i, j, k
            LTYPE_t jj
            Shape_t shape

        for i in prange(nrows, nogil=True, schedule="static"):
            for jj in xrange(self.indptr[i + row0], self.indptr[i + row0  + 1]):
                j = self.indices[jj]
                if col0 <= j < col1:
                    indptr[i + 1] = indptr[i + 1] + 1
        for i in xrange(nrows):
            indptr[i + 1] += indptr[i]

        indices = np.zeros(indptr[nrows], dtype=ITYPE)
        data = np.zeros(indptr[nrows], dtype=DTYPE)

        for i in prange(nrows, nogil=True, schedule="static"):
            k = 0
            for jj in xrange(self.indptr[i + row0], self.indptr[i + row0  + 1]):
                j = self.indices[jj]
                if col0 <= j < col1:
                    indices[indptr[i] + k] = j - col0
                    data[indptr[i] + k] = self.data[jj]
                    k = k + 1
        if self.format == CSR:
           shape = (nrows, col1 - col0)
        else:
           shape = (col1 - col0, nrows)
        cdef KarmaSparse res = KarmaSparse((data, indices, indptr), shape=shape,
                                           format=self.format, copy=False,
                                           has_sorted_indices=1,
                                           has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse aligned_subinterval(self, ITYPE_t row0, ITYPE_t row1):
        """
        This corresponds to submatrix M[row0:row1] for CSR without bounds check
        """
        cdef:
            ITYPE_t nrows = row1 - row0
            LTYPE_t[::1] indptr = np.zeros(nrows + 1, dtype=LTYPE)
            ITYPE_t[::1] indices
            DTYPE_t[::1] data
            ITYPE_t i, size
            Shape_t sparse

        for i in xrange(nrows):
            indptr[i + 1] = indptr[i] + self.indptr[i + row0 + 1] - self.indptr[i + row0]

        indices = np.zeros(indptr[nrows], dtype=ITYPE)
        data = np.zeros(indptr[nrows], dtype=DTYPE)

        for i in prange(nrows, nogil=True, schedule="static"):
            size = indptr[i + 1] - indptr[i]
            memcpy(&indices[indptr[i]], &self.indices[self.indptr[i + row0]], size * sizeof(ITYPE_t))
            memcpy(&data[indptr[i]], &self.data[self.indptr[i + row0]], size * sizeof(DTYPE_t))

        shape = (nrows, self.ncols) if self.format == CSR else (self.ncols, nrows)
        cdef KarmaSparse res = KarmaSparse((data, indices, indptr),
                                           shape=shape, format=self.format, copy=False,
                                           has_sorted_indices=1, has_canonical_format=1)
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef KarmaSparse extractor(self, my_indices, axis):
        """
        produces a KS matrix M such that
        M[i, my_indices[i]] = 1 for all i in xrange(len(my_indices))
                            = 0 otherwise
        """
        cdef ITYPE_t dim, i, row, size
        cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] indices = np.ascontiguousarray(my_indices, dtype=ITYPE)
        cdef ITYPE_t[::1] new_indices
        cdef ITYPE_t length = indices.shape[0]
        cdef LTYPE_t[::1] indptr
        cdef DTYPE_t[::1] data

        dim = self.nrows if self.aligned_axis(axis) else self.ncols
        if length > 0:
            mi = np.min(indices)
            if mi < 0:
                indices[indices < 0] += dim
                mi += dim
            check_bounds(mi, dim)
            ma = np.max(indices)
            check_bounds(ma, dim)

        if self.aligned_axis(axis):
            indptr = np.zeros(length + 1, dtype=LTYPE)
            with nogil:
                for i in xrange(length):
                    row = indices[i]
                    indptr[i + 1] = indptr[i] + self.indptr[row + 1] - self.indptr[row]
            new_indices = np.zeros(indptr[length], dtype=ITYPE)
            data = np.zeros(indptr[length], dtype=DTYPE)
            for i in prange(length, nogil=True, schedule='static'):
                row = indices[i]
                size = self.indptr[row + 1] - self.indptr[row]
                memcpy(&data[indptr[i]], &self.data[self.indptr[row]], size * sizeof(DTYPE_t))
                memcpy(&new_indices[indptr[i]], &self.indices[self.indptr[row]], size * sizeof(ITYPE_t))
            shape = (length, self.ncols) if axis == 1 else (self.ncols, length)
            res = KarmaSparse((data, new_indices, indptr),
                              shape=shape, format=self.format, copy=False,
                              has_sorted_indices=1, has_canonical_format=1)
            return res
        else: # non-aligned axis
            data = np.ones(length, dtype=DTYPE)
            indptr = np.arange(length + 1, dtype=LTYPE)
            extractor = KarmaSparse((data, indices, indptr),
                                    shape=(length, dim),
                                    format=CSR, copy=False,
                                    has_sorted_indices=1,
                                    has_canonical_format=1).tocsc()
            if axis == 1:
                return extractor.dot(self)
            else:
                return self.dot(extractor.transpose(copy=False))

    cdef KarmaSparse get_submatrix(self, ITYPE_t row0, ITYPE_t row1,
                                   ITYPE_t col0, ITYPE_t col1):
        """
        This corresponds to submatrix M[row0:row1, col0:col1]
        """
        if self.format != CSR:
            row0, row1, col0, col1 = col0, col1, row0, row1
        if row0 < 0:
            row0 += self.nrows
        if row1 < 0:
            row1 += self.nrows
        check_bounds(row0, self.nrows + 1)
        check_bounds(row1, self.nrows + 1)
        check_ordered(row0, row1, strict=False)

        if col0 < 0:
            col0 += self.ncols
        if col1 < 0:
            col1 += self.ncols
        check_bounds(col0, self.ncols + 1)
        check_bounds(col1, self.ncols + 1)
        check_ordered(col0, col1, strict=False)

        if row0 == 0 and row1 == self.nrows and col0 == 0 and col1 == self.ncols:
            return self.copy()
        elif col0 == 0 and col1 == self.ncols:
            return self.aligned_subinterval(row0, row1)
        else:
            return self.aligned_get_submatrix(row0, row1, col0, col1)

    cdef KarmaSparse get_row_slice(self, slice sl):
        """
        Takes M[sl,:] for a slice sl
        """
        if sl == slice(None, None, None):
            return self
        start, stop, stride = sl.indices(self.shape[0])
        if stride == 1: # a[1:2] a[1:2:1] or a[:]
            return self.get_submatrix(start, stop, 0, self.shape[1])
        else: # a[1:6:2] or a[:-4]
            my_indices = np.arange(start, stop, stride, dtype=ITYPE)
            return self.extractor(my_indices, axis=1)

    cdef KarmaSparse get_column_slice(self, slice sl):
        """
        Takes M[:, sl] for a slice sl
        """
        if sl == slice(None, None, None):
            return self
        start, stop, stride = sl.indices(self.shape[1])
        if stride == 1: # a[1:2] a[1:2:1] or a[:]
            return self.get_submatrix(0, self.shape[0], start, stop)
        else: # a[1:6:2] or a[:-4]
            my_indices = np.arange(start, stop, stride, dtype=ITYPE)
            return self.extractor(my_indices, axis=0)

    cdef KarmaSparse get_row(self, ITYPE_t row):
        if row < 0:
            row += self.shape[0]
        check_bounds(row, self.shape[0])
        return self.get_submatrix(row, row + 1, 0, self.shape[1])

    cdef KarmaSparse get_column(self, ITYPE_t col):
        if col < 0:
            col += self.shape[1]
        check_bounds(col, self.shape[1])
        return self.get_submatrix(0, self.shape[0], col, col + 1)

    cdef KarmaSparse restrict_along_row(self, key):
        if is_int(key): # a[i]
            return self.get_row(key)
        if isinstance(key, slice):  # a[1:4:2]
            return self.get_row_slice(key)
        if isinstance(key, (np.ndarray, list)): # a[[1,2,-2]]
            return self.extractor(key, axis=1)
        raise NotImplementedError(key)

    cdef KarmaSparse restrict_along_column(self, key):
        if is_int(key):
            return self.get_column(key)
        if isinstance(key, slice):
            return self.get_column_slice(key)
        if isinstance(key, (np.ndarray, list)):
            return self.extractor(key, axis=0)
        raise NotImplementedError(key)

    def __getitem__(self, key):
        if not isinstance(key, tuple): # a[?]
            return self.restrict_along_row(key)
        if len(key) > 2:
            raise IndexError("KarmaSparse cannot be spliced by more than two axes")
        # so key is tuple a[row, col]
        row, col = key
        if is_int(row) and is_int(col): # a[i,j]
            return self.get_single_element(row, col)
        elif isinstance(row, (np.ndarray, list)) and isinstance(col, (np.ndarray, list)):
            return self.sample_values(row, col)
        elif isinstance(row, slice) and isinstance(col, slice) and \
            (row.step is None or row.step == 1) and \
            (col.step is None or col.step == 1):
            return self.get_submatrix(row.start or 0, row.stop or self.shape[0],
                                      col.start or 0, col.stop or self.shape[1])
        elif isinstance(row, slice) and row == slice(None, None, None): # a[:, ?]
            return self.restrict_along_column(col)
        elif isinstance(col, slice) and col == slice(None, None, None): # a[?, :]
            return self.restrict_along_row(row)
        else:
            if self.format == CSR:
                return self.restrict_along_row(row).restrict_along_column(col)
            else:
                return self.restrict_along_column(col).restrict_along_row(row)

    def kronii_dot(self, matrix, factor, power=1):
        if is_karmasparse(matrix):
            return self.kronii_sparse_dot(matrix, factor, power)
        else:
            return self.kronii_dense_dot(np.ascontiguousarray(matrix), factor, power)

    def kronii_dense_dot(self, floating[:,::1] matrix, np.ndarray factor, double power=1.):
        check_shape_comptibility(self.shape[0], matrix.shape[0])
        check_shape_comptibility(factor.shape[0], self.shape[1] * matrix.shape[1])

        if self.format == 'csc':
            return self.swap_slicing().kronii_dense_dot(matrix, factor, power)

        cdef DTYPE_t[::1] result = np.zeros(self.nrows, dtype=np.float64)
        cdef DTYPE_t[::1] _factor = np.asarray(factor, dtype=DTYPE, order="C")

        if self.nnz > 0:
            if np.asarray(matrix).dtype == np.float32:
                kronii_dot[float](self.nrows, matrix.shape[1], &self.indptr[0], &self.indices[0],
                                  &self.data[0], <float*>&matrix[0, 0], &_factor[0], &result[0], power)
            else:
                kronii_dot[double](self.nrows, matrix.shape[1], &self.indptr[0], &self.indices[0],
                                   &self.data[0], <double*>&matrix[0, 0], &_factor[0], &result[0], power)
        return np.asarray(result)

    def kronii_sparse_dot(self, KarmaSparse other, np.ndarray factor, double power=1.):
        check_shape_comptibility(self.shape[0], other.shape[0])
        check_shape_comptibility(factor.shape[0], self.shape[1] * other.shape[1])

        if self.format == 'csc':
            return self.swap_slicing().kronii_sparse_dot(other, factor, power)

        if other.format == 'csc':
            return self.kronii_sparse_dot(other.swap_slicing(), factor, power)

        cdef DTYPE_t[::1] result = np.zeros(self.nrows, dtype=DTYPE)
        cdef DTYPE_t[::1] _factor = np.asarray(factor, dtype=DTYPE, order="C")
        if self.nnz > 0 and other.nnz > 0:
            kronii_sparse_dot(self.nrows, other.ncols,
                              &self.indptr[0], &self.indices[0], &self.data[0],
                              &other.indptr[0], &other.indices[0], &other.data[0],
                              &_factor[0], &result[0], power)

        return np.asarray(result)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def kronii_dense_dot_transpose(self, floating[:,::1] matrix, np.ndarray factor, double power=1.):
        check_shape_comptibility(self.shape[0], matrix.shape[0])
        check_shape_comptibility(factor.shape[0], self.shape[0])

        if self.format == 'csc':
            return self.swap_slicing().kronii_dense_dot_transpose(matrix, factor, power)

        cdef DTYPE_t[::1] _factor = np.asarray(factor, dtype=DTYPE, order="C")
        cdef int ti, n_th

        for ti in prange(1, nogil=True):
            n_th = min(omp_get_num_threads(), 8)  # workaround to detect threads number

        cdef int batch_size = self.nrows / n_th, start, stop
        cdef DTYPE_t[:, ::1] out_tmp = np.zeros((n_th, self.shape[1] * matrix.shape[1]), dtype=DTYPE)

        if self.nnz > 0:
            if np.asarray(matrix).dtype == np.float32:
                with nogil, parallel(num_threads=n_th):
                    ti = threadid()
                    start = batch_size * ti
                    stop = self.nrows if ti + 1 == n_th else batch_size * (ti + 1)
                    kronii_dot_transpose[float](start, stop, matrix.shape[1],
                                                &self.indptr[0], &self.indices[0], &self.data[0],
                                                <float*>&matrix[0, 0],
                                                &_factor[0], &out_tmp[ti, 0], power)
            else:
                with nogil, parallel(num_threads=n_th):
                    ti = threadid()
                    start = batch_size * ti
                    stop = self.nrows if ti + 1 == n_th else batch_size * (ti + 1)
                    kronii_dot_transpose[double](start, stop, matrix.shape[1],
                                                &self.indptr[0], &self.indices[0], &self.data[0],
                                                <double*>&matrix[0, 0],
                                                &_factor[0], &out_tmp[ti, 0], power)
        return np.asarray(out_tmp).sum(axis=0)


    @cython.wraparound(False)
    @cython.boundscheck(False)
    def kronii_sparse_dot_transpose(self, KarmaSparse other, np.ndarray factor, double power=1.):
        check_shape_comptibility(self.shape[0], other.shape[0])
        check_shape_comptibility(factor.shape[0], self.shape[0])

        if self.format == 'csc':
            return self.swap_slicing().kronii_sparse_dot_transpose(other, factor, power)

        if other.format == 'csc':
            return self.kronii_sparse_dot_transpose(other.swap_slicing(), factor, power)

        cdef DTYPE_t[::1] _factor = np.asarray(factor, dtype=DTYPE, order="C")
        cdef int ti, n_th

        for ti in prange(1, nogil=True):
            n_th = min(omp_get_num_threads(), 8)  # workaround to detect threads number

        cdef int batch_size = self.nrows / n_th, start, stop
        cdef DTYPE_t[:, ::1] out_tmp = np.zeros((n_th, self.shape[1] * other.shape[1]), dtype=DTYPE)

        if self.nnz > 0 and other.nnz > 0:
            with nogil, parallel(num_threads=n_th):
                ti = threadid()
                start = batch_size * ti
                stop = self.nrows if ti + 1 == n_th else batch_size * (ti + 1)
                kronii_sparse_dot_transpose(start, stop, other.ncols,
                                            &self.indptr[0], &self.indices[0], &self.data[0],
                                            &other.indptr[0], &other.indices[0], &other.data[0],
                                            &_factor[0], &out_tmp[ti, 0], power)

        return np.asarray(out_tmp).sum(axis=0)

    def kronii_dot_transpose(self, matrix, factor, power=1):
        if is_karmasparse(matrix):
            return self.kronii_sparse_dot_transpose(matrix, factor, power)
        else:
            return self.kronii_dense_dot_transpose(np.ascontiguousarray(matrix), factor, power)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def kronii_second_moment_sparse(self, KarmaSparse matrix):
        # temporal fallback
        check_shape_comptibility(self.shape[0], matrix.shape[0])
        c = self.kronii(matrix)
        return c.T.dot(c)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def kronii_second_moment_dense(self, floating[:,::1] matrix):
        check_shape_comptibility(self.shape[0], matrix.shape[0])

        cdef int size1 = self.shape[1], size2 = matrix.shape[1], dd = size1 * size2
        cdef floating[:,::1] res = np.zeros((dd, dd), dtype=np.asarray(matrix).dtype)
        cdef floating * tmp
        cdef floating val
        cdef KarmaSparse other
        cdef int i, x1, x2, xx1, yy1, y1, y2, ind1_x, ind1_y, ind2_x, ind2_y, ind3_x, ind3_y, ind4_x, ind4_y
        cdef LTYPE_t pos_a, stop_a, pos_b, stop_b, ind_a, ind_b
        cdef long[::1] x, y, order
        cdef cy_syr_type cy_syr

        if floating is float:
            cy_syr = cy_ssyr
        else:
            cy_syr = cy_dsyr

        if self.aligned_axis(0):
            other = self
        else:
            other = self.swap_slicing()

        order = np.argsort(other.count_nonzero(axis=0))
        x, y = map(np.ascontiguousarray, np.triu_indices(size1, 0))

        # parallel over whole upper triangle
        # order is important to better allocate threads
        for i in prange(x.shape[0], nogil=True, schedule='dynamic'):
            x1, y1 = order[x[i]], order[y[i]]
            xx1, yy1 = x1 * size2, y1 * size2
            tmp = &res[xx1, yy1]

            pos_a, stop_a = other.indptr[x1], other.indptr[x1 + 1]
            pos_b, stop_b = other.indptr[y1], other.indptr[y1 + 1]

            while pos_a < stop_a and pos_b < stop_b:
                ind_a, ind_b = other.indices[pos_a], other.indices[pos_b]
                if ind_a == ind_b:
                    cy_syr(size2, other.data[pos_a] * other.data[pos_b], &matrix[ind_b, 0], tmp, dd)
                    pos_a = pos_a + 1
                    pos_b = pos_b + 1
                elif ind_a < ind_b:
                    pos_a = pos_a + 1
                else: # ind_b < ind_a:
                    pos_b = pos_b + 1

            # symmetrising inplace
            for x2 in range(size2):
                for y2 in range(size2):
                    ind1_x, ind1_y = xx1 + x2, yy1 + y2
                    ind2_x, ind2_y = yy1 + x2, xx1 + y2
                    ind3_x, ind3_y = ind2_y, ind2_x
                    ind4_x, ind4_y = ind1_y, ind1_x

                    if res[ind1_x, ind1_y] != 0:
                        val = res[ind1_x, ind1_y]
                    elif res[ind2_x, ind2_y] != 0:
                        val = res[ind2_x, ind2_y]
                    elif res[ind3_x, ind3_y] != 0:
                        val = res[ind2_x, ind2_y]
                    else:
                        val = res[ind4_x, ind4_y]

                    if val != 0:
                        res[ind1_x, ind1_y] = res[ind2_x, ind2_y] = res[ind2_x, ind2_y] = res[ind4_x, ind4_y] = val

        return np.asarray(res)

    def kronii_second_moment(self, matrix):
        """
        returns symmetric matrix x.T.dot(x), where x = kronii(self, matrix)
        """
        if is_karmasparse(matrix):
            return self.kronii_second_moment_sparse(matrix)
        else:
            # we need to prevent blas from internal multi-threading use `with blas_threads(1):`
            return self.kronii_second_moment_dense(np.ascontiguousarray(matrix))

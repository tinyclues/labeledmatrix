#cython: embedsignature=True
#cython: nonecheck=True
#cython: overflowcheck=True
#cython: unraisable_tracebacks=True

cimport cython
from libc.math cimport tanh
from cyperf.tools.sort_tools cimport partial_sort
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

from cython.parallel import prange

REDUCERLIST['max'] = mmax
REDUCERLIST['min'] = mmin
REDUCERLIST['add'] = cadd
REDUCERLIST['multiply'] = mult
REDUCERLIST['divide'] = save_division
REDUCERLIST['subtract'] = cminus
REDUCERLIST['first'] = first
REDUCERLIST['last'] = last
REDUCERLIST['complement'] = complement
REDUCERLIST['truncated_difference'] = trunc_diff


ctypedef fused ITERbis:
    list
    tuple
    np.ndarray[dtype=int, ndim=1]
    np.ndarray[dtype=long, ndim=1]
    np.ndarray[dtype=float, ndim=1]
    np.ndarray[dtype=double, ndim=1]
    np.ndarray[dtype=object, ndim=1]
    object


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[ndim=1, dtype=int] bisect_left(ITER a, ITERbis x):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.
    """
    cdef long lo, hi, mid, nb = len(x), i
    cdef int[::1] out = np.zeros(nb, dtype=np.int32)

    for i in xrange(nb):
        o = x[i]
        lo = 0
        hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < o:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo
    return np.asarray(out)


cdef binary_func get_reducer(string x) nogil except *:
    if REDUCERLIST.count(x) > 0:
        return REDUCERLIST[x]
    else:
        with gil:
            raise ValueError('Unknown reducer : {}'.format(x))

@cython.cdivision(True)
cpdef inline DTYPE_t logistic(DTYPE_t x, DTYPE_t shift=0., DTYPE_t width=1.) nogil:
    """
    >>> logistic(3, 0, 1)
    0.9525741268224333
    >>> logistic(-2, -5, 2)
    0.8175744761936437
    """
    return 0.5 * (tanh((x - shift) / width / 2.) + 1)


cdef inline DTYPE_t computed_quantile(DTYPE_t* data, DTYPE_t quantile,
                                      LTYPE_t size, LTYPE_t dim) nogil:
    cdef:
        LTYPE_t last_negative_indice, j, nb_zero = dim - size
        DTYPE_t res, previous
        LTYPE_t indice_quantile = <LTYPE_t>(dim * quantile)
        DTYPE_t* sorted_data = <DTYPE_t *>malloc(size  * sizeof(DTYPE_t))
        DTYPE_t* temp = <DTYPE_t *>malloc(size  * sizeof(DTYPE_t))

    memcpy(sorted_data, data, size * sizeof(DTYPE_t))
    partial_sort(sorted_data, temp, size, size, False)

    # find where is the first zero
    for j in xrange(size):
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


@cython.wraparound(False)
@cython.boundscheck(False)
def idiv_2d(A[:,:] a, B[:,:] b, const double eps=10**-9):
    cdef long i, j
    cdef double res = 0

    for i in prange(a.shape[0], nogil=True, schedule='static'):
        for j in xrange(a.shape[1]):
            res += idiv_pointwise(max(a[i, j], eps), max(b[i, j], eps))
    return res


@cython.wraparound(False)
@cython.boundscheck(False)
def idiv_flat(A[:] a, B[:] b, const double eps=10**-9):
    cdef long i, j
    cdef double res = 0

    for i in prange(a.shape[0], nogil=True, schedule='static'):
        res += idiv_pointwise(max(a[i], eps), max(b[i], eps))
    return res

@cython.wraparound(False)
@cython.boundscheck(False)
def kronii(A[:,:] a, B[:,:] b):
    cdef long m_a = a.shape[0], n_a = a.shape[1]
    cdef long m_b = b.shape[0], n_b = b.shape[1]
    cdef long ia, ja, jb, ind

    if (m_a != m_b): raise ValueError('operands could not be broadcast together with shape'
                                      '{} and {}.'.format((m_a, n_a), (m_b, n_b)))

    cdef float[:,::1] result = np.zeros(shape=(m_a, n_a * n_b), dtype=np.float32)

    with nogil:
        for ia in xrange(m_a):
            for ja in xrange(n_a):
                ind = n_b * ja
                for jb in xrange(n_b):
                    result[ia, ind] = a[ia, ja] * b[ia, jb]
                    ind += 1
    return np.asarray(result)


# That can be viewed as new KarmaSparse operation
@cython.wraparound(False)
@cython.boundscheck(False)
def indices_truncation_sorted(INT1[::1] indices, INT2[::1] indptr, LTYPE_t[::1] lower_bound, LTYPE_t[::1] upper_bound):
    """
    >>> indices = np.array([ 1,  3,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18,
    ...                      2,  3,  4,  5, 8, 10, 12, 13, 15, 18], dtype=np.int32)
    >>> indptr = np.array([0, 13, 23])
    >>> ll = np.array([3, 5])
    >>> bb = np.array([14, 19])
    >>> indices, indptr = indices_truncation_sorted(indices, indptr, ll, bb)
    >>> indices.resize(indptr[-1])
    >>> indices, indptr
    (array([ 3,  6,  7,  8,  9, 10, 11,  5,  8, 10, 12, 13, 15, 18], dtype=int32), array([ 0,  7, 14]))
    """
    cdef LTYPE_t nrows = indptr.shape[0] - 1
    if nrows <= 0:
        return np.asarray(indices), np.asarray(indptr)

    assert lower_bound.shape[0] == lower_bound.shape[0] == nrows
    assert indptr[nrows] == indices.shape[0]

    cdef LTYPE_t j, i, pos = 0, ll, uu
    cdef INT1 ind

    cdef np.ndarray[INT1, ndim=1, mode='c'] truncated_indices = np.zeros_like(np.asarray(indices))
    cdef INT2[::1] truncated_indptr = np.zeros_like(np.asarray(indptr))

    with nogil:
        for i in xrange(nrows):
            ll, uu = lower_bound[i], upper_bound[i]
            if ll >= uu:
                truncated_indptr[i + 1] = truncated_indptr[i]
                continue
            for j in xrange(indptr[i], indptr[i+1]):
                ind = indices[j]
                if ind < ll:
                    continue
                elif ind >= uu:
                    break
                else:
                    truncated_indices[pos] = ind
                    pos += 1
            truncated_indptr[i + 1] = pos

    return truncated_indices, np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def first_indices_sorted(INT1[::1] indices, INT2[::1] indptr, LTYPE_t[::1] lower_bound, LTYPE_t[::1] upper_bound):
    """
    >>> indices = np.array([ 1,  3,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18,
    ...                      2,  3,  4,  5, 8, 10, 12, 13, 15, 18], dtype=np.int32)
    >>> indptr = np.array([0, 13, 23])
    >>> ll = np.array([3, 5])
    >>> bb = np.array([14, 19])
    >>> indices, indptr = first_indices_sorted(indices, indptr, ll, bb)
    >>> indices.resize(indptr[-1])
    >>> indices, indptr
    (array([3, 5], dtype=int32), array([0, 1, 2]))
    """
    cdef LTYPE_t nrows = indptr.shape[0] - 1
    if nrows <= 0:
        return np.asarray(indices), np.asarray(indptr)

    assert lower_bound.shape[0] == lower_bound.shape[0] == nrows
    assert indptr[nrows] == indices.shape[0]

    cdef LTYPE_t j, i, pos = 0, ll, uu
    cdef INT1 ind

    cdef np.ndarray[INT1, ndim=1, mode='c'] truncated_indices = np.zeros(nrows, dtype=np.asarray(indices).dtype)
    cdef INT2[::1] truncated_indptr = np.zeros_like(np.asarray(indptr))

    with nogil:
        for i in xrange(nrows):
            ll, uu = lower_bound[i], upper_bound[i]
            if ll >= uu:
                truncated_indptr[i + 1] = truncated_indptr[i]
                continue
            for j in xrange(indptr[i], indptr[i + 1]):
                ind = indices[j]
                if ind < ll:
                    continue
                elif ind >= uu:
                    break
                else:
                    truncated_indices[pos] = ind
                    pos += 1
                    break
            truncated_indptr[i + 1] = pos

    return truncated_indices, np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def last_indices_sorted(INT1[::1] indices, INT2[::1] indptr, LTYPE_t[::1] lower_bound, LTYPE_t[::1] upper_bound):
    """
    >>> indices = np.array([ 1,  3,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18,
    ...                      2,  3,  4,  5, 8, 10, 12, 13, 15, 18], dtype=np.int32)
    >>> indptr = np.array([0, 13, 23])
    >>> ll = np.array([3, 5])
    >>> bb = np.array([14, 19])
    >>> indices, indptr = last_indices_sorted(indices, indptr, ll, bb)
    >>> indices.resize(indptr[-1])
    >>> indices, indptr
    (array([11, 18], dtype=int32), array([0, 1, 2]))
    """
    cdef LTYPE_t nrows = indptr.shape[0] - 1
    if nrows <= 0:
        return np.asarray(indices), np.asarray(indptr)

    assert lower_bound.shape[0] == lower_bound.shape[0] == nrows
    assert indptr[nrows] == indices.shape[0]

    cdef LTYPE_t j, i, pos = 0, ll, uu
    cdef INT1 ind

    cdef np.ndarray[INT1, ndim=1, mode='c'] truncated_indices = np.zeros(nrows, dtype=np.asarray(indices).dtype)
    cdef INT2[::1] truncated_indptr = np.zeros_like(np.asarray(indptr))

    with nogil:
        for i in xrange(nrows):
            ll, uu = lower_bound[i], upper_bound[i]
            if ll >= uu:
                truncated_indptr[i + 1] = truncated_indptr[i]
                continue
            for j in xrange(indptr[i + 1] - 1, indptr[i] - 1, -1):
                ind = indices[j]
                if ind < ll:
                    break
                elif ind >= uu:
                    continue
                else:
                    truncated_indices[pos] = ind
                    pos += 1
                    break
            truncated_indptr[i + 1] = pos

    return truncated_indices, np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def indices_truncation_lookup(INT1[::1] indices, INT2[::1] indptr, LTYPE_t[::1] local_dates,
                              LTYPE_t[::1] boundary_dates, LTYPE_t lower, LTYPE_t upper):
    cdef LTYPE_t nrows = indptr.shape[0] - 1
    if nrows <= 0:
        return np.asarray(indices), np.asarray(indptr)

    assert boundary_dates.shape[0] == nrows
    assert indptr[nrows] == indices.shape[0]

    cdef LTYPE_t j, i, pos = 0, ll, uu, date
    cdef INT1 ind

    cdef np.ndarray[INT1, ndim=1, mode='c'] truncated_indices = np.zeros_like(np.asarray(indices))
    cdef INT2[::1] truncated_indptr = np.zeros_like(np.asarray(indptr))

    with nogil:
        for i in prange(nrows):
            ll = boundary_dates[i]
            uu = ll + upper
            ll = ll + lower
            pos = indptr[i]

            for j in xrange(indptr[i], indptr[i+1]):
                ind = indices[j]
                date = local_dates[ind]
                if date >= ll and date < uu:
                    truncated_indices[pos] = ind
                    pos = pos + 1
            truncated_indptr[i + 1] = pos - indptr[i]
        pos = 0
        for i in xrange(nrows):
            for j in xrange(indptr[i], indptr[i] + truncated_indptr[i + 1]):
                truncated_indices[pos] = truncated_indices[j]
                pos += 1
            truncated_indptr[i + 1] += truncated_indptr[i]

    return truncated_indices, np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def last_indices_lookup(INT1[::1] indices, INT2[::1] indptr, LTYPE_t[::1] local_dates,
                        LTYPE_t[::1] boundary_dates, LTYPE_t lower, LTYPE_t upper):
    cdef LTYPE_t nrows = indptr.shape[0] - 1
    if nrows <= 0:
        return np.asarray(indices), np.asarray(indptr)

    assert boundary_dates.shape[0] == nrows
    assert indptr[nrows] == indices.shape[0]

    cdef LTYPE_t j, i, ll, uu, date, max_date, max_ind
    cdef INT1 ind

    cdef np.ndarray[INT1, ndim=1, mode='c'] truncated_indices = np.zeros(nrows, dtype=np.asarray(indices).dtype)
    cdef INT2[::1] truncated_indptr = np.zeros_like(np.asarray(indptr))

    with nogil:
        for i in prange(nrows):
            ll = boundary_dates[i]
            uu = ll + upper
            ll = ll + lower

            max_date = ll - 1
            for j in xrange(indptr[i], indptr[i + 1]):
                ind = indices[j]
                date = local_dates[ind]
                if date >= ll and date < uu and date >= max_date:
                    max_date, max_ind = date, ind
            if max_date >= ll:
                truncated_indices[i] = max_ind
            else:
                truncated_indices[i] = -1

        for i in xrange(nrows):
            if truncated_indices[i] != -1:
                truncated_indices[truncated_indptr[i]] = truncated_indices[i]
                truncated_indptr[i + 1] = truncated_indptr[i] + 1
            else:
                truncated_indptr[i + 1] = truncated_indptr[i]

    return truncated_indices, np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def first_indices_lookup(INT1[::1] indices, INT2[::1] indptr, LTYPE_t[::1] local_dates,
                         LTYPE_t[::1] boundary_dates, LTYPE_t lower, LTYPE_t upper):
    cdef LTYPE_t nrows = indptr.shape[0] - 1
    if nrows <= 0:
        return np.asarray(indices), np.asarray(indptr)

    assert boundary_dates.shape[0] == nrows
    assert indptr[nrows] == indices.shape[0]

    cdef LTYPE_t j, i, ll, uu, date, min_date, min_ind
    cdef INT1 ind

    cdef np.ndarray[INT1, ndim=1, mode='c'] truncated_indices = np.zeros(nrows, dtype=np.asarray(indices).dtype)
    cdef INT2[::1] truncated_indptr = np.zeros_like(np.asarray(indptr))

    with nogil:
        for i in prange(nrows):
            ll = boundary_dates[i]
            uu = ll + upper
            ll = ll + lower

            min_date = uu
            for j in xrange(indptr[i], indptr[i + 1]):
                ind = indices[j]
                date = local_dates[ind]
                if date >= ll and date < uu and date < min_date:
                    min_date, min_ind = date, ind
            if min_date < uu:
                truncated_indices[i] = min_ind
            else:
                truncated_indices[i] = -1

        for i in xrange(nrows):
            if truncated_indices[i] != -1:
                truncated_indices[truncated_indptr[i]] = truncated_indices[i]
                truncated_indptr[i + 1] = truncated_indptr[i] + 1
            else:
                truncated_indptr[i + 1] = truncated_indptr[i]

    return truncated_indices, np.asarray(truncated_indptr)

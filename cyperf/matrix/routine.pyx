# distutils: language = c++
#cython: embedsignature=True
#cython: nonecheck=True
#cython: overflowcheck=True
#cython: unraisable_tracebacks=True

cimport cython
from cpython.set cimport PySet_Contains
from cpython.sequence cimport PySequence_Check
from cpython.string cimport PyString_Check
from cpython.number cimport PyNumber_Check
from cpython.tuple cimport PyTuple_Check
from cpython.bytes cimport PyBytes_Check
from cpython.list cimport PyList_Check
from cpython.unicode cimport PyUnicode_Check, PyUnicode_FromEncodedObject, PyUnicode_AsUTF8String
from cyperf.tools.sort_tools cimport partial_sort_decreasing_quick, partial_sort_increasing_quick
from libc.math cimport fmin, fmax

import numpy as np
cimport numpy as np
from cyperf.tools.types import LTYPE, BOOL, FTYPE

from cyperf.tools.vector import int64Vector
from cyperf.tools.vector cimport int64Vector
from cyperf.tools.vector import float32Vector
from cyperf.tools.vector cimport float32Vector


from cython.parallel import prange
from collections import defaultdict

ctypedef fused SET_FRSET:
    frozenset
    set


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[ndim=1, dtype=int] bisect_left(ITER a, ITER_BIS x):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.
    """
    cdef long lo, hi, mid, nb = len(x), i
    cdef int[::1] out = np.zeros(nb, dtype=np.int32)

    for i in range(nb):
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


@cython.wraparound(False)
@cython.boundscheck(False)
def batch_contains_mask(ITER values, SET_FRSET kt):
    PySequence_Check(values)
    cdef:
        long i, nb = len(values)
        BOOL_t[:] result = np.zeros(nb, dtype=BOOL)
        object x

    for i in range(nb):
        x = values[i]
        try:
            if PySet_Contains(kt, x) == 1:
                result[i] = 1
        except TypeError:
            pass
    return np.asarray(result).view(np.bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def idiv_2d(A[:,:] a, B[:,:] b, const double eps=10**-9):
    cdef long i, j
    cdef double res = 0

    for i in prange(a.shape[0], nogil=True, schedule='static'):
        for j in range(a.shape[1]):
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


@cython.cdivision(True)
cdef inline FTYPE_t time_delta_to_intensity(LTYPE_t delta, FTYPE_t half_life) nogil:
    if half_life != 0.:
        return 2. ** fmax(fmin(delta / half_life, 16.), -16.)
    else:
        return 1.


# That can be viewed as new KarmaSparse operation
@cython.wraparound(False)
@cython.boundscheck(False)
def indices_truncation_lookup(INT3[::1] target_users_position_in_source_index, LTYPE_t[::1] target_dates,
                              INT1[::1] source_index_indices, INT2[::1] source_index_indptr, LTYPE_t[::1] source_dates,
                              LTYPE_t lower, LTYPE_t upper, FTYPE_t half_life, INT1 nb):
    """
    :param target_users_position_in_source_index: array, position of the `target` user keys in the `source` user index
    :param target_dates: array, `target` dates
    :param source_index_indices: representation of the `source` user index
    :param source_index_indptr: representation of the `source` user index
    :param source_dates: array, `source` dates
    :param lower: int, relative lower bound of the date window
    :param upper: int, relative upper bound of the date window
    :param half_life: float, decay to be applied to the time deltas in order to get the intensity
    :param nb: int, number of elements to keep
        if > 0: keeps only the most recent ones (closer to upper)
        if < 0: keeps only the most ancient ones (closer to lower)
        if 0: keeps all of them

    :return: (intensity, truncated_indices, truncated_indptr) to create the ks_intensity
    """
    assert len(target_users_position_in_source_index) == len(target_dates)
    assert source_index_indptr[len(source_index_indptr) - 1] == len(source_index_indices)
    assert len(source_dates) == len(source_index_indices)

    cdef LTYPE_t nrows = len(target_users_position_in_source_index)
    cdef LTYPE_t NAT = np.datetime64('NaT').astype(LTYPE)

    cdef float32Vector truncated_intensity = float32Vector(2 * nrows)
    cdef int64Vector truncated_indices = int64Vector(2 * nrows)
    cdef LTYPE_t[::1] truncated_indptr = np.zeros(nrows + 1, dtype=LTYPE)

    cdef LTYPE_t j, i, dd, uu, date, ind, p, count
    cdef int source_max_nb_matches_one_user = np.max(np.diff(np.asarray(source_index_indptr)))
    cdef INT1[::1] source_indices_one_user = np.zeros(source_max_nb_matches_one_user,
                                                      dtype=np.asarray(source_index_indices).dtype)
    cdef LTYPE_t[::1] source_dates_one_user = np.zeros(source_max_nb_matches_one_user, dtype=LTYPE)

    with nogil:
        for i in range(nrows):
            truncated_indptr[i] = truncated_indices.size()

            p = target_users_position_in_source_index[i]
            if p == -1:
                continue

            dd = target_dates[i]
            if dd == NAT:
                continue

            uu = dd + upper
            ll = dd + lower
            count = 0
            for j in range(source_index_indptr[p], source_index_indptr[p + 1]):
                ind = source_index_indices[j]
                date = source_dates[ind]
                if date == NAT:
                    continue
                if date >= ll and date < uu:
                    source_indices_one_user[count] = ind
                    source_dates_one_user[count] = date
                    count += 1
            if nb > 0:
                partial_sort_decreasing_quick(&source_dates_one_user[0], &source_indices_one_user[0], count, nb)
                count = min(count, nb)
            elif nb < 0:
                partial_sort_increasing_quick(&source_dates_one_user[0], &source_indices_one_user[0], count, -nb)
                count = min(count, -nb)

            for j in range(count):
                truncated_indices.append(source_indices_one_user[j])
                truncated_intensity.append(time_delta_to_intensity(source_dates_one_user[j] - dd, half_life))

        truncated_indptr[nrows] = truncated_indices.size()

    return np.asarray(truncated_intensity), np.asarray(truncated_indices), np.asarray(truncated_indptr)


# That can be viewed as new KarmaSparse operation
@cython.wraparound(False)
@cython.boundscheck(False)
def indices_lookup(INT3[::1] target_users_position_in_source_index,
                   INT1[::1] source_index_indices, INT2[::1] source_index_indptr,
                   INT1 nb):
    """
    :param target_users_position_in_source_index: array, position of the `target` user keys in the `source` user index
    :param source_index_indices: representation of the `source` user index
    :param source_index_indptr: representation of the `source` user index
    :param nb: int, number of elements to keep
        if > 0: keeps only the most recent ones (assuming the source index is ordered)
        if < 0: keeps only the most ancient ones (assuming the source index is ordered)
        if 0: keeps all of them

    :return: (intensity, indices, indptr) to create the ks_intensity
    """
    assert source_index_indptr[len(source_index_indptr) - 1] == len(source_index_indices)

    cdef LTYPE_t nrows = len(target_users_position_in_source_index)

    cdef int64Vector indices = int64Vector(2 * nrows)
    cdef LTYPE_t[::1] indptr = np.zeros(nrows + 1, dtype=LTYPE)

    cdef LTYPE_t j, p, start, stop, count

    with nogil:
        for i in range(nrows):
            indptr[i] = indices.size()

            p = target_users_position_in_source_index[i]
            if p == -1:
                continue

            start = source_index_indptr[p]
            stop = source_index_indptr[p + 1]
            if nb < 0:
                count = min(stop - start, - nb)
                stop = start + count
            elif nb > 0:
                count = min(stop - start, nb)
                start = stop - count

            for j in range(start, stop):
                indices.append(source_index_indices[j])

        indptr[nrows] = indices.size()

    return np.ones(len(indices), dtype=FTYPE), np.asarray(indices), np.asarray(indptr)

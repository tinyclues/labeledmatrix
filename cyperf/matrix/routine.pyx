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
from cpython.list cimport PyList_Check
from cpython.unicode cimport PyUnicode_Check, PyUnicode_FromEncodedObject, PyUnicode_AsUTF8String
from cyperf.tools.sort_tools cimport partial_sort_decreasing_quick, partial_sort_increasing_quick

import numpy as np
cimport numpy as np
from cyperf.tools.types import LTYPE, BOOL

from cyperf.where.indices_where_long cimport Vector
from cyperf.where.indices_where_long import Vector

from cython.parallel import prange
from collections import defaultdict


ctypedef fused ITERbis:
    list
    tuple
    np.ndarray[dtype=int, ndim=1]
    np.ndarray[dtype=long, ndim=1]
    np.ndarray[dtype=float, ndim=1]
    np.ndarray[dtype=double, ndim=1]
    np.ndarray[dtype=object, ndim=1]
    object

ctypedef fused SET_FRSET:
    frozenset
    set


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


cpdef str ESCAPE_CHARS = ' "\''
cpdef object TRANSLATION_TABLE = defaultdict(lambda :63,
                                            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8:
                                             8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
                                             16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23,
                                             24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31,
                                             32: 95, 33: 33, 34: 95, 35: 35, 36: 36, 37: 37, 38: 38, 39: 95,
                                             40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47,
                                             48: 48, 49: 49, 50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55,
                                             56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62, 63: 63,
                                             64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71,
                                             72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79,
                                             80: 80, 81: 81, 82: 82, 83: 83, 84: 84, 85: 85, 86: 86, 87: 87,
                                             88: 88, 89: 89, 90: 90, 91: 91, 92: 92, 93: 93, 94: 94, 95: 95,
                                             96: 96, 97: 97, 98: 98, 99: 99, 100: 100, 101: 101, 102: 102,
                                             103: 103, 104: 104, 105: 105, 106: 106, 107: 107, 108: 108,
                                             109: 109, 110: 110, 111: 111, 112: 112, 113: 113, 114: 114,
                                             115: 115, 116: 116, 117: 117, 118: 118, 119: 119, 120: 120,
                                             121: 121, 122: 122, 123: 123, 124: 124, 125: 125, 126: 126,
                                             127: 127, 224: 97, 225: 97, 226: 97, 227: 97, 228: 97, 229: 97,
                                             231: 99, 232: 101, 233: 101, 234: 101, 235: 101, 236: 105,
                                             237: 105, 238: 105, 239: 105, 241: 110, 242: 111, 243: 111,
                                             244: 111, 245: 111, 246: 111, 249: 117, 250: 117, 251: 117,
                                             252: 117, 253: 121, 255: 121, 65533: 63, 3333333333: 63})


cpdef str cy_safe_slug(object x, unify=True):
    if PyUnicode_Check(x):
        x_u = x
    else:
        x_u = PyUnicode_FromEncodedObject(x, 'utf-8', 'replace')
    slug = PyUnicode_AsUTF8String(x_u.strip(ESCAPE_CHARS).lower().translate(TRANSLATION_TABLE))
    if unify:
        slug = intern(slug)
    return slug


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef cy_domain_from_email_lambda(x, str delimiter='@', str missing='', BOOL_t unify=True):
    cdef list l = x.rsplit(delimiter, 1)

    if len(l) <= 1:
        return missing
    else:
        res = l[1]
        if PyString_Check(res) and unify:
            return intern(res)
        else:
            return res


@cython.wraparound(False)
@cython.boundscheck(False)
def batch_contains_mask(ITER values, SET_FRSET kt):
    PySequence_Check(values)
    cdef:
        long i, nb = len(values)
        BOOL_t[:] result = np.zeros(nb, dtype=BOOL)
        object x

    for i in xrange(nb):
        x = values[i]
        try:
            if PySet_Contains(kt, x) == 1:
                result[i] = 1
        except TypeError:
            pass
    return np.asarray(result).view(np.bool)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef char cy_is_exceptional(object x, SET_FRSET exceptional_set, str exceptional_char):
    cdef str y
    cdef long k, l

    if PyTuple_Check(x) or PyList_Check(x):
        l = len(x)
        if l == 0:
            return 0
        for k in xrange(l):  # all should be exception to get 1
            if not cy_is_exceptional(x[k], exceptional_set, exceptional_char):
                return 0
        return 1

    try:
        if PySet_Contains(exceptional_set, x) == 1:
            return 1
    except TypeError:
        return 0

    if PyString_Check(x):
        y = <str>x
        if y.startswith(exceptional_char):
            return 1
        return 0

    if PyNumber_Check(x) and x != x:  # np.nan
        return 1
    return 0


@cython.wraparound(False)
@cython.boundscheck(False)
def batch_is_exceptional_mask(ITER values, SET_FRSET exceptional_set, str exceptional_char):
    PySequence_Check(values)
    cdef:
        long i, k, nb = len(values)
        BOOL_t[:] result = np.zeros(nb, dtype=BOOL)
        object x
    for i in xrange(nb):
        result[i] = cy_is_exceptional(values[i], exceptional_set, exceptional_char)

    return np.asarray(result).view(np.bool_)


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


# That can be viewed as new KarmaSparse operation
@cython.wraparound(False)
@cython.boundscheck(False)
def indices_truncation_lookup(INT3[::1] target_users_position_in_source_index, LTYPE_t[::1] target_dates,
                              INT1[::1] source_index_indices, INT2[::1] source_index_indptr, LTYPE_t[::1] source_dates,
                              LTYPE_t lower, LTYPE_t upper, INT1 nb):
    """
    :param target_users_position_in_source_index: array, position of the `target` user keys in the `source` user index
    :param target_dates: array, `target` dates
    :param source_index_indices: representation of the `source` user index
    :param source_index_indptr: representation of the `source` user index
    :param source_dates: array, `source` dates
    :param lower: int, relative lower bound of the date window
    :param upper: int, relative upper bound of the date window
    :param nb: int, number of elements to keep
        if > 0: keeps only the most recent ones (closer to upper)
        if < 0: keeps only the most ancient ones (closer to lower)
        if 0: keeps all of them

    :return: (truncated_deltas, truncated_indices, truncated_indptr) to create the ks_intensity
    """
    assert len(target_users_position_in_source_index) == len(target_dates)
    assert source_index_indptr[len(source_index_indptr) - 1] == len(source_index_indices)
    assert len(source_dates) == len(source_index_indices)

    cdef LTYPE_t nrows = len(target_users_position_in_source_index)
    cdef LTYPE_t NAT = np.datetime64('NaT').astype(LTYPE)

    cdef Vector truncated_deltas = Vector(2 * nrows)
    cdef Vector truncated_indices = Vector(2 * nrows)
    cdef LTYPE_t[::1] truncated_indptr = np.zeros(nrows + 1, dtype=LTYPE)

    cdef LTYPE_t j, i, dd, uu, date, ind, p, count
    cdef int source_max_nb_matches_one_user = np.max(np.diff(np.asarray(source_index_indptr)))
    cdef INT1[::1] source_indices_one_user = np.zeros(source_max_nb_matches_one_user,
                                                      dtype=np.asarray(source_index_indices).dtype)
    cdef LTYPE_t[::1] source_dates_one_user = np.zeros(source_max_nb_matches_one_user, dtype=LTYPE)

    with nogil:
        for i in xrange(nrows):
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
            for j in xrange(source_index_indptr[p], source_index_indptr[p + 1]):
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

            for j in xrange(count):
                truncated_indices.append(source_indices_one_user[j])
                truncated_deltas.append(source_dates_one_user[j] - dd)

        truncated_indptr[nrows] = truncated_indices.size()

    return np.asarray(truncated_deltas), np.asarray(truncated_indices), np.asarray(truncated_indptr)

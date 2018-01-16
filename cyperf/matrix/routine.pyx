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


from libc.math cimport tanh
from cyperf.tools.sort_tools cimport partial_sort
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from cyperf.tools.types import LTYPE

from cyperf.where.indices_where_long cimport Vector
from cyperf.where.indices_where_long import Vector

from cython.parallel import prange
from collections import defaultdict


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
cpdef cy_domain_from_email_lambda(x, str delimiter='@', str missing='', bool unify=True):
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
        np.int8_t[:] result = np.zeros(nb, dtype=np.int8)
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
        np.int8_t[:] result = np.zeros(nb, dtype=np.int8)
        object x
    for i in xrange(nb):
        result[i] = cy_is_exceptional(values[i], exceptional_set, exceptional_char)

    return np.asarray(result).view(np.bool_)


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
def indices_truncation_sorted(INT1[::1] positions, INT1[::1] indices, INT2[::1] indptr,
                              LTYPE_t[::1] lower_bound, LTYPE_t[::1] upper_bound):
    """
    >>> positions = np.array([0, 1], dtype=np.int32)
    >>> indices = np.array([ 1,  3,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18,
    ...                      2,  3,  4,  5, 8, 10, 12, 13, 15, 18], dtype=np.int32)
    >>> indptr = np.array([0, 13, 23])
    >>> ll = np.array([3, 5])
    >>> bb = np.array([14, 19])
    >>> indices, indptr = indices_truncation_sorted(positions, indices, indptr, ll, bb)
    >>> indices, indptr
    (array([ 3,  6,  7,  8,  9, 10, 11,  5,  8, 10, 12, 13, 15, 18]), array([ 0,  7, 14]))
    """
    cdef LTYPE_t nrows = len(positions)

    assert len(lower_bound) == len(lower_bound) == nrows
    assert indptr[len(indptr) - 1] == len(indices)

    cdef LTYPE_t j, i, ll, uu
    cdef INT1 ind, p
    cdef LTYPE_t NAT = np.datetime64('NaT').astype(LTYPE)

    cdef Vector truncated_indices = Vector(2 * nrows)
    cdef LTYPE_t[::1] truncated_indptr = np.zeros(nrows + 1, dtype=LTYPE)

    with nogil:
        for i in xrange(nrows):
            truncated_indptr[i] = truncated_indices.size()
            p = positions[i]
            if p == -1:
                continue

            ll, uu = lower_bound[i], upper_bound[i]
            if ll == NAT or uu == NAT or ll >= uu:
                continue

            for j in xrange(indptr[p], indptr[p + 1]):
                ind = indices[j]
                if ind >= ll and ind < uu:
                    truncated_indices.append(ind)
                elif ind >= uu:
                    break

        truncated_indptr[nrows] = truncated_indices.size()

    return np.asarray(truncated_indices), np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def first_indices_sorted(INT1[::1] positions, INT1[::1] indices, INT2[::1] indptr,
                         LTYPE_t[::1] lower_bound, LTYPE_t[::1] upper_bound):
    """
    >>> positions = np.array([0, 1], dtype=np.int32)
    >>> indices = np.array([ 1,  3,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18,
    ...                      2,  3,  4,  5, 8, 10, 12, 13, 15, 18], dtype=np.int32)
    >>> indptr = np.array([0, 13, 23])
    >>> ll = np.array([3, 5])
    >>> bb = np.array([14, 19])
    >>> indices, indptr = first_indices_sorted(positions, indices, indptr, ll, bb)
    >>> indices, indptr
    (array([3, 5]), array([0, 1, 2]))
    """
    cdef LTYPE_t nrows = len(positions)

    assert len(lower_bound) == len(lower_bound) == nrows
    assert indptr[len(indptr) - 1] == len(indices)


    cdef LTYPE_t j, i, ll, uu
    cdef INT1 ind, p
    cdef LTYPE_t NAT = np.datetime64('NaT').astype(LTYPE)

    cdef Vector truncated_indices = Vector(nrows)
    cdef LTYPE_t[::1] truncated_indptr = np.zeros(nrows + 1, dtype=LTYPE)

    with nogil:
        for i in xrange(nrows):
            truncated_indptr[i] = truncated_indices.size()
            p = positions[i]
            if p == -1:
                continue

            ll, uu = lower_bound[i], upper_bound[i]
            if ll == NAT or uu == NAT or ll >= uu:
                continue

            for j in xrange(indptr[p], indptr[p + 1]):
                ind = indices[j]
                if ind >= ll and ind < uu:
                    truncated_indices.append(ind)
                    break
                elif ind >= uu:
                    break

        truncated_indptr[nrows] = truncated_indices.size()
    return np.asarray(truncated_indices), np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def last_indices_sorted(INT1[::1] positions, INT1[::1] indices, INT2[::1] indptr,
                        LTYPE_t[::1] lower_bound, LTYPE_t[::1] upper_bound):
    """
    >>> positions = np.array([0, 1], dtype=np.int32)
    >>> indices = np.array([ 1,  3,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18,
    ...                      2,  3,  4,  5, 8, 10, 12, 13, 15, 18], dtype=np.int32)
    >>> indptr = np.array([0, 13, 23])
    >>> ll = np.array([3, 5])
    >>> bb = np.array([14, 19])
    >>> indices, indptr = last_indices_sorted(positions, indices, indptr, ll, bb)
    >>> indices, indptr
    (array([11, 18]), array([0, 1, 2]))
    """
    cdef LTYPE_t nrows = len(positions)

    assert len(lower_bound) == len(lower_bound) == nrows
    assert indptr[len(indptr) - 1] == len(indices)


    cdef LTYPE_t j, i, ll, uu
    cdef INT1 ind, p
    cdef LTYPE_t NAT = np.datetime64('NaT').astype(LTYPE)

    cdef Vector truncated_indices = Vector(nrows)
    cdef LTYPE_t[::1] truncated_indptr = np.zeros(nrows + 1, dtype=LTYPE)

    with nogil:
        for i in xrange(nrows):
            truncated_indptr[i] = truncated_indices.size()
            p = positions[i]
            if p == -1:
                continue

            ll, uu = lower_bound[i], upper_bound[i]
            if ll == NAT or uu == NAT or ll >= uu:
                continue

            for j in xrange(indptr[p + 1] - 1, indptr[p] - 1, -1):
                ind = indices[j]
                if ind >= ll and ind < uu:
                    truncated_indices.append(ind)
                    break
                elif ind < ll:
                    break

        truncated_indptr[nrows] = truncated_indices.size()
    return np.asarray(truncated_indices), np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def indices_truncation_lookup(INT1[::1] positions, INT1[::1] indices, INT2[::1] indptr,
                              LTYPE_t[::1] local_dates,
                              LTYPE_t[::1] boundary_dates, LTYPE_t lower, LTYPE_t upper):
    assert len(positions) == len(boundary_dates)
    assert indptr[len(indptr) - 1] == len(indices)

    cdef LTYPE_t nrows = len(positions)
    cdef LTYPE_t NAT = np.datetime64('NaT').astype(LTYPE)

    cdef Vector truncated_indices = Vector(2 * nrows)
    cdef LTYPE_t[::1] truncated_indptr = np.zeros(nrows + 1, dtype=LTYPE)

    cdef LTYPE_t j, i, ll, uu, date
    cdef INT1 ind, p

    with nogil:
        for i in xrange(nrows):
            truncated_indptr[i] = truncated_indices.size()

            p = positions[i]
            if p == -1:
                continue

            ll = boundary_dates[i]
            if ll == NAT:
                continue

            uu = ll + upper
            ll = ll + lower
            for j in xrange(indptr[p], indptr[p + 1]):
                ind = indices[j]
                date = local_dates[ind]
                if date == NAT:
                    continue
                if date >= ll and date < uu:
                    truncated_indices.append(ind)

        truncated_indptr[nrows] = truncated_indices.size()

    return np.asarray(truncated_indices), np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def last_indices_lookup(INT1[::1] positions, INT1[::1] indices, INT2[::1] indptr, LTYPE_t[::1] local_dates,
                        LTYPE_t[::1] boundary_dates, LTYPE_t lower, LTYPE_t upper):
    assert len(positions) == len(boundary_dates)
    assert indptr[len(indptr) - 1] == len(indices)

    cdef LTYPE_t nrows = len(positions)
    cdef LTYPE_t NAT = np.datetime64('NaT').astype(LTYPE)

    cdef Vector truncated_indices = Vector(nrows)
    cdef LTYPE_t[::1] truncated_indptr = np.zeros(nrows + 1, dtype=LTYPE)

    cdef LTYPE_t j, i, ll, uu, date, max_date, max_ind
    cdef INT1 ind, p

    with nogil:
        for i in xrange(nrows):
            truncated_indptr[i] = truncated_indices.size()
            p = positions[i]
            if p == -1:
                continue

            ll = boundary_dates[i]
            if ll == NAT:
                continue

            uu = ll + upper
            ll = ll + lower

            max_date = ll - 1
            for j in xrange(indptr[p], indptr[p + 1]):
                ind = indices[j]
                date = local_dates[ind]
                if date == NAT:
                    continue
                if date >= ll and date < uu and date >= max_date:
                    max_date, max_ind = date, ind
            if max_date >= ll:
                truncated_indices.append(max_ind)
        truncated_indptr[nrows] = truncated_indices.size()

    return np.array(truncated_indices), np.asarray(truncated_indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def first_indices_lookup(INT1[::1] positions, INT1[::1] indices, INT2[::1] indptr, LTYPE_t[::1] local_dates,
                         LTYPE_t[::1] boundary_dates, LTYPE_t lower, LTYPE_t upper):

    assert len(positions) == len(boundary_dates)
    assert indptr[len(indptr) - 1] == len(indices)

    cdef LTYPE_t nrows = len(positions)
    cdef LTYPE_t NAT = np.datetime64('NaT').astype(LTYPE)

    cdef Vector truncated_indices = Vector(nrows)
    cdef LTYPE_t[::1] truncated_indptr = np.zeros(nrows + 1, dtype=LTYPE)

    if nrows <= 0:
        return np.asarray(indices), np.asarray(indptr)

    cdef LTYPE_t j, i, ll, uu, date, min_date, min_ind
    cdef INT1 ind, p

    with nogil:
        for i in xrange(nrows):
            truncated_indptr[i] = truncated_indices.size()
            p = positions[i]
            if p == -1:
                continue

            ll = boundary_dates[i]
            if ll == NAT:
                continue

            uu = ll + upper
            ll = ll + lower
            min_date = uu

            for j in xrange(indptr[p], indptr[p + 1]):
                ind = indices[j]
                date = local_dates[ind]
                if date == NAT:
                    continue
                if date >= ll and date < uu and date < min_date:
                    min_date, min_ind = date, ind
            if min_date < uu:
                truncated_indices.append(min_ind)

        truncated_indptr[nrows] = truncated_indices.size()

    return np.array(truncated_indices), np.asarray(truncated_indptr)

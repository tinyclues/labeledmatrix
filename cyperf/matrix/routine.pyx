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

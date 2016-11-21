# cython: embedsignature=True
# cython: overflowcheck=True
# cython: unraisable_tracebacks=True

cimport cython
from cpython.sequence cimport PySequence_Check
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF, PyObject
from cpython.dict cimport PyDict_GetItem
from cpython.sequence cimport PySequence_Check

import numpy as np
from types import DTYPE

cdef DTYPE_t Nan = np.nan


cdef bool check_values(ITER values, dtype=np.int32) except? False:
    assert PySequence_Check(values)
    if len(values) > np.iinfo(dtype).max:
        raise MemoryError("Length of list is too large : {} > {}"
                          .format(len(values), np.iinfo(dtype).max))
    return True


cpdef list take_indices_on_iterable(ITER_t iterable, INDICE_t indices):
    assert PySequence_Check(iterable)
    assert PySequence_Check(indices)
    cdef object x
    cdef long i, j, nb = len(indices)
    cdef list result = PyList_New(nb)

    for i in xrange(nb):
        with cython.wraparound(False), cython.boundscheck(False):
            j = indices[i]
        x = iterable[j]
        PyList_SET_ITEM(result, i, x)
        Py_INCREF(x)
    return result


cpdef ITER_NP_t take_indices_on_numpy(ITER_NP_t ar, INDICE_t indices):
    assert PySequence_Check(indices)
    cdef long i, j, nb = len(indices)
    cdef ITER_NP_t result = np.empty(nb, dtype=ar.dtype)

    for i in xrange(nb):
        with cython.wraparound(False), cython.boundscheck(False):
            j = indices[i]
        result[i] = ar[j]
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef list apply_python_dict(dict mapping, ITER indices, object default, bool keep_same):
    """
        Apply `mapping` (python dict) over iterable `indices`, equivalent (with about x4 speed factor) to

        ll = lambda x: mapping.get(x, x if keep_same else default)
        map(ll, indices)

        >>> mapping = {'X': 1, 'y': 4}
        >>> iterable = ['X', 3, np.arange(4), 'y', 'y']
        >>> apply_python_dict(mapping, iterable, -1, False)
        [1, -1, -1, 4, 4]
        >>> apply_python_dict(mapping, tuple(iterable), -1, False)
        [1, -1, -1, 4, 4]
        >>> apply_python_dict(mapping, tuple(iterable), -1, True)
        [1, 3, array([0, 1, 2, 3]), 4, 4]
    """
    assert PySequence_Check(indices)

    cdef long nb = len(indices), i
    cdef object x, ind
    cdef list result = PyList_New(nb)

    for i in xrange(nb):
        ind = indices[i]
        obj = PyDict_GetItem(mapping, ind)
        if obj is not NULL:
            x = <object>obj
        else:
            x = ind if keep_same else default

        PyList_SET_ITEM(result, i, x)
        Py_INCREF(x)

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[dtype=DTYPE_t, ndim=1] cast_to_float_array(ITER values, str casting):
    """
    >>> x = [4, 3, '3']

    >>> cast_to_float_array(x, 'safe') #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    TypeError: a float is required

    >>> cast_to_float_array(x, 'same_kind')
    array([  4.,   3.,  nan])

    >>> cast_to_float_array(x, 'unsafe')
    array([ 4.,  3.,  3.])

    >>> x = np.array(x, dtype=np.int)
    >>> cast_to_float_array(x, 'safe')
    array([ 4.,  3.,  3.])

    >>> x = (4, 're', 3)
    >>> cast_to_float_array(x, 'safe')
    Traceback (most recent call last):
    ...
    TypeError: a float is required

    >>> cast_to_float_array(x, 'same_kind')
    array([  4.,  nan,   3.])

    >>> cast_to_float_array(x, 'unsafe')
    array([  4.,  nan,   3.])

    >>> x = np.array(x, dtype=np.object)
    >>> cast_to_float_array(x, 'safe')
    Traceback (most recent call last):
    ...
    TypeError: a float is required

    >>> cast_to_float_array(np.array([4, 3]), 'safe')
    array([ 4.,  3.])

    """
    check_values(values)
    if isinstance(values, np.ndarray) and 'O' not in values.dtype.str:
        return values.astype(DTYPE, casting=casting, copy=False)

    cdef long i, size = len(values)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] result = np.empty(size, dtype=DTYPE)
    cdef DTYPE_t x

    if casting == 'unsafe':
        for i in xrange(size):
            try:
                x = float(values[i])
            except:
                x = Nan
            result[i] = x
    elif casting == 'same_kind':
        for i in xrange(size):
            try:
                x = values[i]
            except:
                x = Nan
            result[i] = x
    elif casting == 'safe':
        for i in xrange(size):
            result[i] = values[i]
    else:
        raise ValueError('casting should be one of {"unsafe", "safe", "same_kind"}')

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[dtype=np.int32_t, ndim=1] python_feature_hasher(ITER inp, int nb_feature):
    """
       Returns an index corresponding to the hash function of any kind of object modulo nb_features.
       The hash function employed is the python hash function.

        Parameters
        ----------
        inp : list of any kind of object (except list)
        nb_feature : number of features (max: np.iinfo(np.int32).max)

        Returns
        -------
        out : ndarray of indices (ndim=1)

        Examples
        --------
        python_feature_hasher([1, 'toto', (1, 4), Error], 2**10)
        array([1, 684, 846, 687], dtype=int32)

    """
    cdef long i, nb = len(inp)
    cdef int[::1] result = np.empty(nb, dtype=np.int32)

    for i in xrange(nb):
        result[i] = <long>(hash(inp[i])) % nb_feature
    return np.asarray(result)

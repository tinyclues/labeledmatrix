# distutils: language = c++
# cython: embedsignature=True
# cython: overflowcheck=True
# cython: unraisable_tracebacks=True

cimport cython
from cpython.sequence cimport PySequence_Check
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF, PyObject
from cpython.dict cimport PyDict_GetItem
from cpython.number cimport PyNumber_Check, PyNumber_Float, PyNumber_Long
from cpython.unicode cimport PyUnicode_Check, PyUnicode_AsEncodedString, PyUnicode_FromEncodedObject
from cpython.string cimport PyString_Check, PyString_CheckExact
from cpython.bytes cimport PyBytes_Check

import numpy as np
from cyperf.tools.types import DTYPE, LTYPE


cpdef bool check_values(ITER values, dtype=np.int32) except? False:
    assert PySequence_Check(values)
    if len(values) > np.iinfo(dtype).max:
        raise MemoryError("Length of list is too large : {} > {}"
                          .format(len(values), np.iinfo(dtype).max))
    return True


cpdef list take_indices_on_iterable(ITER iterable, INDICE_t indices):
    assert PySequence_Check(iterable)
    assert PySequence_Check(indices)
    cdef object x
    cdef long i, j, nb = len(indices)
    cdef list result = PyList_New(nb)

    for i in range(nb):
        with cython.wraparound(False), cython.boundscheck(False):
            j = indices[i]
        x = iterable[j]
        PyList_SET_ITEM(result, i, x)
        Py_INCREF(x)
    return result


cpdef ITER_NP take_indices_on_numpy(ITER_NP ar, INDICE_t indices):
    assert PySequence_Check(indices)
    cdef long i, j, nb = len(indices)
    cdef ITER_NP result = np.empty(nb, dtype=ar.dtype)

    for i in range(nb):
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

    for i in range(nb):
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
def apply_python_dict_int(dict mapping, ITER indices, long default):
    """
        Apply `mapping` (python dict) over iterable `indices`, equivalent (with about x4 speed factor) to

        ll = lambda x: mapping.get(x, default)
        map(ll, indices)

        >>> mapping = {'X': 1, 'y': 4}
        >>> iterable = ['X', 3, np.arange(4), 'y', 'y']
        >>> apply_python_dict_int(mapping, iterable, -1)
        array([ 1, -1, -1,  4,  4])
        >>> apply_python_dict_int(mapping, tuple(iterable), -1)
        array([ 1, -1, -1,  4,  4])
        >>> apply_python_dict_int({'X': 'b'}, tuple(iterable), -1) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: an integer is required
    """
    assert PySequence_Check(indices)

    cdef long nb = len(indices), i
    cdef long[::1] result = np.empty(nb, dtype=np.long)

    for i in range(nb):
        obj = PyDict_GetItem(mapping, indices[i])
        if obj is not NULL:
            result[i] = <long>(<object>obj)
        else:
            result[i] = default

    return np.asarray(result)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef list inplace_default_setter(list inp,
                                  np.ndarray[dtype=BOOL_t, ndim=1, mode='c', cast=True] mask,
                                  object default):
    """
        Replace mask elements in list in place by default value:
        --------
        >>> x = ['a'] * 10
        >>> mask = np.arange(10) % 2
        >>> inplace_default_setter(x, mask.astype(np.int8), 'b')
        ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']

    """
    assert len(inp) == len(mask)

    cdef:
        long i, nb = len(inp)

    for i in range(nb):
        if mask[i]:
            inp[i] = default

    return inp


def build_safe_decorator(default, exceptions=(Exception, )):
    """
    Pure python function but x2 faster when used from cython space

    >>> safe_int = build_safe_decorator(default=0)(int)
    >>> list(map(safe_int, ['4', '4.1', 'RRRR', 5.4]))
    [4, 0, 0, 5]
    """
    def decorator(func):
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions:
                return default
        return new_func

    return decorator

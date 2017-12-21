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
from cpython.string cimport PyString_Check

import numpy as np
from types import DTYPE, LTYPE


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

    for i in xrange(nb):
        obj = PyDict_GetItem(mapping, indices[i])
        if obj is not NULL:
            result[i] = <long>(<object>obj)
        else:
            result[i] = default

    return np.asarray(result)


@cython.wraparound(False)
@cython.boundscheck(False)
def cast_to_float_array(ITER values, str casting="unsafe", DTYPE_t default=np.nan):
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

    """
    check_values(values)
    if isinstance(values, np.ndarray) and values.dtype.kind != 'O':
        try:
            return values.astype(DTYPE, casting=casting, copy=False)
        except (ValueError, TypeError):
            pass

    cdef long i, size = len(values)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] result = np.empty(size, dtype=DTYPE)

    if casting == 'unsafe':
        for i in xrange(size):
            if PyNumber_Check(values[i]):
                result[i] = <DTYPE_t>values[i]
            else:
                try:
                    result[i] = PyNumber_Float(values[i])
                except:
                    result[i] = default
    elif casting == 'same_kind':
        for i in xrange(size):
            try:
                result[i] = <DTYPE_t>values[i]
            except:
                result[i] = default
    elif casting == 'safe':
        for i in xrange(size):
            result[i] = <DTYPE_t>values[i]
    else:
        raise ValueError('casting should be one of {"unsafe", "safe", "same_kind"}')

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
def cast_to_long_array(ITER values, str casting="unsafe", LTYPE_t default=np.iinfo(LTYPE).min):
    """
    >>> x = [4, 3, '3']

    >>> cast_to_long_array(x, 'safe') #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    TypeError: an integer is required

    >>> cast_to_long_array(x, 'unsafe')
    array([4, 3, 3])

    """
    check_values(values)
    if isinstance(values, np.ndarray) and values.dtype.kind != 'O':
        try:
            return values.astype(LTYPE, casting=casting, copy=False)
        except (ValueError, TypeError):
            pass

    cdef long i, size = len(values)
    cdef np.ndarray[dtype=LTYPE_t, ndim=1] result = np.empty(size, dtype=LTYPE)

    if casting == 'unsafe':
        for i in xrange(size):
            if PyNumber_Check(values[i]):
                if values[i] != values[i]:
                    result[i] = default
                else:
                    result[i] = <LTYPE_t>values[i]
            else:
                try:
                    result[i] = PyNumber_Long(PyNumber_Float(values[i]))
                except:
                    result[i] = default
    elif casting == 'same_kind':
        for i in xrange(size):
            try:
                result[i] = <LTYPE_t>values[i]
            except:
                result[i] = default
    elif casting == 'safe':
        for i in xrange(size):
            result[i] =  <LTYPE_t>values[i]
    else:
        raise ValueError('casting should be one of {"unsafe", "safe", "same_kind"}')

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
def cast_to_ascii(ITER values, str default='`!#CoercerError'):
    """
    Equivalent (up to default value in case of error) to
    x: x.encode('ascii', errors='ignore') \
                if isinstance(x, unicode) \
                else str(unicode(x, 'ascii', errors='ignore')) \
                if isinstance(x, basestring) else str(x)

    """
    cdef:
        long i, nb = len(values)
        object x
        str sx
        list result = []

    for i in xrange(nb):
        x = values[i]
        if PyUnicode_Check(x):
            sx = PyUnicode_AsEncodedString(x, 'ascii', 'ignore')
        elif PyString_Check(x):
            sx = str(PyUnicode_FromEncodedObject(x, 'ascii', 'ignore'))
        else:
            try:
                sx = str(x)
            except:
                sx = default
        result.append(sx)

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
def cast_to_unicode(ITER values, input_encoding='utf-8', unicode default=u'`!#CoercerError'):
    """
    Equivalent (up to default value in case of error) to
    (unicode(string, self.input_encoding, errors='ignore') if isinstance(string, str) else string)\
          .encode('utf-8', errors='ignore') if isinstance(string, basestring) else unicode(string)

    """
    cdef:
        long i, nb = len(values)
        object x, sx
        list result = []
        char * ie = input_encoding

    for i in xrange(nb):
        x = values[i]
        if PyString_Check(x):
            sx = PyUnicode_AsEncodedString(PyUnicode_FromEncodedObject(x, ie, 'ignore'),
                                           'utf-8', 'ignore')
        elif PyUnicode_Check(x):
            sx = PyUnicode_AsEncodedString(x, 'utf-8', 'ignore')
        else:
            try:
                sx = unicode(x)
            except:
                sx = default
        result.append(sx)

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


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef list inplace_default_setter(list inp,
                                  np.ndarray[dtype=np.int8_t, ndim=1, mode='c', cast=True] mask,
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

    for i in xrange(nb):
        if mask[i]:
            inp[i] = default

    return inp


def build_safe_decorator(default, exceptions=(Exception, )):
    """
    Pure python function but x2 faster when used from cython space

    >>> safe_int = build_safe_decorator(default=0)(int)
    >>> map(safe_int, ['4', '4.1', 'RRRR', 5.4])
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

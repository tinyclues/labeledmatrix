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


cdef bool check_values(ITER values, dtype=np.int32) except? False:
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
def cast_to_float_array(ITER values, str casting="unsafe", DTYPE_t default=np.nan):
    """
    It should always return a copy !

    >>> x = [4, 3, '3']

    >>> cast_to_float_array(x, 'safe') #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    TypeError: ...

    >>> cast_to_float_array(x, 'same_kind')
    array([ 4.,  3., nan])

    >>> cast_to_float_array(x, 'unsafe')
    array([4., 3., 3.])

    """
    check_values(values)
    if isinstance(values, np.ndarray) and values.dtype.kind not in ['O', 'S', 'U']:
        try:
            return values.astype(DTYPE, casting=casting)
        except (ValueError, TypeError):
            pass

    cdef long i, size = len(values)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] result = np.empty(size, dtype=DTYPE)

    if casting == 'unsafe':
        for i in range(size):
            if PyNumber_Check(values[i]):
                result[i] = <DTYPE_t>values[i]
            else:
                try:
                    result[i] = PyNumber_Float(values[i])
                except:
                    result[i] = default
    elif casting == 'same_kind':
        for i in range(size):
            try:
                result[i] = <DTYPE_t>values[i]
            except:
                result[i] = default
    elif casting == 'safe':
        for i in range(size):
            result[i] = <DTYPE_t>values[i]
    else:
        raise ValueError('casting should be one of {"unsafe", "safe", "same_kind"}')

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
def cast_to_long_array(ITER values, str casting="unsafe", LTYPE_t default=np.iinfo(LTYPE).min):
    """
    It should always return a copy !

    >>> x = [4, 3, '3']

    >>> cast_to_long_array(x, 'safe') #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    TypeError: an integer is required

    >>> cast_to_long_array(x, 'unsafe')
    array([4, 3, 3])

    """
    check_values(values)
    if isinstance(values, np.ndarray) and values.dtype.kind not in ['O', 'S', 'U']:
        try:
            return values.astype(LTYPE, casting=casting)
        except (ValueError, TypeError):
            pass

    cdef long i, size = len(values)
    cdef np.ndarray[dtype=LTYPE_t, ndim=1] result = np.empty(size, dtype=LTYPE)

    if casting == 'unsafe':
        for i in range(size):
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
        for i in range(size):
            try:
                result[i] = <LTYPE_t>values[i]
            except:
                result[i] = default
    elif casting == 'safe':
        for i in range(size):
            result[i] =  <LTYPE_t>values[i]
    else:
        raise ValueError('casting should be one of {"unsafe", "safe", "same_kind"}')

    return result


# TO remove in PY3
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
        object x, sx
        list result = []

    for i in range(nb):
        x = values[i]
        if PyUnicode_Check(x):
            sx = PyUnicode_AsEncodedString(x, 'ascii', 'ignore')
        elif PyString_Check(x) or PyBytes_Check(x):
            sx = str(PyUnicode_FromEncodedObject(x, 'ascii', 'ignore'))
        else:
            try:
                sx = str(x)
            except:
                sx = default
        result.append(sx)

    return result


# TO remove in PY3
@cython.wraparound(False)
@cython.boundscheck(False)
def cast_to_unicode(ITER values, char * input_encoding=b'utf-8', unicode default=u'`!#CoercerError'):
    """
    Equivalent (up to default value in case of error) to
    (unicode(string, self.input_encoding, errors='ignore') if isinstance(string, str) else string)\
          .encode('utf-8', errors='ignore') if isinstance(string, basestring) else unicode(string)

    """
    cdef:
        long i, nb = len(values)
        object x, sx
        list result = []

    for i in range(nb):
        x = values[i]
        if PyUnicode_Check(x):
            sx = PyUnicode_AsEncodedString(x, 'utf-8', 'ignore')
        elif PyString_Check(x) or PyBytes_Check(x):
            sx = PyUnicode_AsEncodedString(PyUnicode_FromEncodedObject(x, input_encoding, 'ignore'), 'utf-8', 'ignore')
        else:
            try:
                sx = unicode(x)
            except:
                sx = default
        result.append(sx)

    return result


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


def coalesce_is_not_none(*args, object default):
    """
    Pure python fonction but x2 faster when defined from cython space.
    It is also x2 faster than coalesce_generic with predicate=lambda x: x is not None.

    Examples
    --------
    >>> coalesce_is_not_none(None, 2, 4, default=0)
    2
    >>> coalesce_is_not_none(None, None, None, default=0)
    0
    """
    for arg in args:
        if arg is not None:
            return arg
    else:
        return default


def coalesce_generic(*args, object predicate, object default):
    """
    Pure python fonction but x2 faster when defined from cython space.

    Examples
    --------
    >>> coalesce_generic(0, -12, 42, predicate=lambda x: x > 0, default=-1)
    42
    >>> coalesce_generic('a', 'b', 'c', predicate=lambda x: x > 'e', default='')
    ''
    """
    for arg in args:
        if predicate(arg):
            return arg
    else:
        return default


@cython.wraparound(False)
@cython.boundscheck(False)
def cy_safe_intern(ITER a):
    """
    x3 faster equivalent to map(safe_intern, a) where
    safe_intern = lambda x: intern(str(x)) if isinstance(x, str) else x

    >>> numpy_string = np.array(['TTTT'])[0]
    >>> s1 = 'foo!'
    >>> s2 = 'foo!'
    >>> a = [s1, s2, numpy_string, numpy_string, 4]
    >>> a[0] is a[1]
    False
    >>> b = cy_safe_intern(a)
    >>> b
    ['foo!', 'foo!', 'TTTT', 'TTTT', 4]
    >>> b[0] is b[1]
    True
    >>> b[2] is b[3]
    True
    """
    assert PySequence_Check(a)

    cdef long i, nb = len(a)
    cdef list result = PyList_New(nb)
    cdef object x, out

    for i in range(nb):
        x = a[i]
        if PyString_CheckExact(x):
            out = intern(x)
        elif PyString_Check(x):  # numpy string
            out = intern(str(x))
        else:
            out = x  # keep it as it was

        PyList_SET_ITEM(result, i, out)
        Py_INCREF(out)
    return result


cdef class Unifier(dict):
    """
        Return a reference to a deduplicated object.
        That mimics python intern mechanism but works also for non-string objects.

        The Unifier is used to deduplicate objects. It takes an object as an
        argument. It tries to find it in its internal store. If the object if found,
        it is returned (i.e. a reference to this object). Otherwise the unifier adds
        the object in its internal store. As the internal store behaves as a
        mapping, it hashes objects. If an object cannot be hashed, it is returned as-is.

    """

    def unify(self, obj):
        """
            it's here for doc-test purpose.

            Examples:

            Let's define a variable *a* and add it into the unifier: ::

                >>> a = 'abc'
                >>> unifier = Unifier()
                >>> ref_a = unifier.unify(a)
                >>> ref_a is a
                True

            *ref_a* references *a*. Now we assign to another variable *b*, the same
            value `'abc'` as *a* and pass it to the unifier: ::

                >>> b = 'abc'
                >>> ref_b = unifier.unify(b)

            The unifier also references *a* because both variables contain the same
            value: ::

                >>> ref_b is a
                True

            Now we try this with a value that does not support hashing: ::

                >>> l = [1,2,3]
                >>> ref_l = unifier.unify(l)
                >>> ref_l is l
                True
                >>> ref_l2 = unifier.unify([1,2,3])
                >>> ref_l2 is not l
                True

        """
        try:
            return self[obj]
        except KeyError:
            self[obj] = obj
        except TypeError:
            pass
        return obj

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def map(self, ITER seq):
        """
        Idea : to use limited memory dict here
        optimized version of map(self.unify, seq)
        """
        assert PySequence_Check(seq)

        cdef long i, nb = len(seq)
        cdef list result = PyList_New(nb)
        cdef object x, out
        cdef dict unifier = <dict>self

        for i in range(nb):
            x = seq[i]
            obj = PyDict_GetItem(unifier, x)
            if obj is not NULL:
                out = <object>obj
            else:
                try:
                    unifier[x] = x
                except TypeError:
                    pass
                out = x

            PyList_SET_ITEM(result, i, out)
            Py_INCREF(out)

        return result

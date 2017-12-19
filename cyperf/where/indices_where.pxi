#cython: embedsignature=True
#cython: wraparound=False
#cython: boundscheck=False


import cython
from cpython cimport PyAnySet_Check, PyDict_Check, PyNumber_Check, PyBool_Check
from cpython.sequence cimport PySequence_Check


class stdvector_base:
     pass

cdef class Vector:

    # cdef vector[ITYPE_t] vector_buffer

    def __cinit__(Vector self, long n=1):
        self.vector_buffer.reserve(n)

    def __dealloc__(Vector self):
        self.vector_buffer.clear()

    def __len__(Vector self):
        return self.vector_buffer.size()

    cdef long size(Vector self) nogil:
        return self.vector_buffer.size()

    cdef inline void append(Vector self, ITYPE_t x) nogil:
        self.vector_buffer.push_back(x)

    def __array__(self):
        return self.asarray()

    cpdef np.ndarray[dtype=ITYPE_t, ndim=1] asarray(Vector self):
        self.vector_buffer.shrink_to_fit()  # this needs c++11

        dtype = np.dtype(ITYPE)
        if len(self) == 0:
            return np.array([], dtype=dtype)
        # trick to pass by buffer and avoid any data being copied!!
        # see:
        # http://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        # http://docs.cython.org/src/userguide/buffer.html
        # https://jakevdp.github.io/blog/2014/05/05/introduction-to-the-python-buffer-protocol/
        # https://developers.google.com/protocol-buffers/docs/pythontutorial#extending-a-protocol-buffer

        base = stdvector_base()
        base.__array_interface__ = dict(
            data = (<np.uintp_t>self.vector_buffer.data(), False),
            descr = dtype.descr,
            shape = (self.vector_buffer.size(),),
            strides = (dtype.itemsize,),
            typestr = dtype.str,
            version = 3)
        base.Vector = self
        return np.asarray(base)


cpdef inline bool has_contains_attr(object value) except? False:
    if PySequence_Check(value) or PyAnySet_Check(value) or PyDict_Check(value):
        return True
    elif PyNumber_Check(value) or PyBool_Check(value):
        return False
    else:
        return hasattr(value, '__contains__')  # very long call


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where(ITER column, object value=None):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column[i]:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_not(ITER column, object value=None):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if not column[i]:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_in(ITER column, object value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    if not isinstance(value, (set, dict)):
        try:
            value = set(value)
        except:
            pass

    for i in xrange(size):
        if column[i] in value:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_not_in(ITER column, object value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    if not isinstance(value, (set, dict)):
        try:
            value = set(value)
        except:
            pass

    for i in xrange(size):
        if column[i] not in value:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_eq(ITER column, VAL_T value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column[i] == value:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_ne(ITER column, VAL_T value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column[i] != value:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_lt(ITER column, VAL_T value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column[i] < value:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_le(ITER column, VAL_T value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column[i] <= value:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_gt(ITER column, VAL_T value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column[i] > value:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_ge(ITER column, VAL_T value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column[i] >= value:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_between(ITER column, object value):
    check_values(column, dtype=ITYPE)
    cdef object down = value[0], up = value[1]
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if down <= column[i] < up:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_contains(ITER column, object value):
    check_values(column, dtype=ITYPE)
    cdef object x
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        x = column[i]
        if has_contains_attr(x) and value in x:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_not_contains(ITER column, object value):
    check_values(column, dtype=ITYPE)
    cdef object x
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        x = column[i]
        if has_contains_attr(x) and value not in x:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_operator(ITER column,
                                                               object operator_function,
                                                               object value):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if operator_function(column[i], value):
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_callable(ITER column, object fn):
    check_values(column, dtype=ITYPE)
    cdef ITYPE_t i, size = len(column)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if fn(column[i]):
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_same(ITER column1, ITER_BIS column2):
    check_values(column1, dtype=ITYPE)
    check_values(column2, dtype=ITYPE)
    assert len(column1) == len(column2)
    cdef ITYPE_t i, size = len(column1)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column1[i] == column2[i]:
            result.append(i)
    return result.asarray()


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] indices_where_not_same(ITER column1, ITER_BIS column2):
    check_values(column1, dtype=ITYPE)
    check_values(column2, dtype=ITYPE)
    assert len(column1) == len(column2)
    cdef ITYPE_t i, size = len(column1)
    cdef Vector result = Vector(size)

    for i in xrange(size):
        if column1[i] != column2[i]:
            result.append(i)
    return result.asarray()

# distutils: language = c++
#cython: wraparound=False
#cython: boundscheck=False

from cpython cimport PyAnySet_Check, PyDict_Check, PyNumber_Check, PyBool_Check, PySequence_Check

cimport numpy as np
from cyperf.tools.types cimport bool, ITER, ITER_BIS
from cyperf.tools.vector cimport int32Vector, int64Vector
from cyperf.tools.getter cimport check_values


ctypedef fused IVec:
    int32Vector
    int64Vector


ctypedef fused VAL_T:
    int
    long
    float
    double
    object


cdef bool has_contains_attr(object value) except? False:
    if PySequence_Check(value) or PyAnySet_Check(value) or PyDict_Check(value):
        return True
    elif PyNumber_Check(value) or PyBool_Check(value):
        return False
    else:
        return hasattr(value, '__contains__')  # very long call


def indices_where(IVec result, ITER column, object value=None):
    cdef long i, size = len(column)
    for i in range(size):
        if column[i]:
            result.append(i)
    return result.asarray()


def indices_where_not(IVec result, ITER column, object value=None):
    cdef long i

    for i in range(len(column)):
        if not column[i]:
            result.append(i)
    return result.asarray()


def indices_where_in(IVec result, ITER column, object value):
    cdef long i

    if not isinstance(value, (set, dict)):
        try:
            value = set(value)
        except:
            pass

    for i in range(len(column)):
        if column[i] in value:
            result.append(i)
    return result.asarray()


def indices_where_not_in(IVec result, ITER column, object value):
    cdef long i

    if not isinstance(value, (set, dict)):
        try:
            value = set(value)
        except:
            pass

    for i in range(len(column)):
        if column[i] not in value:
            result.append(i)
    return result.asarray()


def indices_where_eq(IVec result, ITER column, VAL_T value):
    cdef long i

    for i in range(len(column)):
        if column[i] == value:
            result.append(i)
    return result.asarray()


def indices_where_ne(IVec result, ITER column, VAL_T value):
    cdef long i

    for i in range(len(column)):
        if column[i] != value:
            result.append(i)
    return result.asarray()


def indices_where_lt(IVec result, ITER column, VAL_T value):
    cdef long i

    for i in range(len(column)):
        if column[i] < value:
            result.append(i)
    return result.asarray()


def indices_where_le(IVec result, ITER column, VAL_T value):
    cdef long i

    for i in range(len(column)):
        if column[i] <= value:
            result.append(i)
    return result.asarray()


def indices_where_gt(IVec result, ITER column, VAL_T value):
    cdef long i

    for i in range(len(column)):
        if column[i] > value:
            result.append(i)
    return result.asarray()


def indices_where_ge(IVec result, ITER column, VAL_T value):
    cdef long i

    for i in range(len(column)):
        if column[i] >= value:
            result.append(i)
    return result.asarray()


def indices_where_between(IVec result, ITER column, object value):
    cdef object down = value[0], up = value[1]
    cdef long i

    for i in range(len(column)):
        if down <= column[i] < up:
            result.append(i)
    return result.asarray()


def indices_where_contains(IVec result, ITER column, object value):
    cdef object x
    cdef long i

    for i in range(len(column)):
        x = column[i]
        if has_contains_attr(x) and value in x:
            result.append(i)
    return result.asarray()


def indices_where_not_contains(IVec result, ITER column, object value):
    cdef object x
    cdef long i

    for i in range(len(column)):
        x = column[i]
        if has_contains_attr(x) and value not in x:
            result.append(i)
    return result.asarray()


def indices_where_operator(IVec result, ITER column, object operator_function, object value):
    cdef long i

    for i in range(len(column)):
        if operator_function(column[i], value):
            result.append(i)
    return result.asarray()


def indices_where_callable(IVec result, ITER column, object fn):
    cdef long i

    for i in range(len(column)):
        if fn(column[i]):
            result.append(i)
    return result.asarray()


def indices_where_same(IVec result, ITER column1, ITER_BIS column2):
    assert PySequence_Check(column2)
    assert len(column1) == len(column2)
    cdef long i

    for i in range(len(column1)):
        if column1[i] == column2[i]:
            result.append(i)
    return result.asarray()


def indices_where_not_same(IVec result, ITER column1, ITER_BIS column2):
    assert PySequence_Check(column2)
    assert len(column1) == len(column2)
    cdef long i

    for i in range(len(column1)):
        if column1[i] != column2[i]:
            result.append(i)
    return result.asarray()

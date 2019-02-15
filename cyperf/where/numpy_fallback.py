from __future__ import absolute_import
import numpy as np


def _indices_from_condition(cond, length):
    """
    array (of dtype np.string) == numeric) currently returns False
    FutureWarning: elementwise comparison failed; returning scalar instead,
    but in the future will perform elementwise comparison
    >>> a = np.arange(4).astype('S')
    """
    if cond is False:
        return np.array([], dtype=np.int32)
    elif cond is True:
        return np.arange(length)
    else:
        return np.where(cond)[0]


def indices_where(column, value=None):
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html
    if column.dtype.kind in ['S', 'U', 'O', 'V']:
        return np.where(column)[0]
    else:
        return np.where(column.astype(np.bool_, copy=False))[0]


def indices_where_not(column, value=None):
    if column.dtype.kind == 'S':
        return np.where(column == b"")[0]
    elif column.dtype.kind == 'U':
        return np.where(column == u"")[0]
    else:
        return np.where(~column.astype(np.bool_, copy=False))[0]


def indices_where_eq(column, value):
    return _indices_from_condition(column == value, len(column))


def indices_where_ne(column, value):
    return _indices_from_condition(column != value, len(column))


def indices_where_lt(column, value):
    return _indices_from_condition(column < value, len(column))


def indices_where_le(column, value):
    return _indices_from_condition(column <= value, len(column))


def indices_where_gt(column, value):
    return _indices_from_condition(column > value, len(column))


def indices_where_ge(column, value):
    return _indices_from_condition(column >= value, len(column))


def indices_where_between(column, value):
    down, up = value
    return _indices_from_condition((down <= column) & (column < up), len(column))


def indices_where_same(column1, column2):
    assert len(column1) == len(column2)
    return _indices_from_condition(column1 == column2, len(column1))


def indices_where_not_same(column1, column2):
    assert len(column1) == len(column2)
    return _indices_from_condition(column1 != column2, len(column1))

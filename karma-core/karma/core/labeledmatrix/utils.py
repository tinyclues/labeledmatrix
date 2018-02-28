#
# Copyright tinyclues, All rights reserved
#

import numpy as np
from cytoolz import merge as dict_merge

from cyperf.matrix.karma_sparse import KarmaSparse, ks_hstack, ks_diag, dense_pivot
from cyperf.indexing.indexed_list import reversed_index
from karma.core.utils.collaborative_tools import simple_counter
from karma.learning.matrix_utils import safe_multiply, align_along_axis, safe_add


def lm_aggregate_pivot(dataframe, key, axis, values=None, aggregator="sum", sparse=True):
    """
    This can be used as routine to compute pivot matrices with associative aggregators (#, sum, min, max, !, last)
    # TODO : iterate over different values/aggregators and to use in df.pivot
    # TODO : implement mean aggregators

    Compare with the example from dataframe.pivot :

        >>> from karma import DataFrame, Column
        >>> d = DataFrame()
        >>> d['gender'] = Column(['1', '1', '2', '2', '1', '2', '1'])
        >>> d['revenue'] = Column([100,  42,  60,  30,  80,  35,  33])
        >>> d['csp'] = Column(['+', '-', '+', '-', '+', '-', '-'])

        >>> lm_aggregate_pivot(d, 'gender', 'csp', 'revenue', 'sum').to_dense()\
            .to_vectorial_dataframe().preview()  #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      180.0    75.0
        2      60.0     65.0

        >>> lm_aggregate_pivot(d, 'gender', 'csp', 'revenue', 'min', sparse=False)\
            .to_vectorial_dataframe().preview() #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      80.0     33.0
        2      60.0     30.0

        >>> lm_aggregate_pivot(d, 'gender', 'csp', 'revenue', 'max').to_dense()\
            .to_vectorial_dataframe().preview() #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      100.0    42.0
        2      60.0     35.0
    """
    aggregator_map = {'sum': 'add',
                      'min': 'min',
                      'max': 'max',
                      '!': 'first',
                      'last': 'last',
                      'first': 'first'}
    from karma.core.labeledmatrix import LabeledMatrix
    if values is not None:
        val = np.asarray(dataframe[values][:])
        val = val.astype(np.promote_types(val.dtype, np.float32), copy=False)
    else:
        val = np.ones(len(dataframe), dtype=np.float32)

    val_a, ind_a = dataframe[key].reversed_index()
    val_b, ind_b = dataframe[axis].reversed_index()

    shape = (len(val_a), len(val_b))
    if sparse:
        matrix = KarmaSparse((val, (ind_a, ind_b)), shape=shape, format="csr",
                             aggregator=aggregator_map[aggregator])
    else:
        matrix = dense_pivot(ind_a, ind_b, val, shape=shape,
                             aggregator=aggregator_map[aggregator], default=0)
    return LabeledMatrix((val_a, val_b), matrix)


def aeq(matrix1, matrix2):
    return np.allclose((KarmaSparse(matrix1) - KarmaSparse(matrix2)).norm(), 0)


def lm_occurence(val0, val1, dense_output=False):
    """
    >>> lm = lm_occurence([0, 1, 1, 0], ['a', 'b', 'a', 'a'])
    >>> lm.to_flat_dataframe().sort_by('similarity', reverse=True).preview() #doctest: +NORMALIZE_WHITESPACE
    ------------------------
    col0 | col1 | similarity
    ------------------------
    0      a      2.0
    1      b      1.0
    1      a      1.0
    """
    from karma.core.labeledmatrix import LabeledMatrix
    return LabeledMatrix(*simple_counter((val0, val1), sparse=not dense_output))


def lm_from_dict(my_dict, dense_output=False):
    """
    >>> my_dict = {'a': 'x', 'c': 'y', 'b': 'y'}
    >>> lm = lm_from_dict(my_dict).sort()
    >>> lm.label
    (['a', 'b', 'c'], ['x', 'y'])
    >>> aeq(lm.matrix, np.array([[1, 0],[0, 1],[0, 1]]))
    True
    """
    return lm_occurence(my_dict.keys(), my_dict.values(), dense_output)


def lm_decayed_pivot_from_dataframe(dataframe, key, axis, axis_deco=None,
                                    date_column='date', half_life=300.):
    """
    Computes a pivot and returns it as a LabeledMatrix from an event DataFrame
    (with a date).
    The values in the cells of the returned LabeledMatrix represents the number
    of occurences of an axis instance for a key. If the event occured before
    the most recent event its weight is reduced.
    >>> from karma.core.dataframe import DataFrame
    >>> data = DataFrame([['abc@fr', 1, 'first', '2015-02-12'],
    ...                   ['jkl@uk', 1, 'first', '2015-03-12'],
    ...                   ['abc@fr', 4, 'fourth', '2015-04-12'],
    ...                   ['bcd@de', 4, 'fourth', '2015-05-12'],
    ...                   ['bcd@de', 4, 'fourth', '2015-06-12'],
    ...                   ['bcd@de', 4, 'fourth', '2012-02-12'],
    ...                   ['bcd@de', 4, 'fourth', '2013-02-12'],
    ...                   ['bcd@de', 4, 'fourth', '2014-02-12'],
    ...                   ['bcd@de', 4, 'fourth', '2015-02-13'],
    ...                   ['abc@fr', 1, 'first', '2015-02-14'],
    ...                   ['abc@fr', 1, 'first', '2015-02-15'],
    ...                   ['abc@fr', 1, 'first', '2015-02-16'],
    ...                   ['abc@fr', 1, 'first', '2015-02-17'],
    ...                   ['abc@fr', 1, 'first', '2015-02-18'],
    ...                   ['bcd@de', 3, 'third', '2015-02-19']],
    ...                   ['name', 'cat', 'long_cat', 'date'])
    >>> res1 = lm_decayed_pivot_from_dataframe(data, key='name', axis='cat')
    >>> from karma.core.labeledmatrix import LabeledMatrix
    >>> isinstance(res1, LabeledMatrix)
    True
    >>> res1.matrix.toarray()
    array([[1.        , 0.80850765, 0.        ],
           [0.86854149, 0.        , 1.        ],
           [0.        , 0.        , 0.77021511]])
    >>> res1.label
    ([1, 4, 3], ['abc@fr', 'jkl@uk', 'bcd@de'])
    >>> res2 = lm_decayed_pivot_from_dataframe(data, key='name', axis='cat',
    ...                                             half_life=30.)
    >>> np.all(res2.matrix.toarray() <= res1.matrix.toarray())
    True
    >>> res3 = lm_decayed_pivot_from_dataframe(data, key='name', axis='cat',
    ...                                        half_life=30.)
    >>> res3.deco
    ({}, {})
    >>> deco = data.horizontal_dict('cat', 'long_cat')
    >>> res3 = lm_decayed_pivot_from_dataframe(data, key='name', axis='cat',
    ...                                        axis_deco=deco, half_life=30.)
    >>> res3.deco
    ({1: 'first', 3: 'third', 4: 'fourth'}, {})
    """
    from karma.core.labeledmatrix import LabeledMatrix
    deco = axis_deco if axis_deco else {}
    date_arr = np.array(dataframe[date_column][:], dtype='datetime64[D]')
    decayed_series = 2 ** (-(date_arr.max() - date_arr).astype(np.float64) / half_life)
    key_values, key_ind = reversed_index(dataframe[key][:])
    axis_values, axis_ind = reversed_index(dataframe[axis][:])

    pivot_lm = LabeledMatrix((axis_values, key_values),
                             KarmaSparse((decayed_series, (axis_ind, key_ind)), format="csr"),
                             (deco, {}))
    # WARNING is this still usefull ?
    pivot_lm = pivot_lm._minimum(pivot_lm.nonzero_mask())
    return pivot_lm


def lmdiag(key, values, sdeco={}, dense_output=False):
    """
    >>> lm = lmdiag(["b", "c"], [3, 50000000]).sort()
    >>> lm.label
    (['b', 'c'], ['b', 'c'])
    >>> aeq(lm.matrix, np.array([[3, 0], [0, 50000000]]))
    True
    >>> lm.matrix.format
    'csr'
    """
    # XXX : we must get rid of this circular import
    from karma.core.labeledmatrix import LabeledMatrix
    matrix = ks_diag(np.asarray(values))
    if dense_output:
        return LabeledMatrix((key, key), matrix.toarray(), deco=(sdeco, sdeco))
    else:
        return LabeledMatrix((key, key), matrix, deco=(sdeco, sdeco))


def co(axis):
    """
    Given an axis, return the other one.

    :param axis: an axis, in range [0, 1]
    :return: the co-axis in range [0, 1]

    Exemples: ::

        >>> co(0)
        1
        >>> co(1)
        0
        >>> co(3)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        RuntimeError: axis 3 is out of range [0,1]

    """
    # TODO: check usage of this
    if axis == 0:
        return 1
    elif axis == 1:
        return 0
    else:
        raise RuntimeError("axis {} is out of range [0,1]".format(axis))


def lm_block(my_dict, dense_output=False):
    """
    >>> my_dict = {'a': 'x', 'c': 'y', 'b': 'y'}
    >>> lm = lm_block(my_dict).sort()
    >>> lm.label
    (['a', 'b', 'c'], ['a', 'b', 'c'])
    >>> aeq(lm.matrix.toarray(), np.array([[1, 0, 0],[0, 1, 1], [0, 1, 1]]))
    True
    """
    lm_flat = lm_from_dict(my_dict, dense_output=dense_output)
    return lm_flat._dot(lm_flat.transpose())


def _zipmerge(d1, d2):
    """
    >>> aa = {'a': 3, 'c': 4}
    >>> bb = {'a': 5, 'b': 4}
    >>> _zipmerge((aa, bb), (bb, aa))
    ({'a': 5, 'c': 4, 'b': 4}, {'a': 3, 'c': 4, 'b': 4})
    """
    return tuple([dict_merge(x, y) for x, y in zip(d1, d2)])


def hstack(list_of_lm, dense_output=False):
    """
    >>> from karma.core.labeledmatrix import LabeledMatrix
    >>> lm1 = LabeledMatrix((['b', 'a'], ['x']), np.array([[4], [7]])).sort()
    >>> lm2 = LabeledMatrix((['c'], ['x', 'z', 'y']), np.array([[7, 9, 8]])).sort()
    >>> lm3 = LabeledMatrix((['a', 'd'], ['z', 'w', 'x']), np.array([[1, 5, 20], [-1, 4, -20]])).sort()
    >>> lm = hstack([lm1, lm2, lm3])
    >>> df = lm.to_dense().sort().to_vectorial_dataframe()
    >>> df['col1'] = df['dtype(col1, dtype=np.int)']
    >>> df.preview() #doctest: +NORMALIZE_WHITESPACE
    -------------------------------------------------------------------
    col0 | col1:0 | col1:1 | col1:2 | col1:3 | col1:4 | col1:5 | col1:6
    -------------------------------------------------------------------
    a      7        0        0        0        5        20       1
    b      4        0        0        0        0        0        0
    c      0        7        8        9        0        0        0
    d      0        0        0        0        4        -20      -1
    >>> mat_before = lm.matrix.copy()
    >>> x = lm.sum(axis=0)
    >>> aeq(mat_before, lm.matrix)
    True
    """
    from karma.core.labeledmatrix import LabeledMatrix
    total_rows = list_of_lm[0].row
    row_deco = {}
    for lm in list_of_lm[1:]:
        total_rows = total_rows.union(lm.row)[0]
        row_deco.update(lm.row_deco)
    aligned_matrix_list = []
    for lm in list_of_lm:
        row_index = lm.row.align(total_rows)[1]
        aligned_matrix_list.append(align_along_axis(lm.matrix, row_index, 1, extend=True))
    result = ks_hstack(aligned_matrix_list)
    if dense_output:
        result = result.toarray()
    return LabeledMatrix((total_rows, range(result.shape[1])), result, deco=(row_deco, {}))


def lm_sum(list_of_lm):
    """
    >>> from karma.core.labeledmatrix import LabeledMatrix
    >>> lm1 = LabeledMatrix((['b', 'a'], ['x']), np.array([[4], [7]]))
    >>> lm2 = LabeledMatrix((['c'], ['x', 'z', 'y']), np.array([[7, 9, 8]])).to_sparse()
    >>> lm3 = LabeledMatrix((['a', 'd'], ['z', 'w', 'x']), np.array([[1, 5., 20], [-1, 4, -20]]))
    >>> res = lm_sum([lm1, lm2, lm3]).to_dense().sort()
    >>> res.label
    (['a', 'b', 'c', 'd'], ['w', 'x', 'y', 'z'])
    >>> res.matrix
    array([[  5.,  27.,   0.,   1.],
           [  0.,   4.,   0.,   0.],
           [  0.,   7.,   8.,   9.],
           [  4., -20.,   0.,  -1.]])
    """
    from karma.core.labeledmatrix import LabeledMatrix
    if not list_of_lm:
        raise ValueError('Empty list')
    total_rows, total_columns = list_of_lm[0].row, list_of_lm[0].column
    row_deco, column_deco = list_of_lm[0].deco
    for lm in list_of_lm[1:]:
        total_rows = total_rows.union(lm.row)[0]
        total_columns = total_columns.union(lm.column)[0]
        row_deco.update(lm.row_deco)
        column_deco.update(lm.column_deco)
    result = 0
    for lm in list_of_lm:
        row_index = lm.row.align(total_rows)[1]
        column_index = lm.column.align(total_columns)[1]
        matrix = align_along_axis(lm.matrix, row_index, 1, extend=True)
        matrix = align_along_axis(matrix, column_index, 0, extend=True)
        result = safe_add(matrix, result)
    return LabeledMatrix((total_rows, total_columns),
                         result, deco=(row_deco, column_deco))


def lm_product(list_of_lm):
    """
    >>> from karma.core.labeledmatrix import LabeledMatrix
    >>> lm1 = LabeledMatrix((['b', 'a'], ['x']), np.array([[4], [7]])).to_sparse()
    >>> lm2 = LabeledMatrix((['a'], ['x', 'z', 'y']), np.array([[7, 9, 8]]))
    >>> lm3 = LabeledMatrix((['a', 'd'], ['z', 'w', 'x']), np.array([[1, 5, 20], [-1, 4, -20]]))
    >>> res = lm_product([lm1, lm2, lm3]).to_dense()
    >>> res.label
    (['a'], ['x'])
    >>> aeq(res.matrix, np.array([[980]]))
    True
    """
    from karma.core.labeledmatrix import LabeledMatrix
    if not list_of_lm:
        raise ValueError('Empty list')

    total_rows, total_columns = list_of_lm[0].row, list_of_lm[0].column
    row_deco, column_deco = list_of_lm[0].deco
    for lm in list_of_lm[1:]:
        total_rows = total_rows.intersection(lm.row)[0]
        total_columns = total_columns.intersection(lm.column)[0]

    result = 1
    for lm in list_of_lm:
        row_index = lm.row.align(total_rows)[1]
        column_index = lm.column.align(total_columns)[1]
        matrix = align_along_axis(lm.matrix, row_index, 1, extend=False)
        matrix = align_along_axis(matrix, column_index, 0, extend=False)
        result = safe_multiply(matrix, result)
    return LabeledMatrix((total_rows, total_columns),
                         result, deco=(row_deco, column_deco))

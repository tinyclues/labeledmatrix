#
# Copyright tinyclues, All rights reserved
#
from six.moves import zip
from warnings import warn

import numpy as np
from cytoolz import merge as dict_merge

from cyperf.matrix.karma_sparse import KarmaSparse, DTYPE, ks_hstack, ks_diag, dense_pivot
from cyperf.indexing.indexed_list import reversed_index

from karma.core.column import WriteValuesOnDiskException
from karma.core.utils.collaborative_tools import simple_counter
from karma.learning.matrix_utils import safe_multiply, align_along_axis, safe_add
from karma.types import is_exceptional_mask, generic_ndarray_safe_cast
from six.moves import range

PIVOT_AGGREGATORS_LIST = ['sum', 'min', 'max', '!', 'last', 'first', 'mean', 'std']


def lm_aggregate_pivot(dataframe, key, axis, values=None, aggregator="sum", sparse=True, default=0):
    """
    :param dataframe: DataFrame
    :param key: str columnName corresponding to the index
    :param axis: str columName corresponding to axis for the pivot.
    :param values: str columns on which the aggregator will be used
    :param aggregator: str (min, max, sum, first, last)
    :param sparse: bool returns sparse result
    :param default : float64 : default value for dense pivot.
    :return: LabeledMatrix

    This can be used as routine to compute pivot matrices with associative aggregators (#, sum, min, max, !, last)
    # TODO : iterate over different values/aggregators and to use in df.pivot
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

        >>> lm_aggregate_pivot(d, 'gender', 'csp', 'revenue', 'mean').to_dense()\
            .to_vectorial_dataframe().preview()  #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      90.0     37.5
        2      60.0     32.5

        >>> lm_aggregate_pivot(d, 'gender', 'csp', 'revenue', 'std').to_dense()\
            .to_vectorial_dataframe().preview()  #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      10.0     4.5
        2      0.0      2.5

        >>> lm_aggregate_pivot(d, 'gender', 'csp', 'revenue', 'min', sparse=False)\
            .to_vectorial_dataframe().preview() #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      80      33
        2      60      30

        >>> lm_aggregate_pivot(d, 'gender', 'csp', 'revenue', 'max').to_dense()\
            .to_vectorial_dataframe().preview() #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      100.0    42.0
        2      60.0     35.0

    """
    from karma.core.column import safe_dtype_cast
    aggregator_map = {'sum': 'add',
                      'min': 'min',
                      'max': 'max',
                      '!': 'first',
                      'last': 'last',
                      'first': 'first',
                      'mean': _lm_aggregate_pivot_mean,
                      'std': _lm_aggregate_pivot_std
                      }
    if len(dataframe) == 0:
        raise ValueError('empty dataframe provided to lm aggregate pivot')
    if aggregator not in aggregator_map.keys():
        raise ValueError('aggregator {} does not exist'.format(aggregator))
    aggregator = aggregator_map[aggregator]
    vectorial_cols = set([key, axis, values]).intersection(dataframe.vectorial_column_names)
    if len(vectorial_cols) != 0:
        raise TypeError("Some columns are vectorials not compatible {}".format(vectorial_cols))
    if values is not None:
        data = dataframe[values][:]
        if sparse:
            val = safe_dtype_cast(data, np.float32)
        else:
            if isinstance(data, np.ndarray):
                val = generic_ndarray_safe_cast(data)
            else:
                # case where its not ndarray
                if np.any(is_exceptional_mask(data)):
                    # we NEED na we have to use float, we choose 32
                    try:
                        val = safe_dtype_cast(data, np.float32)
                    except WriteValuesOnDiskException:
                        raise TypeError('Some values to aggregate are not numeric')
                else:
                    val = np.asarray(data)
            if val.dtype not in [np.float64, np.float32, np.int64, np.int32, np.uint8]:
                # the dtype is not numeric
                raise TypeError('Error the dtype {} is not compatible w/ lm pivot'.format(val.dtype))
    else:
        # count aggregator so the values should be integer if not sparse
        if not sparse:
            val = np.ones(len(dataframe), dtype=np.int64)
        else:
            val = np.ones(len(dataframe), dtype=np.float32)

    ri_key = dataframe[key].reversed_index()
    ri_axis = dataframe[axis].reversed_index()

    if callable(aggregator):
        return aggregator(val, ri_key, ri_axis, sparse, default)
    else:
        return _lm_aggregate_pivot(val, ri_key, ri_axis, aggregator, sparse, default)


def _lm_aggregate_pivot(val, ri_key, ri_axis, aggregator, sparse, default=0):
    """
    private method called by lm_aggregate

    :param val: np array
    :param ri_key: tuple (unique values, reverse index) for key
    :param ri_axis: tuple (unique values, reverse index) for axis
    :param aggregator: str (add, min, max, first)
    :param sparse: bool
    :return: LabeledMatrix
    """
    from karma.core.labeledmatrix import LabeledMatrix

    val_key, ind_key = ri_key
    val_axis, ind_axis = ri_axis

    shape = (len(val_key), len(val_axis))
    if sparse:
        matrix = KarmaSparse((val, (ind_key, ind_axis)), shape=shape, format="csr", aggregator=aggregator)
    else:
        matrix = dense_pivot(ind_key, ind_axis, val, shape=shape, aggregator=aggregator, default=default)
    return LabeledMatrix((val_key, val_axis), matrix)


def _lm_aggregate_pivot_mean(val, ri_key, ri_axis, sparse=True, default=0):
    """
    private method called by lm_aggregate_pivot

    :param val: np array
    :param ri_key: tuple (unique values, reverse index) for key
    :param ri_axis: tuple (unique values, reverse index) for axis
    :param sparse: bool
    :return: LabeledMatrix
    """
    # Now sparse matrix ignore nan values :
    # So now when calculating the mean you should divide by the number of element without nan values.
    # To do so we replace 1 by nan so that KarmaSparse will ignore its value.
    ones = np.ones(len(val), dtype=np.float32)
    ones[np.isnan(val)] = np.nan
    lm_sum = _lm_aggregate_pivot(val, ri_key, ri_axis, aggregator='add', sparse=sparse, default=default)
    lm_cardinality = _lm_aggregate_pivot(ones,
                                         ri_key, ri_axis, aggregator='add', sparse=sparse, default=default)
    return lm_sum.divide(lm_cardinality)


def _lm_aggregate_pivot_std(val, ri_key, ri_axis, sparse=True, default=0):
    """
        private method called by lm_aggregate_pivot

        :param val: np array
        :param ri_key: tuple (unique values, reverse index) for key
        :param ri_axis: tuple (unique values, reverse index) for axis
        :param sparse: bool
        :return: LabeledMatrix
        """
    lm_mean = _lm_aggregate_pivot_mean(val, ri_key, ri_axis, sparse=sparse, default=default)
    lm_sum_square = _lm_aggregate_pivot_mean(val ** 2, ri_key, ri_axis, sparse=sparse, default=default)
    lm_var = lm_sum_square - lm_mean.power(2)
    return lm_var.power(0.5)


def aeq(matrix1, matrix2):
    return np.allclose(matrix1, matrix2, rtol=1e-7)


def lm_occurence(val0, val1, dense_output=False):
    """
    >>> lm = lm_occurence([0, 1, 1, 0], ['a', 'b', 'a', 'a'])
    >>> lm.to_flat_dataframe().sort_by('similarity', reverse=True).preview() #doctest: +NORMALIZE_WHITESPACE
    ------------------------
    col0 | col1 | similarity
    ------------------------
    0      a      2.0
    1      a      1.0
    1      b      1.0
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
    array([[1.       , 0.8085077, 0.       ],
           [0.8685415, 0.       , 1.       ],
           [0.       , 0.       , 0.7702151]], dtype=float32)
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
    decayed_series = 2 ** (-(date_arr.max() - date_arr).astype(DTYPE) / half_life)
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


def zipmerge(d1, d2):
    """
    >>> aa = {'a': 3, 'c': 4}
    >>> bb = {'a': 5, 'b': 4}
    >>> zipmerge((aa, bb), (bb, aa))
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
           [  4., -20.,   0.,  -1.]], dtype=float32)
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


def lm_compute_vol_at_cutoff(lm, potential_cutoff):
    """
    Compute the number of lines required to reach a share of the total potential

    :param lm: labeled_matrix. scores_as_labeled_matrix
    :param potential_cutoff: float.
    :return: dict. Key will be a topic, value is the volume of users required to reach a given share of the  total
    potential
    """
    scores_array = lm.to_dense().matrix
    # Get total potential
    total_potential = scores_array.sum(axis=0)

    # Sort scores_array per topic
    scores_ordered_array = np.sort(scores_array, axis=0)[::-1]

    # Compute array of users part of the potential cutoff
    in_potential_cutoff_array = ((scores_ordered_array.cumsum(axis=0) / total_potential) < potential_cutoff)
    # Compute users-in-potential-cutoff volume share
    vol_at_cutoff = in_potential_cutoff_array.sum(axis=0).astype('float32') / scores_array.shape[0]

    return dict(zip(lm.column, vol_at_cutoff))


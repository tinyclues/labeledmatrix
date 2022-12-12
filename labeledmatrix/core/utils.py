import numpy as np
import pandas as pd
from toolz import merge as dict_merge

from cyperf.matrix.karma_sparse import KarmaSparse, DTYPE, ks_hstack, ks_diag, dense_pivot
from cyperf.indexing.indexed_list import reversed_index

PIVOT_AGGREGATORS_LIST = ['sum', 'min', 'max', '!', 'last', 'first', 'mean', 'std']


def lm_aggregate_pivot(dataframe, key, axis, values=None, aggregator="sum", sparse=True):
    """
    :param dataframe: DataFrame
    :param key: str columnName corresponding to the index
    :param axis: str columName corresponding to axis for the pivot.
    :param values: str columns on which the aggregator will be used
    :param aggregator: str (min, max, sum, first, last)
    :param sparse: bool returns sparse result
    :return: LabeledMatrix

    This can be used as routine to compute pivot matrices with associative aggregators (#, sum, min, max, !, last)
    # TODO : iterate over different values/aggregators and to use in df.pivot
    Compare with the example from dataframe.pivot :

        >>> d = pd.DataFrame()
        >>> d['gender'] = ['1', '1', '2', '2', '1', '2', '1']
        >>> d['revenue'] = [100,  42,  60,  30,  80,  35,  33]
        >>> d['csp'] = ['+', '-', '+', '-', '+', '-', '-']
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
                      'first': 'first',
                      'mean': _lm_aggregate_pivot_mean,
                      'std': _lm_aggregate_pivot_std
                      }
    if aggregator not in list(aggregator_map.keys()):
        raise ValueError('aggregator {} does not exist'.format(aggregator))
    aggregator = aggregator_map[aggregator]

    if values is not None:
        val = dataframe[values].values
        if sparse:
            val = val.astype(np.promote_types(val.dtype, np.float32), copy=False)
    else:
        val = np.ones(len(dataframe), dtype=np.float32)

    ri_key = pd.factorize(dataframe[key])
    ri_axis = pd.factorize(dataframe[axis])

    if callable(aggregator):
        return aggregator(val, ri_key, ri_axis, sparse)
    else:
        return _lm_aggregate_pivot(val, ri_key, ri_axis, aggregator, sparse)


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
    val_key, ind_key = ri_key
    val_axis, ind_axis = ri_axis

    shape = (len(ind_key), len(ind_axis))
    if sparse:
        matrix = KarmaSparse((val, (val_key, val_axis)), shape=shape, format="csr", aggregator=aggregator)
    else:
        matrix = dense_pivot(val_key, val_axis, val, shape=shape, aggregator=aggregator, default=default)
    from labeledmatrix.core.labeledmatrix import LabeledMatrix
    return LabeledMatrix((ind_key, ind_axis), matrix)


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


def lm_decayed_pivot_from_dataframe(dataframe, key, axis, axis_deco=None,
                                    date_column='date', half_life=300.):
    """
    Computes a pivot and returns it as a LabeledMatrix from an event DataFrame
    (with a date).
    The values in the cells of the returned LabeledMatrix represents the number
    of occurences of an axis instance for a key. If the event occured before
    the most recent event its weight is reduced.
    >>> data = pd.DataFrame([['abc@fr', 1, 'first', '2015-02-12'],
    ...                      ['jkl@uk', 1, 'first', '2015-03-12'],
    ...                      ['abc@fr', 4, 'fourth', '2015-04-12'],
    ...                      ['bcd@de', 4, 'fourth', '2015-05-12'],
    ...                      ['bcd@de', 4, 'fourth', '2015-06-12'],
    ...                      ['bcd@de', 4, 'fourth', '2012-02-12'],
    ...                      ['bcd@de', 4, 'fourth', '2013-02-12'],
    ...                      ['bcd@de', 4, 'fourth', '2014-02-12'],
    ...                      ['bcd@de', 4, 'fourth', '2015-02-13'],
    ...                      ['abc@fr', 1, 'first', '2015-02-14'],
    ...                      ['abc@fr', 1, 'first', '2015-02-15'],
    ...                      ['abc@fr', 1, 'first', '2015-02-16'],
    ...                      ['abc@fr', 1, 'first', '2015-02-17'],
    ...                      ['abc@fr', 1, 'first', '2015-02-18'],
    ...                      ['bcd@de', 3, 'third', '2015-02-19']],
    ...                     columns=['name', 'cat', 'long_cat', 'date'])
    >>> res1 = lm_decayed_pivot_from_dataframe(data, key='name', axis='cat')
    >>> from labeledmatrix.core.labeledmatrix import LabeledMatrix
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
    deco = axis_deco if axis_deco else {}
    date_arr = np.array(dataframe[date_column][:], dtype='datetime64[D]')
    decayed_series = 2 ** (-(date_arr.max() - date_arr).astype(DTYPE) / half_life)
    key_values, key_ind = reversed_index(dataframe[key].values)
    axis_values, axis_ind = reversed_index(dataframe[axis].values)

    from labeledmatrix.core.labeledmatrix import LabeledMatrix
    pivot_lm = LabeledMatrix((axis_values, key_values),
                             KarmaSparse((decayed_series, (axis_ind, key_ind)), format="csr"),
                             (deco, {}))
    # WARNING is this still usefull ?
    pivot_lm = pivot_lm._minimum(pivot_lm.nonzero_mask())
    return pivot_lm


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


def zipmerge(d1, d2):
    """
    >>> aa = {'a': 3, 'c': 4}
    >>> bb = {'a': 5, 'b': 4}
    >>> zipmerge((aa, bb), (bb, aa))
    ({'a': 5, 'c': 4, 'b': 4}, {'a': 3, 'c': 4, 'b': 4})
    """
    return tuple([dict_merge(x, y) for x, y in zip(d1, d2)])


def lm_compute_volume_at_cutoff(lm, potential_cutoff):
    """
    Compute the number of lines required to reach a share of the total potential

    :param lm: labeled_matrix. scores_as_labeled_matrix
    :param potential_cutoff: float.
    :return: dict. Keys are topics (columns in the input lm), values are the volume of users (rows in the input lm)
                   required to reach a given share of the total potential (sum of the lm values for each column).
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

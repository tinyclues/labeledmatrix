import string
from typing import Union, List, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cyperf.indexing.indexed_list import reversed_index, IndexedList
from cyperf.matrix.karma_sparse import KarmaSparse, ks_diag, dense_pivot

from labeledmatrix.learning.matrix_utils import keep_sparse, safe_multiply, pseudo_element_inverse, safe_add
from labeledmatrix.learning.utils import use_seed


# TODO challenge reversed_index through this file vs pd.factorize


def from_zip_occurrence(input1: Union[List[Any], np.ndarray],
                        input2: Union[List[Any], np.ndarray]):
    """
    TODO doc
    TODO move to unit tests
    >>> from karma.synthetic.basic import basic_dataframe  # FIXME
    >>> from labeledmatrix.core.labeledmatrix import LabeledMatrix
    >>> df = basic_dataframe(100)
    >>> lm = LabeledMatrix(*simple_counter((df['a'].values, df['b'].values)))
    >>> lm.matrix.sum() == 100
    True
    >>> lm.row == df.drop_duplicates('a')['a'].values
    True
    >>> lm.column == df.drop_duplicates('b')['b'].values
    True
    >>> df1 = df.shuffle()
    >>> lm1 = LabeledMatrix(*simple_counter((df1['a'].values, df1['b'].values)))
    >>> lm1.sort() == lm.sort()
    True
    >>> for i in range(len(df)):
    ...     a, b = df['a'][i], df['b'][i]
    ...     assert lm1[a, b] == df.counts(('a', 'b'))[(a,b)]
    """
    if len(input1) != len(input2):
        raise ValueError(f'Inputs must have same length, got {len(input1)} and {len(input2)} instead')
    unique_values1, indices1 = reversed_index(input1)
    unique_values2, indices2 = reversed_index(input2)
    matrix = KarmaSparse((indices1, indices2), shape=(len(unique_values1), len(unique_values2)), format="csr")
    if not keep_sparse(matrix):
        matrix = matrix.toarray()
    return (unique_values1, unique_values2), matrix


def from_random(shape=(4, 3), density=0.5, seed=None, square=False):
    """
    :param shape: the shape of the `LabeledMatrix` (rows, columns)
    :param density: ratio of non-zero elements
    :param seed: seed that could be used to seed the random generator
    :param square: same labels for rows and columns
    """
    alphabet = list(string.ascii_lowercase)

    def _random_word(length):
        while True:
            yield ''.join(np.random.choice(alphabet, size=length))

    def _generate_words(nb, length):
        random_word = _random_word(length)
        words = []
        words_set = set()
        while len(words) < nb:
            word = next(random_word)
            if word not in words_set:
                words_set.add(word)
                words.append(word)
        return words

    with use_seed(seed):
        row = _generate_words(shape[0], 5)
        if square:
            column = row
        else:
            column = _generate_words(shape[1], 5)
        matrix = KarmaSparse(sp.rand(len(row), len(column), density, format="csr", random_state=seed))

    return (row, column), matrix


def from_diagonal(keys, values, keys_deco=None, sparse=True):
    matrix = ks_diag(np.asarray(values))
    if not sparse:
        matrix = matrix.toarray()

    return (keys, keys), matrix, (keys_deco, keys_deco)


def _lm_aggregate_pivot(val, key_indices, key_uniques, col_indices, col_uniques, aggregator, sparse: bool = True,
                        default: float = 0.):
    """
    private method called by lm_aggregate

    :param val: np array
    :param key_indices: reverse index for key
    :param key_uniques: unique values of key
    :param col_indices: reverse index for column
    :param col_uniques: unique values of column
    :param aggregator: str
    :param sparse: bool
    :param default:
    :return: LabeledMatrix
    """
    shape = (len(key_uniques), len(col_uniques))
    if sparse:
        matrix = KarmaSparse((val, (key_indices, col_indices)), shape=shape, format="csr", aggregator=aggregator)
    else:
        matrix = dense_pivot(key_indices, col_indices, val, shape=shape, aggregator=aggregator, default=default)
    return (key_uniques, col_uniques), matrix


def _lm_aggregate_pivot_mean(val, key_indices, key_uniques, col_indices, col_uniques, sparse=True, default=0):
    """
    private method called by lm_aggregate_pivot

    :param val: np array
    :param ri_key: tuple (unique values, reverse index) for key
    :param ri_axis: tuple (unique values, reverse index) for axis
    :param sparse: bool
    :return: LabeledMatrix
    """
    # Sparse matrix ignore nan values :
    # So when calculating the mean you should divide by the number of element without nan values.
    # To do so we replace 1 by nan so that KarmaSparse will ignore those values.
    ones = np.ones(len(val), dtype=np.float32)
    ones[np.isnan(val)] = np.nan

    labels, matrix = _lm_aggregate_pivot(val, key_indices, key_uniques, col_indices, col_uniques,
                                         aggregator='add', sparse=sparse, default=default)
    _, cardinality_matrix = _lm_aggregate_pivot(ones, key_indices, key_uniques, col_indices, col_uniques,
                                                aggregator='add', sparse=sparse, default=default)

    return labels, (matrix, cardinality_matrix)


def _lm_aggregate_pivot_std(val, key_indices, key_uniques, col_indices, col_uniques, sparse=True, default=0):
    """
        private method called by lm_aggregate_pivot

        :param val: np array
        :param ri_key: tuple (unique values, reverse index) for key
        :param ri_axis: tuple (unique values, reverse index) for axis
        :param sparse: bool
        :return: LabeledMatrix
        """
    labels, matrix_mean = _lm_aggregate_pivot_mean(val, key_indices, key_uniques, col_indices, col_uniques,
                                                   sparse=sparse, default=default)
    _, matrix_squares_mean = _lm_aggregate_pivot_mean(val ** 2, key_indices, key_uniques, col_indices, col_uniques,
                                                      sparse=sparse, default=default)
    return labels, (matrix_mean, matrix_squares_mean)


KNOWN_AGGREGATORS = {'sum': 'add', 'min': 'min', 'max': 'max',
                     '!': 'first', 'first': 'first', 'last': 'last',
                     'mean': _lm_aggregate_pivot_mean, 'std': _lm_aggregate_pivot_std}


def from_pivot(dataframe: pd.DataFrame,
               index: Optional[Union[str, List[str]]] = None,
               columns: Optional[Union[str, List[str]]] = None,
               values: Optional[str] = None,
               aggregator: Union[str, Callable] = "sum",
               sparse: bool = True) -> Tuple[Tuple[IndexedList, IndexedList], Union[KarmaSparse, np.ndarray]]:
    """
    Analog of pandas.pivot
        :param dataframe: input Dataframe
        :param index: same as in pandas.pivot: column or list of columns to use as result's index.
                      If None, uses existing index.
        :param columns: same as in pandas pivot: column or list of columns to use as result’s columns.
                      If None, all columns are taken.
        :param values: same as in pandas pivot: optional column's name to take values from.  TODO support list of columns here
        :param aggregator: string from 'sum' (by default), 'min', 'max', 'first', 'last', 'mean', 'std'
        :param sparse: boolean to return sparse result instead of dense one
        :return: Tuple (row labels, columns labels), pivot matrix
    """
    def _reversed_index(cols: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(cols, list) and len(cols) == 1:
            cols = cols[0]
        if isinstance(cols, str):
            return reversed_index(dataframe[cols])[::-1]
        return pd.MultiIndex.from_frame(dataframe[cols]).factorize()

    if index is None:
        key_indices, key_uniques = reversed_index(dataframe.index)[::-1]
    else:
        key_indices, key_uniques = _reversed_index(index)

    if columns is None:
        columns = dataframe.columns
    col_indices, col_uniques = _reversed_index(columns)

    if values is not None:
        val = dataframe[values].values
        # we will use KarmaSparse(float32) implementation of pivot, so casting values to needed type
        val = val.astype(np.promote_types(val.dtype, np.float32), copy=False)
    else:
        val = np.ones(len(dataframe), dtype=np.float32)

    if isinstance(aggregator, str):
        if aggregator not in KNOWN_AGGREGATORS:
            raise ValueError(f'Unknown aggregator `{aggregator}`')
        aggregator = KNOWN_AGGREGATORS[aggregator]

    if callable(aggregator):
        return aggregator(val, key_indices, key_uniques, col_indices, col_uniques, sparse)
    else:
        return _lm_aggregate_pivot(val, key_indices, key_uniques, col_indices, col_uniques, aggregator, sparse)

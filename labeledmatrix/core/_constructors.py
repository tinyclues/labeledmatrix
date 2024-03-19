import string
from typing import Union, List, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cyperf.indexing.indexed_list import reversed_index, IndexedList
from cyperf.matrix.karma_sparse import KarmaSparse, ks_diag, dense_pivot

from labeledmatrix.core.random import use_seed
from labeledmatrix.learning.matrix_utils import keep_sparse


def from_zip_occurrence(row_labels: Union[List[Any], np.ndarray],
                        column_labels: Union[List[Any], np.ndarray]):
    """
    Initializing LabeledMatrix from COO format (potentially corrdinate pairs may be repeated).
    Data will be set to number of co-occurrences of each pair.
    """
    if len(row_labels) != len(column_labels):
        raise ValueError(f'Inputs must have same length, got {len(row_labels)} and {len(column_labels)} instead')
    unique_values1, indices1 = reversed_index(row_labels)
    unique_values2, indices2 = reversed_index(column_labels)
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

    return (keys, keys), matrix, (keys_deco or {}, keys_deco or {})


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
    :param key_indices: reverse index for key
    :param key_uniques: unique values of key
    :param col_indices: reverse index for column
    :param col_uniques: unique values of column
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
    :param key_indices: reverse index for key
    :param key_uniques: unique values of key
    :param col_indices: reverse index for column
    :param col_uniques: unique values of column
    :param sparse: bool
    :return: LabeledMatrix
        """
    labels, matrices_mean = _lm_aggregate_pivot_mean(val, key_indices, key_uniques, col_indices, col_uniques,
                                                     sparse=sparse, default=default)
    _, matrices_squares_mean = _lm_aggregate_pivot_mean(val ** 2, key_indices, key_uniques, col_indices, col_uniques,
                                                        sparse=sparse, default=default)
    return labels, (matrices_mean[0], matrices_squares_mean[0], matrices_mean[1])


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
        :param values: same as in pandas pivot: optional column's name to take values from.
                       TODO support list of columns here
        :param aggregator: string from 'sum' (by default), 'min', 'max', 'first', 'last', 'mean', 'std'
        :param sparse: boolean to return sparse result instead of dense one
        :return: Tuple (row labels, columns labels), pivot matrix
    """
    def _reversed_index(cols: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(cols, list) and len(cols) == 1:
            cols = cols[0]
        if isinstance(cols, str):
            return reversed_index(dataframe[cols].values)[::-1]
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
        if sparse:
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
    return _lm_aggregate_pivot(val, key_indices, key_uniques, col_indices, col_uniques, aggregator, sparse)


def from_ragged_tensor(row_labels, indices, column_labels=None, values=None):
    import tensorflow as tf

    assert row_labels.shape[0] == indices.shape[0]
    if values is not None:
        assert indices.values.shape[0] == values.values.shape[0]
        assert tf.reduce_all(tf.equal(indices.row_splits, values.row_splits))
    else:
        values = tf.ones_like(indices, dtype=tf.float32)
    if column_labels is not None:
        assert tf.reduce_max(indices.values) + 1 == column_labels.shape[0]
    else:
        column_labels, indices_flat = tf.unique(indices.values)
        indices = indices.with_values(indices_flat)

    row_labels, column_labels = np.asarray(row_labels), np.asarray(column_labels)

    matrix = KarmaSparse((values.values.numpy(), indices.values.numpy(), indices.row_splits.numpy()),
                         shape=(row_labels.shape[0], column_labels.shape[0]), format="csr")
    if row_labels.dtype.kind == 'O':
        row_labels = row_labels.astype(str)
    if column_labels.dtype.kind == 'O':
        column_labels = column_labels.astype(str)
    return (row_labels, column_labels), matrix


def from_pyarrow_list_array(row_labels, indices, column_labels=None, values=None):
    import pyarrow as pa

    assert len(row_labels) == len(indices)
    if values is not None:
        assert len(indices.values) == len(values.values)
        assert pa.compute.all(pa.compute.equal(indices.offsets, values.offsets))
    else:
        values = pa.ListArray.from_arrays(indices.offsets, np.ones(len(indices.values), dtype=np.float32))
    if column_labels is not None:
        assert pa.compute.max(indices.values).as_py() + 1 == len(column_labels)
    else:
        dict_array = pa.compute.dictionary_encode(indices.values)
        column_labels, indices = dict_array.dictionary, pa.ListArray.from_arrays(indices.offsets, dict_array.indices)

    row_labels, column_labels = np.asarray(row_labels), np.asarray(column_labels)

    matrix = KarmaSparse((np.asarray(values.values), np.asarray(indices.values), np.asarray(indices.offsets)),
                         shape=(row_labels.shape[0], column_labels.shape[0]), format="csr")
    return (row_labels, column_labels), matrix

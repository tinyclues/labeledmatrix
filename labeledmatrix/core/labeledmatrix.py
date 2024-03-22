from __future__ import annotations

from numbers import Number
import random
from typing import Dict, Any, Union, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix as is_scipysparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from toolz import merge as dict_merge
from toolz.dicttoolz import keymap

from cyperf.clustering.hierarchical import WardTree
from cyperf.indexing.indexed_list import IndexedList
from cyperf.matrix.karma_sparse import ks_diag, KarmaSparse, is_karmasparse
from cyperf.tools import take_indices, sigmoid
from cyperf.tools.getter import apply_python_dict

from labeledmatrix.learning.affinity_propagation import affinity_propagation
from labeledmatrix.learning.co_clustering import co_clustering
from labeledmatrix.learning.hierarchical import clustering_dispatcher
from labeledmatrix.learning.matrix_utils import (align_along_axis, argmax_dispatch, safe_maximum, truncate_by_count,
                                                 pairwise_buddy,
                                                 buddies_matrix, rank_matrix, rank_dispatch, keep_sparse, safe_sum,
                                                 safe_min, safe_minimum, truncate_by_cumulative, truncate_with_cutoff,
                                                 nonzero_mask, number_nonzero, safe_dot,
                                                 truncate_by_budget, complement, pseudo_element_inverse, safe_multiply,
                                                 safe_max, anomaly, safe_add, safe_mean, normalize, truncated_dot)
from labeledmatrix.learning.nmf import nmf, nmf_fold
from labeledmatrix.learning.randomize_svd import randomized_svd
from labeledmatrix.learning.sparse_tail_clustering import sparse_tail_clustering
from labeledmatrix.learning.tail_clustering import tail_clustering

from ._constructors import from_zip_occurrence, from_random, from_diagonal, from_pivot, from_ragged_tensor, \
    from_pyarrow_list_array
from ._exporters import to_vectorial_dataframe, to_flat_dataframe, to_list_dataframe
from .random import UseSeed
from .utils import co_axis, aeq, zipmerge, is_integer

__all__ = ['LabeledMatrix', 'LabeledMatrixException']


class LabeledMatrixException(Exception):
    pass


class LabeledMatrix:
    @classmethod
    def from_zip_occurrence(cls, row_labels, column_labels) -> LabeledMatrix:
        """
        Initializing LabeledMatrix from COO format (potentially corrdinate pairs may be repeated).
        Data will be set to number of co-occurrences of each pair.
        >>> lm = LabeledMatrix.from_zip_occurrence([0, 1, 1, 0], ['a', 'b', 'a', 'a'])
        >>> lm.to_flat_dataframe().sort_values('similarity', ascending=False) #doctest: +NORMALIZE_WHITESPACE
        ------------------------
        col0 | col1 | similarity
        ------------------------
        0      a      2.0
        1      a      1.0
        1      b      1.0
        """
        return cls(*from_zip_occurrence(row_labels, column_labels))

    @classmethod
    def from_dict(cls, dictionary: Dict[Any, Any]) -> LabeledMatrix:
        """
        Initializing LabeledMatrix from COO format defined as dict: keys correspond to row labels and values to columns
        Values are equal number of co-occurrences of each pair (0 or 1)
        >>> my_dict = {'a': 'x', 'c': 'y', 'b': 'y'}
        >>> lm = LabeledMatrix.from_dict({'a': 'x', 'c': 'y', 'b': 'y'})
        >>> lm.to_flat_dataframe().sort_values('col0', ascending=False)
        ------------------------
        col0 | col1 | similarity
        ------------------------
        a      x      1.0
        b      y      1.0
        c      y      1.0
        """
        return cls.from_zip_occurrence(list(dictionary.keys()), list(dictionary.values()))

    @classmethod
    def from_random(cls, shape=(4, 3), density=0.5, sparse=True, seed=None, square=False) -> LabeledMatrix:
        """
        Returns a randomized LabeledMatrix.

        :param shape: the shape of the `LabeledMatrix` (rows, columns)
        :param density: ratio of non-zero elements
        :param sparse: determines if the inner container will be sparse or not
        :param seed: seed that could be used to seed the random generator
        :param square: same labels for rows and columns
        :return: A randomized LabeledMatrix

        Exemples: ::

            >>> lm = LabeledMatrix.from_random(seed=12).sort()
            >>> lm.row
            ['lemal', 'piuzo', 'pivqv', 'wthra']
            >>> lm.column
            ['fkgbs', 'gcqvk', 'vteol']
            >>> lm.matrix.toarray().shape
            (4, 3)
            >>> LabeledMatrix.from_random(shape=(10, 10)).nnz()
            50
        """
        lm = cls(*from_random(shape, density, seed, square))
        if sparse:
            return lm.to_sparse()
        return lm.to_dense()

    @classmethod
    def from_diagonal(cls, labels, values, keys_deco=None, sparse=True) -> LabeledMatrix:
        """
        Initializing diagonal LabeledMatrix with row labels equal to column labels
        :param labels: labels for both rows and columns
        :param values: values to be put on diagonal, must have the same shape as corresponding labels
        :param keys_deco: dict with deco for labels
        :param sparse: boolean whether to use sparse backend, True by default
        >>> lm = LabeledMatrix.from_diagonal(['b', 'c'], [3, 50000000]).sort()
        >>> lm.label
        (['b', 'c'], ['b', 'c'])
        >>> aeq(lm.matrix, np.array([[3, 0], [0, 50000000]]))
        True
        >>> lm.matrix.format
        'csr'
        """
        return cls(*from_diagonal(labels, values, keys_deco, sparse))

    @classmethod
    def from_pivot(cls, dataframe: pd.DataFrame,
                   index: Optional[Union[str, List[str]]] = None,
                   columns: Optional[Union[str, List[str]]] = None,
                   values: Optional[str] = None, aggregator: str = 'sum',
                   sparse: bool = True) -> LabeledMatrix:
        """
        Analog of pandas.pivot
        :param dataframe: input Dataframe
        :param index: same as in pandas.pivot: column or list of columns to use as result's index.
                      If None, uses existing index.
        :param columns: same as in pandas pivot: column or list of columns to use as result’s columns.
                      If None, all columns are taken.
        :param values: as in pandas pivot: optional column's name to take values from.
                       Currently can take only one column's value
        :param aggregator: string from 'sum' (by default), 'min', 'max', 'first', 'last', 'mean', 'std'
        :param sparse: boolean to return sparse result instead of dense one
        :return: LabeledMatrix object

        >>> d = pd.DataFrame()
        >>> d['gender'] = ['1', '1', '2', '2', '1', '2', '1']
        >>> d['revenue'] = [100,  42,  60,  30,  80,  35,  33]
        >>> d['csp'] = ['+', '-', '+', '-', '+', '-', '-']
        >>> LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue',
        ...                          sparse=False).to_vectorial_dataframe()  #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      180.0    75.0
        2      60.0     65.0

        >>> LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'mean',
        ...                          sparse=False).to_vectorial_dataframe()  #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      90.0     37.5
        2      60.0     32.5

        >>> LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'std',
        ...                          sparse=False).to_vectorial_dataframe()  #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      10.0     4.5
        2      0.0      2.5

        >>> LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'min',
        ...                          sparse=False).to_vectorial_dataframe() #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      80.0     33.0
        2      60.0     30.0

        >>> LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'max',
        ...                          sparse=False).to_vectorial_dataframe() #doctest: +NORMALIZE_WHITESPACE
        ----------------------
        col0 | col1:+ | col1:-
        ----------------------
        1      100.0    42.0
        2      60.0     35.0
        """
        res = from_pivot(dataframe, index, columns, values, aggregator, sparse)
        if aggregator == 'mean':
            labels, (matrix, cardinality_matrix) = res
            lm, lm_cardinality = cls(labels, matrix), cls(labels, cardinality_matrix)
            return lm._divide(lm_cardinality)
        if aggregator == 'std':
            labels, (matrix_sum, matrix_squares_sum, cardinality_matrix) = res
            lm_sum, lm_squares_sum = cls(labels, matrix_sum), cls(labels, matrix_squares_sum)
            lm_cardinality = cls(labels, cardinality_matrix)
            return (lm_squares_sum._divide(lm_cardinality)._add(-lm_sum._divide(lm_cardinality).power(2))).power(0.5)
        return cls(*res)

    @classmethod
    def from_ragged_tensor(cls, row_labels: Union[list, np.ndarray], indices,
                           column_labels: Optional[Union[list, np.ndarray]] = None, values=None) -> LabeledMatrix:
        """
        Constructing sparse LabeledMatrix from tensorflow.RaggedTensor with column labels/indices
        Similar to from_pyarrow_list_array
        :param row_labels: list/np.array/tf.Tensor with row labels
        :param indices: tf.RaggedTensor with non-zero indices by line: it can be integer indices or column labels
        :param column_labels: optional list/np.array/tf.Tensor with column labels
                              by default indices' values will be used as column labels
        :param values: optional tf.RaggedTensor with matrix values, should be aligned with indices.
                       by default values will be equal to 1
        # TODO fill doctests outputs
        >>> import tensorflow as tf
        >>> tensor = tf.RaggedTensor.from_row_splits(['a', 'b', 'a'], [0, 1, 3, 3])
        >>> tensor
        >>> LabeledMatrix.from_ragged_tensor(['r0', 'r1', 'r2'], tensor)
        >>> matrix_values = tf.RaggedTensor.from_row_splits([3, 8, 2], [0, 1, 3, 3])
        >>> LabeledMatrix.from_ragged_tensor(['r0', 'r1', 'r2'], tensor, values=matrix_values)
        >>> tensor = tf.RaggedTensor.from_row_splits([0, 1, 0], [0, 1, 3, 3])
        >>> LabeledMatrix.from_ragged_tensor(['r0', 'r1', 'r2'], tensor, column_labels=['x', 'y'])
        """
        return cls(*from_ragged_tensor(row_labels, indices, column_labels, values))

    @classmethod
    def from_pyarrow_list_array(cls, row_labels: Union[list, np.ndarray], indices,
                                column_labels: Optional[Union[list, np.ndarray]] = None, values=None) -> LabeledMatrix:
        """
        Constructing sparse LabeledMatrix from pyarrow.ListArray with column labels/indices
        Similar to from_ragged_tensor
        :param row_labels: list/np.array/pa.Array with row labels
        :param indices: pa.ListArray with non-zero indices by line: it can be integer indices or column labels
        :param column_labels: optional list/np.array/pa.Array with column labels
                              by default indices' values will be used as column labels
        :param values: optional pa.ListArray with matrix values, should be aligned with indices.
                       by default values will be equal to 1
        # TODO fill doctests outputs
        >>> import pyarrow as pa
        >>> pa_array = pa.ListArray.from_pandas([['a'], ['b', 'a'], []])
        >>> pa_array
        >>> LabeledMatrix.from_pyarrow_list_array(['r0', 'r1', 'r2'], pa_array)
        >>> matrix_values = pa.ListArray.from_pandas([[3], [8, 2], []])
        >>> LabeledMatrix.from_pyarrow_list_array(['r0', 'r1', 'r2'], pa_array, values=matrix_values)
        >>> pa_array = pa.ListArray.from_pandas([[0], [1, 0], []])
        >>> LabeledMatrix.from_pyarrow_list_array(['r0', 'r1', 'r2'], pa_array, column_labels=['x', 'y'])
        """
        return cls(*from_pyarrow_list_array(row_labels, indices, column_labels, values))

    @classmethod
    def from_xarray(cls, values) -> LabeledMatrix:
        """
        Constructing dense LabeledMatrix from 2-dimensional xarray.DataArray
        Dims' coordinates are directly put into row and column labels
        :param values: xr.DataArray
        # TODO fill doctests outputs
        >>> import xarray as xr
        >>> array = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=("x", "y"), coords={"x": ['a', 'b']})
        >>> LabeledMatrix.from_xarray(array)
        """
        assert values.ndim == 2
        return cls(tuple(list(values.get_index(dim)) for dim in values.dims), values.data)

    def to_vectorial_dataframe(self, deco=False) -> pd.DataFrame:
        """
        Export LabeledMatrix to pd.DataFrame as matrix:
            self.row and self.column gives dataframe's indexes (row and column)
            column values are equal to self.matrix
        Names of index entries can be changed using decoration stored in self.deco (deco=True option)
        >>> mat = np.array([[4, 5, 0],
        ...                 [0, 3, 0],
        ...                 [0, 0, 0],
        ...                 [1, 0, 0]])
        >>> lm = LabeledMatrix((['a', 'b', 'c', 'd'], ['x', 'y', 'z']), mat, deco=({'a': 'AA'}, {'y': 'YY'}))
        >>> lm.to_vectorial_dataframe() #doctest: +NORMALIZE_WHITESPACE
           x  y  z
        a  4  5  0
        b  0  3  0
        c  0  0  0
        d  1  0  0
        >>> lm.to_vectorial_dataframe(deco=True) #doctest: +NORMALIZE_WHITESPACE
            x  YY  z
        AA  4   5  0
        b   0   3  0
        c   0   0  0
        d   1   0  0
        >>> lm.to_sparse().to_vectorial_dataframe() #doctest: +NORMALIZE_WHITESPACE
             x    y    z
        a  4.0  5.0  0.0
        b  0.0  3.0  0.0
        c  0.0  0.0  0.0
        d  1.0  0.0  0.0
        """
        return to_vectorial_dataframe(self, deco)

    def to_flat_dataframe(self, row='col0', col='col1', dist='similarity', **kwargs) -> pd.DataFrame:
        """
        Return a DataFrame with three columns (row, col and dist) from LabeledMatrix.

        kwargs:
            - deco_row: name of the decoration column for row
            - deco_col: name of the decoration column for col

        >>> mat = np.array([[4, 5, 0],
        ...                 [0, 3, 0],
        ...                 [0, 0, 2],
        ...                 [1, 0, 0]])
        >>> lm = LabeledMatrix((['a', 'b', 'c', 'd'], ['x', 'y', 'z']), mat)
        >>> lm.set_deco(row_deco={'b': 'B', 'c': 'C', 'a': 'A'})
        >>> lm.to_sparse().to_flat_dataframe().sort_values('similarity', ascending=False)
          col0 col1  similarity deco_col0
        1    a    y         5.0         A
        0    a    x         4.0         A
        2    b    y         3.0         B
        3    c    z         2.0         C
        4    d    x         1.0      None
        """
        return to_flat_dataframe(self, row, col, dist, **kwargs)

    def to_list_dataframe(self, col='col', prefix='list_of_', exclude=False) -> pd.DataFrame:
        """
        Return a DataFrame with columns col, list_of_col.
        For each row, sort the non zero values and return the column labels as list_of_col
        >>> mat = np.array([[4, 0, 0, 0],
        ...                 [5, 4, 1, 0],
        ...                 [0, 1, 4, 5],
        ...                 [1, 0, 5, 4]])
        >>> lm = LabeledMatrix(2*[['a', 'b', 'c', 'd']], mat)
        >>> lm.to_list_dataframe(exclude=True) #doctest: +NORMALIZE_WHITESPACE
          col list_of_col
        0   b      [a, c]
        1   c      [d, b]
        2   d      [c, a]
        >>> lm.to_sparse().to_list_dataframe(exclude=True)
        ...             #doctest: +NORMALIZE_WHITESPACE
          col list_of_col
        0   b      [a, c]
        1   c      [d, b]
        2   d      [c, a]
        >>> lm = LabeledMatrix([['a', 'b', 'c', 'd'], ['w', 'x', 'y', 'z']], mat)
        >>> lm.to_list_dataframe(exclude=True) #doctest: +NORMALIZE_WHITESPACE
          col list_of_col
        0   a         [w]
        1   b   [w, x, y]
        2   c   [z, y, x]
        3   d   [y, z, w]
        """
        return to_list_dataframe(self, col, prefix, exclude)

    def to_ragged_tensor(self, return_nonzero_mask):
        """
        :param return_nonzero_mask: boolean if we should return values too
        :return:
        """
        # FIXME
        # FIXME how to package ? tuple of row: Tensor, column: Tensor, indices: RaggedTensor, values: RaggedTensor ?

    def to_pyarrow(self, return_nonzero_mask):
        """
        :param return_nonzero_mask: boolean if we should return values too
        :return:
        """
        # FIXME
        # FIXME same question as for to_ragged_tensor

    def to_xarray(self, row_dimension: Optional[str] = None, column_dimension: Optional[str] = None):
        """
        Transforming LabeledMatrix into 2-dimensional xarray.DataArray
        row labels and column labels are put to corresponding dimensions' coordinates
        :param row_dimension: dimension name for matrix rows, be default "row"
        :param column_dimension: dimension name for matrix columns, be default "column"
        >>> lm  = LabeledMatrix((['a', 'b'], ['x', 'y', 'z']), [[1, 2, 3], [4, 5, 6]])
        >>> lm.to_xarray('i', 'j')  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <xarray.DataArray (i: 2, j: 3)> Size: ...
        array([[1, 2, 3],
               [4, 5, 6]])
        Coordinates:
          * i        (i) <U1 ... 'a' 'b'
          * j        (j) <U1 ... 'x' 'y' 'z'
        """
        import xarray as xr
        attrs = {}
        if row_dimension is None:
            row_dimension = 'row'
        if column_dimension is None:
            column_dimension = 'column'
        if self.row_deco:
            attrs[f'{row_dimension} deco'] = self.row_deco
        if self.column_deco:
            attrs[f'{column_dimension} deco'] = self.column_deco
        return xr.DataArray(self.to_dense().matrix, dims=(row_dimension, column_dimension),
                            coords={row_dimension: list(self.row), column_dimension: list(self.column)}, attrs=attrs)

    def to_tensorflow_model(self, input_name, output_name, default='zero', coordinates=None):
        """
        :param input_name:
        :param output_name:
        :param default:
        :param coordinates:
        :return:
        """
        # FIXME: TF lookup + embeddings

    def __init__(self,
                 row_column_labels: Tuple[Union[List[Any], np.ndarray, IndexedList],
                                          Union[List[Any], np.ndarray, IndexedList]],
                 matrix: Union[np.ndarray, KarmaSparse],
                 deco=({}, {})):
        """
        :param row_column_labels: tuple with 2 entries: row labels and column labels.
                                  Labels must be unique (separately for each list)
        :param matrix: 2d numpy array or KarmaSparse matrix with values
        :param deco: tuple decoration dictionaries with additional labels for rows and columns respectively
        >>> matrix = np.array([[4, 6, 5], [7, 9, 8], [1, 3, 2]])
        >>> row, column = ['b', 'c', 'a'], ['x', 'z', 'y']
        >>> lm = LabeledMatrix((row, column), matrix.copy())
        >>> lm #doctest: +NORMALIZE_WHITESPACE
        <LabeledMatrix with properties :
         * dense numpy
         * dtype int64
         * dimension (3, 3)
         * number of non-zero elements 9
         * density of non-zero elements 1.0
         * min element 1
         * max element 9
         * sum elements 45>
        >>> lm.label == (row, column)
        True
        >>> lm.set_deco(row_deco={'b': 'B', 'c': 'C', 'a': 'A'},
        ...             column_deco={'x': 'X', 'z': 'Z', 'y': 'Y'})

        >>> aeq(lm.matrix, matrix)
        True
        >>> lm.row == row
        True
        >>> lm.column == column
        True

        >>> lm = LabeledMatrix((row, ['x', 'z', 'x']), matrix.copy()) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: List should have unique values

        >>> LabeledMatrix((row, np.array(['x', 'y', 'z'])), matrix.copy()).column
        ['x', 'y', 'z']
        """
        (row, column) = row_column_labels  # FIXME
        self.check_format((row, column), matrix)
        if is_scipysparse(matrix):
            matrix = KarmaSparse(matrix)
        self.is_sparse = is_karmasparse(matrix)
        if not isinstance(row, IndexedList):
            if isinstance(row, np.ndarray):
                row = row.tolist()
            row = IndexedList(list(row) if not isinstance(row, list) else row)
        if not isinstance(column, IndexedList):
            if isinstance(column, np.ndarray):
                column = column.tolist()
            column = IndexedList(list(column) if not isinstance(column, list) else column)
        self.matrix = matrix
        self.row, self.column = row, column
        self.label = (self.row, self.column)
        self.deco = deco
        self.row_deco, self.column_deco = self.deco

        self._nnz = None

    def __eq__(self, other):
        return isinstance(other, LabeledMatrix) \
               and (self.row == other.row) and (self.column == other.column) \
               and aeq(self.matrix, other.matrix)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        >>> matrix = np.array([[4, 6, 5], [7, 9, 8], [1, 3, 2]])
        >>> row, column = ['b', 'c', 'a'], ['x', 'z', 'y']
        >>> lm = LabeledMatrix((row, column), matrix.copy())
        >>> lm.shape
        (3, 3)
        """
        return self.matrix.shape

    @property
    def dtype(self) -> np.dtype:
        """
        Returns underlying matrix dtype. Note that sparse matrices will be casted to float32
        >>> matrix = np.array([[4, 6, 0], [0, 9, 8], [0, 0, 2]])
        >>> row, column = ['b', 'c', 'a'], ['x', 'z', 'y']
        >>> LabeledMatrix((row, column), matrix.astype(np.int16)).dtype
        dtype('int16')
        >>> LabeledMatrix((row, column), matrix.astype(np.float32)).dtype
        dtype('float32')
        >>> LabeledMatrix((row, column), matrix).to_sparse().dtype
        dtype('float32')
        """
        return self.matrix.dtype

    @staticmethod
    def check_format(row_column_labels: Tuple[Union[List[Any], np.ndarray, IndexedList],
                                              Union[List[Any], np.ndarray, IndexedList]],
                     matrix: Union[np.ndarray, KarmaSparse]):
        """
        Used to check a number of assertion on the content of a LabeledMatrix.

        :param row_column_labels: tuple (row and column) labels, labels should be unique and of the correct
                                  shape, with respect to the given matrix
        :param matrix: a numpy or scipy two dimensional array
        :raise: LabeledMatrixException

        Exemple: ::

            >>> LabeledMatrix.check_format((['a','b','c'], ['1', '2']),
            ...                            [[0, 1, 0],[1, 0, 1]])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            labeled_matrix_fixed.LabeledMatrixException: Unacceptable matrix type: <class 'list'>
            >>> LabeledMatrix.check_format((['a','b','c'], ['1', '2']),
            ...                            np.array([2]))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            labeled_matrix_fixed.LabeledMatrixException: Wrong number of dimension: 1
            >>> LabeledMatrix.check_format((['a','b','c'], ['1']),
            ...                            np.array([[0, 1, 0],[1, 0, 1]]))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            labeled_matrix_fixed.LabeledMatrixException: Number of rows 3 should corresponds to matrix.shape[0]=2
            >>> LabeledMatrix.check_format((['a', 'c'], ['1', '2']),
            ...                            np.array([[0, 1, 0],[1, 0, 1]]))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            labeled_matrix_fixed.LabeledMatrixException: Number of columns 2 should corresponds to matrix.shape[1]=3
            >>> LabeledMatrix.check_format((['a', 'b'], ['1', '2', '3']),
            ...                            np.array([[0, 1, 0], [1, 0, 1]]))  # doctest: +ELLIPSIS

        """
        (row, column) = row_column_labels
        if not (is_scipysparse(matrix) or is_karmasparse(matrix) or isinstance(matrix, np.ndarray)):
            raise LabeledMatrixException(f'Unacceptable matrix type: {type(matrix)}')
        if matrix.ndim != 2:
            raise LabeledMatrixException(f'Wrong number of dimension: {matrix.ndim}')
        # get default labels
        if len(row) != matrix.shape[0]:
            raise LabeledMatrixException(
                f'Number of rows {len(row)} should corresponds to matrix.shape[0]={matrix.shape[0]}')
        if len(column) != matrix.shape[1]:
            raise LabeledMatrixException(
                f'Number of columns {len(column)} should corresponds to matrix.shape[1]={matrix.shape[1]}')

    def __repr__(self):
        properties = ['']

        if self.is_sparse:
            properties += [f'sparse of format {self.matrix.format}']
        else:
            properties += ['dense numpy']
        properties += [
            f'dtype {self.matrix.dtype}',
            f'dimension {self.matrix.shape}',
            f'number of non-zero elements {self.nnz()}',
            f'density of non-zero elements {np.round(self.density(), 7)}',
        ]
        if self.nnz():
            properties += [
                f'min element {self.min(axis=None)}',
                f'max element {self.max(axis=None)}',
                f'sum elements {self.sum(axis=None)}'
            ]
        properties = '\n * '.join(properties)
        return f'<LabeledMatrix with properties :{properties}>'

    def __getitem__(self, labels: Tuple[Any, Any]):
        """
        Access to matrix item by its labels. Missing labels will return 0 value.
        :param labels: tuple with requested row label and column label
        >>> lm = LabeledMatrix((['a', 'b', 'c'], ['a', 'b', 'c']) , np.arange(9).reshape(3, 3))
        >>> lm #doctest: +NORMALIZE_WHITESPACE
        <LabeledMatrix with properties :
         * dense numpy
         * dtype int64
         * dimension (3, 3)
         * number of non-zero elements 8
         * density of non-zero elements 0.8888889
         * min element 0
         * max element 8
         * sum elements 36>
        >>> lm.matrix #doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        >>> lm['a', 'c']
        2
        """
        if len(labels) != 2:
            raise ValueError('Dimension of input should be 2.')
        x, y = labels
        if (x in self.row) and (y in self.column):
            return self.matrix[self.row.index(x), self.column.index(y)]
        return 0

    @property
    def is_square(self) -> bool:
        """
        Check if matrix is square, meaning row labels and column labels are the same
        >>> lm1 = LabeledMatrix((['b', 'a'], ['d', 'b']), np.arange(4).reshape(2, 2))
        >>> lm1.is_square
        False
        >>> lm1.symmetrize_label().is_square
        True
        """
        return self.row == self.column

    @property
    def _has_sorted_row(self) -> bool:
        return self.row.is_sorted()

    @property
    def _has_sorted_column(self) -> bool:
        return self.column.is_sorted()

    def nnz(self) -> int:
        """
        nnz as non-zero! Returns the number of non-zero elements contained in the matrix
        """
        if self._nnz is None:
            self._nnz = number_nonzero(self.matrix)
        return self._nnz

    def density(self) -> float:
        return self.nnz() * 1. / np.product(self.matrix.shape)

    def copy(self) -> LabeledMatrix:
        """
        Copy underlying matrix and return new LabeledMatrix object with same labels and deco
        """
        return LabeledMatrix(self.label, self.matrix.copy(), deco=self.deco)

    def rand_row(self):
        return random.choice(self.row)

    def rand_column(self):
        return random.choice(self.column)

    def rand_label(self):
        """
        Chooses a random label pair of row and column from the matrix
        """
        return self.rand_row(), self.rand_column()

    def set_deco(self, row_deco=None, column_deco=None):
        def _check_dict(deco):
            if not isinstance(deco, dict):
                raise ValueError('Decoration should be a dict')
            return deco

        if row_deco is not None:
            self.row_deco = _check_dict(row_deco)
        if column_deco is not None:
            self.column_deco = _check_dict(column_deco)
        self.deco = (self.row_deco, self.column_deco)

    def to_dense(self) -> LabeledMatrix:
        """
        Transform matrix backend into numpy array and return new LabeledMatrix object
        Return same object if it is already dense
        >>> import scipy.sparse as sp
        >>> lm = LabeledMatrix([['b', 'c'], ['a', 'd', 'e']],
        ...                    sp.rand(2, 3, 0.5, format='csr'))
        >>> isinstance(lm.to_dense().matrix, np.ndarray)
        True
        >>> lm.to_dense().to_sparse() == lm
        True
        """
        if self.is_sparse:
            return LabeledMatrix(self.label, self.matrix.toarray(), deco=self.deco)
        return self

    def to_sparse(self) -> LabeledMatrix:
        """
        Transform matrix backend into KarmaSparse matrix and return new LabeledMatrix object
        Return same object if it is already sparse
        >>> lm = LabeledMatrix([['b', 'c'], ['a', 'd']], np.array([[1,2], [3,0]]))
        >>> isinstance(lm.to_sparse(), KarmaSparse)
        True
        >>> lm.to_sparse().to_dense() == lm
        True
        """
        if self.is_sparse:
            return self
        return LabeledMatrix(self.label, KarmaSparse(self.matrix), deco=self.deco)

    def to_optimal_format(self) -> LabeledMatrix:
        """
        Casts to sparse if density < min_density, casts to dense if density > min_density
        Returns a view if the condition is already met or in case of equality

        min_density is a constant currently set to 20%
        """
        if keep_sparse(self.matrix):
            return self.to_sparse()
        return self.to_dense()

    def align(self, other: LabeledMatrix,
              axes=((0, 0, False), (1, 1, False))) -> Tuple[LabeledMatrix, LabeledMatrix]:
        """
        Aligns two LabeledMatrix according to provided axis
        For instance, if axes = [axis] and axis = (0, 1, True) means that
            * self.column (axis[0]=0) will be aligned with other.column (axis[1]=1)
            * by taking the union of the labels (axis[2]=True)

        The third parameter in each element of axes` corresponds to outer/inner join in SQL logic.
            * if axis[2] == False, the default value, the intersection (inner) of labels will be taken
            * if axis[2] == True, the union (outer) of labels will be taken
            * if axis[2] == None, the labels of `other` matrix will be taken

        >>> lm1 = LabeledMatrix((['a', 'b'], ['c', 'd']), np.arange(4).reshape(2, 2))
        >>> lm2 = LabeledMatrix((['a', 'c'], ['b', 'd']), - np.arange(4).reshape(2, 2))
        >>> lm1_inner, lm2_inner = lm1.align(lm2)
        >>> lm1_inner.label == (['a'], ['d'])
        True
        >>> lm1_inner.label == lm2_inner.label
        True
        >>> aeq(lm1_inner.matrix, np.array([[1]]))
        True
        >>> aeq(lm2_inner.matrix, np.array([[-1]]))
        True
        >>> lm1_outer, lm2_outer = lm1.align(lm2, axes=[(0, 0, True), (1, 1, True)])
        >>> lm1_outer.label
        (['a', 'b', 'c'], ['c', 'd', 'b'])
        >>> lm2_outer.label == lm1_outer.label
        True
        >>> aeq(lm1_outer.matrix, np.array([[0, 1, 0], [2, 3, 0], [0, 0, 0]]))
        True
        >>> aeq(lm2_outer.matrix, np.array([[ 0, -1, 0], [0,  0,  0], [ 0, -3, -2]]))
        True

        if the third parameter is None, this means the following:
            * self LM will be aligned to have exactly the same shape as other (along provided axis)
            * other LM will not be changed

        >>> lm1_outer_bis, lm2_bis = lm1.align(lm2, axes=[(0, 0, None), (1, 1, None)])
        >>> lm1_outer_bis.label == lm2.label
        True
        >>> aeq(lm1_outer_bis.matrix, np.array([[0, 1],[0, 0]]))
        True
        >>> lm12bis, lm2_bis = lm1.align(lm2, axes=[(0, 0, None)])
        >>> lm12bis.column == lm2_bis.column == lm2.column
        True
        """
        self_copy, other_copy = self, other
        for (ax_self, ax_other, extend) in axes:
            label_self = self_copy.label[co_axis(ax_self)]
            label_other = other_copy.label[co_axis(ax_other)]
            if extend is None:  # string alignment
                new, self_arg, other_arg = label_self.align(label_other)
            elif extend:  # extension
                new, self_arg, other_arg = label_self.union(label_other)
            else:  # restriction
                new, self_arg, other_arg = label_self.intersection(label_other)
            if len(new) == 0:
                raise LabeledMatrixException('align has returned an empty labeled matrix')

            mat_self = align_along_axis(self_copy.matrix, self_arg, ax_self,
                                        extend is None or extend)
            mat_other = align_along_axis(other_copy.matrix, other_arg, ax_other,
                                         extend is None or extend)
            if ax_self == 0:
                self_copy = LabeledMatrix((self_copy.row, new), mat_self)
            else:
                self_copy = LabeledMatrix((new, self_copy.column), mat_self)
            if extend is not None:
                if ax_other == 0:
                    other_copy = LabeledMatrix((other_copy.row, new), mat_other)
                else:
                    other_copy = LabeledMatrix((new, other_copy.column), mat_other)
        self_copy.set_deco(*self.deco)
        other_copy.set_deco(*other.deco)
        return self_copy, other_copy

    def extend_row(self,
                   rows: Union[List[Any], np.ndarray, IndexedList],
                   deco: Optional[dict] = None) -> LabeledMatrix:
        """
        Extend row labels with additional labels, corresponding rows will be set to 0
        :param rows: list of additional labels
        :param deco: optional decoration dictionary for additional labels
        >>> lm = LabeledMatrix(2*[['a', 'c']], np.array([[1,2], [3,4]]))
        >>> aeq(lm.extend_row(['b']).sort().matrix, np.array([[1, 2],[0, 0],[3, 4]]))
        True
        >>> lm.extend_row(['b']).sort().label
        (['a', 'b', 'c'], ['a', 'c'])
        >>> aeq(lm.to_sparse().extend_row(['b']).sort().to_dense().matrix,
        ...     np.array([[1, 2],[0, 0],[3, 4]]))
        True
        >>> lm.to_sparse().extend_row(['b']).sort().label
        (['a', 'b', 'c'], ['a', 'c'])
        >>> lm.to_sparse().extend_row(['b']).sort().row
        ['a', 'b', 'c']
        """
        all_rows, arg_row, _ = self.row.union(rows)
        return LabeledMatrix((all_rows, self.column),
                             align_along_axis(self.matrix, arg_row, 1, True),
                             deco=(dict_merge(self.row_deco, deco or {}), self.column_deco))

    def extend_column(self,
                      columns: Union[List[Any], np.ndarray, IndexedList],
                      deco: Optional[dict] = None) -> LabeledMatrix:
        """
        Extend column labels with additional labels, corresponding columns will be set to 0
        :param columns: list of additional labels
        :param deco: optional decoration dictionary for additional labels
        >>> lm = LabeledMatrix(2*[['a', 'c']], np.array([[1,2], [3,4]]))
        >>> lm.extend_column(['b']).sort().matrix
        array([[1, 0, 2],
               [3, 0, 4]])
        >>> lm.extend_column(['b']).sort().label
        (['a', 'c'], ['a', 'b', 'c'])
        >>> aeq(lm.to_sparse().extend_column(['b']).sort().to_dense().matrix,
        ...     np.array([[1, 0, 2],[3, 0, 4]]))
        True
        >>> lm.to_sparse().extend_column(['b']).sort().label
        (['a', 'c'], ['a', 'b', 'c'])
        >>> lm.to_sparse().extend_column(['b']).sort().column
        ['a', 'b', 'c']
        """
        all_columns, arg_column, _ = self.column.union(columns)
        return LabeledMatrix((self.row, all_columns),
                             align_along_axis(self.matrix, arg_column, 0, True),
                             deco=(self.row_deco, dict_merge(self.column_deco, deco or {})))

    def extend(self,
               label: Tuple[Union[List[Any], np.ndarray, IndexedList], Union[List[Any], np.ndarray, IndexedList]],
               deco: Tuple[dict, dict] = ({}, {})) -> LabeledMatrix:
        """
        Extend labels with additional labels, corresponding rows & columns will be set to 0
        :param label: tuple with additional labels for rows and columns respectively
        :param deco: optional tuple with decoration dictionaries for additional row and column labels respectively
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,4]]))
        >>> lm.extend(2*[['a']]).matrix
        array([[0, 0, 0],
               [0, 1, 2],
               [0, 3, 4]])
        >>> lm.to_sparse().extend((['a'], ['a'])).matrix.toarray().astype(np.int64)
        array([[0, 0, 0],
               [0, 1, 2],
               [0, 3, 4]])
        >>> lm.to_sparse().extend(2*[['a', 'd', 'c']]).matrix.toarray().astype(np.int64)
        array([[1, 2, 0, 0],
               [3, 4, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
        >>> lm.to_sparse().extend(2*[['a', 'd', 'c']]).label
        (['b', 'c', 'a', 'd'], ['b', 'c', 'a', 'd'])
        >>> lm = LabeledMatrix([['b', 'c'], ['x']], np.array([[1], [3]]))
        >>> lm.extend((['a', 'b'], ['y'])).matrix
        array([[1, 0],
               [3, 0],
               [0, 0]])
        >>> lm.extend((['a', 'b'], ['y'])).label
        (['b', 'c', 'a'], ['x', 'y'])
        >>> lm.to_sparse().extend((['a', 'b'], ['y'])).matrix.toarray().astype(np.int64)
        array([[1, 0],
               [3, 0],
               [0, 0]])
        """
        if label[0] and label[1]:
            return self.extend_column(label[1], deco[1]).extend_row(label[0], deco[0])
        if label[0]:
            return self.extend_row(label[0], deco[0])
        if label[1]:
            return self.extend_column(label[1], deco[1])
        return self.copy()

    def restrict_row(self, rows: Union[List[Any], np.ndarray, IndexedList]) -> LabeledMatrix:
        """
        Return a submatrix restricted to a list of rows
        """
        common_rows, arg_row, _ = self.row.intersection(rows)
        if not common_rows:
            raise LabeledMatrixException('restrict has returned an empty labeled matrix')
        return LabeledMatrix((common_rows, self.column),
                             align_along_axis(self.matrix, arg_row, 1),
                             deco=self.deco)

    def restrict_column(self, columns: Union[List[Any], np.ndarray, IndexedList]) -> LabeledMatrix:
        """
        Return a submatrix restricted to a list of columns
        """
        common_columns, arg_column, _ = self.column.intersection(columns)
        if not common_columns:
            raise LabeledMatrixException('restrict has returned an empty labeled matrix')
        return LabeledMatrix((self.row, common_columns),
                             align_along_axis(self.matrix, arg_column, 0),
                             deco=self.deco)

    def restrict(self, label: Tuple[Union[List[Any], np.ndarray, IndexedList],
                                    Union[List[Any], np.ndarray, IndexedList]]) -> LabeledMatrix:
        """
        Return a submatrix: keep only the rows and columns with the given labels.
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,4]]))
        >>> rlm = lm.extend(2*[['a', 'd', 'c']]).restrict(2*[['c', 'a']])
        >>> rlm.sort().matrix
        array([[0, 0],
               [0, 4]])
        >>> rlm.sort().label
        (['a', 'c'], ['a', 'c'])
        >>> rlm = lm.to_sparse().extend(2*[['a', 'd', 'c']]).restrict(2*[['d','b','c']])
        >>> rlm.sort().matrix.toarray().astype(np.int64)
        array([[1, 2, 0],
               [3, 4, 0],
               [0, 0, 0]])
        >>> rlm.sort().label
        (['b', 'c', 'd'], ['b', 'c', 'd'])
        >>> lm = LabeledMatrix([['b', 'c'], ['x']], np.array([[1], [3]]))
        >>> aeq(lm.restrict((['b', 'a'], None)).matrix, np.array([[1]]))
        True
        >>> lm.restrict((['b', 'a'], None)).label
        (['b'], ['x'])
        >>> aeq(lm.transpose().to_sparse().restrict((None, ['b', 'a'])).matrix, np.array([[1]]))
        True
        """
        if label[0] is not None and label[1] is not None:
            return self.restrict_row(label[0]).restrict_column(label[1])
        if label[0] is None:
            return self.restrict_column(label[1])
        if label[1] is None:
            return self.restrict_row(label[0])
        return self.copy()

    def exclude_row(self, rows: Union[List[Any], np.ndarray, IndexedList]) -> LabeledMatrix:
        """
        Return a submatrix without a list of rows
        """
        keep_row, arg_row = self.row.difference(rows)
        if not keep_row:
            raise LabeledMatrixException('exclude has returned an empty labeled matrix')
        return LabeledMatrix((keep_row, self.column),
                             align_along_axis(self.matrix, arg_row, 1),
                             deco=self.deco)

    def exclude_column(self, columns: Union[List[Any], np.ndarray, IndexedList]) -> LabeledMatrix:
        """
        Return a submatrix without a list of columns
        """
        keep_column, arg_column = self.column.difference(columns)
        if not keep_column:
            raise LabeledMatrixException('exclude has returned an empty labeled matrix')
        return LabeledMatrix((self.row, keep_column),
                             align_along_axis(self.matrix, arg_column, 0),
                             deco=self.deco)

    def exclude(self, label: Tuple[Union[List[Any], np.ndarray, IndexedList],
                                   Union[List[Any], np.ndarray, IndexedList]]) -> LabeledMatrix:
        """
        Return a submatrix without the rows and columns with the given labels.
        >>> lm = LabeledMatrix([['b', 'c'], ['x', 'y', 'z']], np.arange(6).reshape(2,3))
        >>> lm.exclude((None, ['y', 'h'])).label
        (['b', 'c'], ['x', 'z'])
        >>> lm.exclude((None, ['y', 'h'])).matrix
        array([[0, 2],
               [3, 5]])
        >>> lm.exclude((['b', 'h'], ['z'])).label
        (['c'], ['x', 'y'])
        >>> lm.exclude((['b', 'h'], ['z'])).matrix
        array([[3, 4]])
        """
        if label[0] is not None and label[1] is not None:
            return self.exclude_row(label[0]).exclude_column(label[1])
        if label[0] is None:
            return self.exclude_column(label[1])
        if label[1] is None:
            return self.exclude_row(label[0])
        return self.copy()

    def symmetrize_label(self, restrict: bool = False) -> LabeledMatrix:
        """
        Return the square matrix with same labels for rows and columns
        :param restrict: boolean whether to take the intersection of row and column labels for resulting matrix (True)
                         or their union (False)
        >>> lm1 = LabeledMatrix((['b', 'a'], ['d', 'b']),
        ...                     np.arange(4).reshape(2, 2))
        >>> lm_square = lm1.symmetrize_label()
        >>> lm_square.label
        (['b', 'a', 'd'], ['b', 'a', 'd'])
        >>> aeq(lm_square.matrix, np.array([[1, 0, 0],[3, 0, 2],[0, 0, 0]]))
        True
        >>> lm2 = LabeledMatrix((['b', 'a', 'd'], ['d', 'b', 'c']),
        ...                     np.arange(9).reshape(3, 3))
        >>> lm_square = lm2.symmetrize_label(restrict=True)
        >>> lm_square.label
        (['b', 'd'], ['b', 'd'])
        >>> aeq(lm_square.matrix, np.array([[1, 0],[7, 6]]))
        True
        """
        if self.is_square:
            return self
        if restrict:
            lm = self.restrict_row(self.column).restrict_column(self.row)
        else:
            lm = self.extend_row(self.column).extend_column(self.row)
        return lm.align(lm, [(0, 1, None)])[0]

    def _take_on_row(self, indices=None) -> LabeledMatrix:
        """
        >>> lm = LabeledMatrix((['b', 'c'], ['y', 'x', 'z']), np.arange(6).reshape(2, 3))
        >>> lm.label[0]
        ['b', 'c']
        >>> lm._take_on_row([1, 0]).label[0]
        ['c', 'b']
        """
        if indices is None:
            return self
        row = take_indices(self.row, indices)
        return LabeledMatrix((row, self.column), self.matrix[indices], self.deco)

    def _take_on_column(self, indices=None):
        """
        >>> lm = LabeledMatrix((['b', 'c'], ['y', 'x', 'z']), np.arange(6).reshape(2, 3))
        >>> lm.label[1]
        ['y', 'x', 'z']
        >>> lm._take_on_column([2, 0, 1]).label[1]
        ['z', 'y', 'x']
        """
        if indices is None:
            return self
        column = take_indices(self.column, indices)
        return LabeledMatrix((self.row, column), self.matrix[:, indices], self.deco)

    def sort_row(self) -> LabeledMatrix:
        """
        Return a new matrix with row labels sorted in increasing order
        """
        if not self._has_sorted_row:
            row, argsort_row = self.row.sorted()
            return LabeledMatrix((row, self.column), self.matrix[argsort_row], self.deco)
        return self

    def sort_column(self) -> LabeledMatrix:
        """
        Return a new matrix with column labels sorted in increasing order
        """
        if not self._has_sorted_column:
            column, argsort_column = self.column.sorted()
            return LabeledMatrix((self.row, column), self.matrix[:, argsort_column],
                                 self.deco)
        return self

    def sort_by_deco(self) -> LabeledMatrix:
        """
        Order row and column labels by the order given by the decoration dictionaries
        >>> lm = LabeledMatrix((['b', 'c'], ['y', 'x', 'z']), np.arange(6).reshape(2, 3),
        ...                    deco=({'b': 'f', 'c': 'a'}, {'x': 'k', 'z': 'l'}))
        >>> lm.sort().label
        (['b', 'c'], ['x', 'y', 'z'])
        >>> lm.sort_by_deco().label
        (['c', 'b'], ['x', 'z', 'y'])
        """
        if len(self.row_deco) != 0:
            _, row_indices = IndexedList([self.row_deco.get(row_, row_) for row_ in self.row.list]).sorted()
        else:
            row_indices = slice(None)

        if len(self.column_deco) != 0:
            _, column_indices = IndexedList([self.column_deco.get(col_, col_) for col_ in self.column.list]).sorted()
        else:
            column_indices = slice(None)

        return self._take_on_row(row_indices)._take_on_column(column_indices)

    def sort(self) -> LabeledMatrix:
        """
        Return a new matrix with both row and column labels sorted in increasing order
        >>> lm = LabeledMatrix((['b', 'c'], ['y', 'x', 'z']), np.arange(6).reshape(2, 3))
        >>> lm['c', 'x'] == lm.sort()['c', 'x'] == 4
        True
        >>> lm.label
        (['b', 'c'], ['y', 'x', 'z'])
        >>> lm.sort().label
        (['b', 'c'], ['x', 'y', 'z'])
        >>> (lm.sort() - lm).nnz()
        0
        """
        return self.sort_row().sort_column()

    def rename_row(self, prefix='', suffix='', mapping=None) -> LabeledMatrix:
        """
        Returns LabeledMatrix with changed row names
        Args:
            prefix: prefix to add to row names
            suffix: suffix to add to row names
            mapping: translate names with respect to a dict
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1, 0], [3, 4]]), deco=({'b': 'b_deco'}, {}))
        >>> renamed_lm = lm.rename_row(prefix='a_', suffix='_z', mapping={'b': 'h'})
        >>> list(renamed_lm.row)
        ['a_h_z', 'a_c_z']
        >>> renamed_lm.row_deco
        {'a_h_z': 'b_deco'}
        """
        if mapping is None:
            mapping = {}
        rows = [f'{prefix}{row}{suffix}' for row in apply_python_dict(mapping, self.row, None, keep_same=True)]
        row_deco = keymap(dict(zip(self.row, rows)).get, self.row_deco)
        return LabeledMatrix((rows, self.column), self.matrix, (row_deco, self.column_deco))

    def rename_column(self, prefix='', suffix='', mapping=None) -> LabeledMatrix:
        """
        Returns LabeledMatrix with changed column names
        Args:
            prefix: prefix to add to column names
            suffix: suffix to add to column names
            mapping: translate names with respect to a dict
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1, 0], [3, 4]]), deco=({}, {'x': 'x_deco'}))
        >>> renamed_lm = lm.rename_column(prefix='a_', suffix='_z', mapping={'x': 'h'})
        >>> list(renamed_lm.column)
        ['a_h_z', 'a_y_z']
        >>> renamed_lm.column_deco
        {'a_h_z': 'x_deco'}
        """
        return self.transpose().rename_row(prefix, suffix, mapping).transpose()

    def sample_rows(self, p: Union[int, float]) -> LabeledMatrix:
        """
        Take a subsample of matrix rows of a given size
        :param p: either float between 0 and 1 indicating proportion of rows to keep or integer number of rows
        >>> lm = LabeledMatrix((['b', 'c', 'd'], ['x', 'y', 'z']),
        ...                    np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]]))
        >>> lm.sample_rows(12) #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Sample larger than population or is negative
        >>> lm.sample_rows(0.2) #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        labeled_matrix_fixed.LabeledMatrixException: p should be > 0, currently is 0.
        >>> sample1 = lm.sample_rows(2)
        >>> sample1.matrix.shape
        (2, 3)
        >>> sample2 = lm.sample_rows(0.8)
        >>> sample2.matrix.shape
        (2, 3)
        """
        if p <= 0:
            raise LabeledMatrixException(f'p should be > 0, currently is {p}.')
        if p < 1:
            p *= len(self.row)
        p = int(p)
        return self.restrict_row(random.sample(self.row.list, p))

    def sample_columns(self, p: Union[int, float]) -> LabeledMatrix:
        """
        Take a subsample of matrix columns of a given size
        :param p: either float between 0 and 1 indicating proportion of columns to keep or integer number of columns
        >>> lm = LabeledMatrix((['b', 'c', 'd'], ['x', 'y', 'z']),
        ...                    np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]]))
        >>> lm.sample_columns(12) #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Sample larger than population or is negative
        >>> lm.sample_rows(0.2) #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        labeled_matrix_fixed.LabeledMatrixException: p should be > 0, currently is 0.
        >>> sample1 = lm.sample_columns(2)
        >>> sample1.matrix.shape
        (3, 2)
        >>> sample2 = lm.sample_columns(0.4)
        >>> sample2.matrix.shape
        (3, 1)
        """
        if p <= 0:
            raise ValueError(f'p should be > 0, currently is {p}.')
        if p < 1:
            p *= len(self.column)
        p = int(p)
        return self.restrict_column(random.sample(self.column.list, p))

    def transpose(self) -> LabeledMatrix:
        """
        Matrix transposition
        """
        return LabeledMatrix((self.column, self.row), self.matrix.transpose(),
                             deco=(self.deco[1], self.deco[0]))

    def zeros(self, force_sparse: bool = False) -> LabeledMatrix:
        """
        Return a matrix of the same shape and same labels but filled with zeros
        :param force_sparse: boolean whether return sparse zeros matrix even if current matrix is dense
        >>> lm = LabeledMatrix([['b', 'c']] * 2, np.array([[1,2], [3,4]]))
        >>> aeq(lm.zeros().matrix, np.zeros((2,2)))
        True
        >>> lm.zeros().label == lm.label
        True
        >>> aeq(lm.to_sparse().zeros().matrix.toarray(), np.zeros((2,2)))
        True
        """
        if self.is_sparse or force_sparse:
            matrix_zeros = KarmaSparse(self.matrix.shape)
        else:
            matrix_zeros = np.zeros(self.matrix.shape, dtype=self.matrix.dtype)
        return LabeledMatrix(self.label, matrix_zeros, deco=self.deco)

    def without_zeros(self, axis: Optional[int] = None, min_nonzero: int = 1) -> LabeledMatrix:
        """
        Return a matrix without rows and/or columns with all values strictly smaller than min_nonzero
        :param axis: axis along which to apply the operation
                     0 keep all rows, remove zero columns
                     1 remove zero rows, keep all columns
                     None remove both zero rows and zero columns
        :param min_nonzero: threshold on values to consider them non-zero, default 1
        >>> lm = LabeledMatrix((['b', 'c', 'd'], ['x', 'y', 'z']),
        ...                    np.array([[1,0, 0], [0, 0, 0], [0, 1, 0]]))
        >>> lm.matrix
        array([[1, 0, 0],
               [0, 0, 0],
               [0, 1, 0]])
        >>> lm.without_zeros(axis=0).label
        (['b', 'c', 'd'], ['x', 'y'])
        >>> lm.without_zeros(axis=1).label
        (['b', 'd'], ['x', 'y', 'z'])
        >>> lm.to_sparse().without_zeros(axis=1).label
        (['b', 'd'], ['x', 'y', 'z'])
        >>> lm.to_sparse().without_zeros(axis=0).label
        (['b', 'c', 'd'], ['x', 'y'])
        """
        if axis is None:
            return self.without_zeros(axis=1, min_nonzero=min_nonzero) \
                .without_zeros(axis=0, min_nonzero=min_nonzero)
        index = np.where(safe_sum(nonzero_mask(self.matrix), axis=axis) >= min_nonzero)[0]
        if len(index) == 0:
            raise LabeledMatrixException('without_zeros has returned an empty labeled matrix')
        if len(index) == self.matrix.shape[co_axis(axis)]:
            return self
        to_keep = self.label[co_axis(axis)].select(index)
        if axis == 1:
            return LabeledMatrix((to_keep, self.column), self.matrix[index], deco=self.deco)
        return LabeledMatrix((self.row, to_keep), self.matrix[:, index], deco=self.deco)

    def nonzero_mask(self) -> LabeledMatrix:
        """
        Replace each non-zero entry by 1
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[1,0], [3,0]]))
        >>> aeq(lm.nonzero_mask().matrix, np.array([[1, 0], [1, 0]]))
        True
        >>> aeq(lm.to_sparse().nonzero_mask().matrix, np.array([[1, 0], [1, 0]]))
        True
        """
        return LabeledMatrix(self.label, nonzero_mask(self.matrix), deco=self.deco)

    def rank(self, axis: int = 1, reverse: bool = False) -> LabeledMatrix:
        """
        Return ranks of entries along given axis in ascending order (descending if reverse is True)
        >>> mat = np.array([[1, 2, 4, 3],
        ...                 [0, 0, 0, 0],
        ...                 [2, 1, 3, 4]])
        >>> lm = LabeledMatrix([range(3), range(4)], mat)
        >>> lm.rank().matrix
        array([[0, 1, 3, 2],
               [0, 1, 2, 3],
               [1, 0, 2, 3]])
        >>> lm.rank(reverse=True).matrix
        array([[3, 2, 0, 1],
               [3, 2, 1, 0],
               [2, 3, 1, 0]])
        """
        return LabeledMatrix(self.label, rank_matrix(self.matrix, axis=axis, reverse=reverse), deco=self.deco)

    def diagonal(self) -> LabeledMatrix:
        """
        Keep only diagonal values (where row label is equal to column label)
        Currently works only on square matrix (row and column labels are the same)
        >>> lm = LabeledMatrix((['b', 'c'], ['b', 'c']), np.array([[4, 6], [7, 9]]))
        >>> aeq(lm.to_sparse().diagonal().matrix, np.array([[4, 0],[0, 9]]))
        True
        >>> lm.diagonal().matrix
        array([[4, 0],
               [0, 9]])
        """
        if not self.is_square:
            raise LabeledMatrixException('diagonal() works only on square matrices.')
        if self.is_sparse:
            diag_matrix = ks_diag(self.matrix.diagonal(), format=self.matrix.format)
        else:
            diag_matrix = np.diagflat(self.matrix.diagonal())
        return LabeledMatrix(self.label, diag_matrix, deco=self.deco)

    def without_diagonal(self) -> LabeledMatrix:
        """
        Replace diagonal values (where row label is equal to column label) by 0
        >>> lm = LabeledMatrix((['b', 'c'], ['b', 'c']), np.array([[4, 6], [7, 9]]))
        >>> lm.to_sparse().without_diagonal().matrix.toarray()
        array([[0., 6.],
               [7., 0.]], dtype=float32)
        >>> lm = LabeledMatrix((['b', 'c'], ['b', 'c', 'd']), np.array([[4, 6, 8], [7, 9, 3]]))
        >>> lm.without_diagonal().matrix
        array([[0, 6, 8],
               [7, 0, 3]])
        """
        if len(self.row) <= len(self.column):
            diag_indices = [(i, self.column.index(row)) for i, row in enumerate(self.row)
                            if row in self.column]
        else:
            diag_indices = [(self.row.index(column), i) for i, column in enumerate(self.column)
                            if column in self.row]
        if not diag_indices:
            return self

        diag_indices = np.array(diag_indices, dtype=np.int32).transpose()
        return LabeledMatrix(self.label,
                             complement(self.matrix, (diag_indices[0], diag_indices[1])),
                             deco=self.deco)

    def without_mask(self, mask: LabeledMatrix) -> LabeledMatrix:
        """
        Take all labels with non-zero values from mask, replace values with same labels in self by 0
        >>> lm = LabeledMatrix((['b', 'c'], ['b', 'c']), np.array([[4, 6], [7, 9]]))
        >>> lm.without_mask(LabeledMatrix((['b'], ['b']), np.array([[1]]))).matrix
        array([[0, 6],
               [7, 9]])
        >>> lm.without_mask(LabeledMatrix((['b'], ['c']), np.array([[1]]))).matrix
        array([[4, 0],
               [7, 9]])
        >>> lm = LabeledMatrix((['b', 'c'], ['b', 'c', 'd']), np.array([[4, 6, 8], [7, 9, 3]]))
        >>> lm.without_mask(LabeledMatrix((['c'], ['b', 'c', 'd']), np.array([[1, 0, 1]]))).matrix
        array([[4, 6, 8],
               [0, 9, 0]])
        >>> lm.without_mask(LabeledMatrix((['c'], ['b', 'c', 'd']), np.array([[1, 0, 1]])).to_sparse()).matrix
        array([[4, 6, 8],
               [0, 9, 0]])
        """
        mask = mask.align(self, axes=[(0, 0, None), (1, 1, None)])[0]
        return LabeledMatrix(self.label,
                             complement(self.matrix, mask.matrix),
                             deco=self.deco)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def _add(self, other: LabeledMatrix) -> LabeledMatrix:
        """
        Point-wise sum, assuming that labels are the same
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1,2], [3,4]]))
        >>> lm._add(lm).matrix
        array([[2, 4],
               [6, 8]])
        >>> aeq(lm.to_sparse()._add(lm.to_sparse()).matrix.toarray(), np.array([[2, 4], [6, 8]]))
        True
        >>> aeq(lm.to_sparse()._add(lm).matrix.toarray(), np.array([[2, 4], [6, 8]]))
        True
        >>> aeq(lm._add(lm.to_sparse()).matrix.toarray(), np.array([[2, 4], [6, 8]]))
        True
        """
        return LabeledMatrix(self.label, safe_add(self.matrix, other.matrix),
                             deco=zipmerge(self.deco, other.deco))

    def add(self, other: LabeledMatrix) -> LabeledMatrix:
        """
        >>> lm1 = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,4]]))
        >>> lm2 = LabeledMatrix(2*[['a', 'c']], np.array([[1,2], [3,4]]))
        >>> slm = lm2.add(lm1)
        >>> slm.label
        (['a', 'c', 'b'], ['a', 'c', 'b'])
        >>> slm.matrix
        array([[1, 2, 0],
               [3, 8, 3],
               [0, 2, 1]])
        >>> np.all(slm.matrix == lm2.to_sparse().add(lm1.to_sparse()).matrix.toarray())
        True
        >>> slm1 = lm1.add(lm2).sort()
        >>> (slm1 - slm).nnz()
        0
        """
        aligned_self, aligned_other = self.align(other, axes=[(0, 0, True), (1, 1, True)])
        return aligned_self._add(aligned_other)

    def _scalar_add(self, scalar) -> LabeledMatrix:
        """
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1,0], [0,4]]))
        >>> aeq(lm._scalar_add(-2.5).matrix, np.array([[-1.5,  0],[ 0,  1.5]]))
        True
        >>> aeq(lm._scalar_add(-2).matrix, np.array([[-1,  0],[ 0,  2]]))
        True
        >>> aeq(lm.to_sparse()._scalar_add(-2).matrix, np.array([[-1,  0],[ 0,  2]]))
        True
        >>> aeq(lm.to_sparse()._scalar_add(-2.).matrix, np.array([[-1,  0],[ 0,  2]]))
        True
        >>> lm._scalar_add(-2.).matrix
        array([[-1.,  0.],
               [ 0.,  2.]])
        """
        if self.is_sparse:
            return LabeledMatrix(self.label, self.matrix + scalar, deco=self.deco)
        matrix = self.matrix.copy().astype(max(self.matrix.dtype, type(scalar)))
        matrix[matrix.nonzero()] += scalar
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def __add__(self, other):
        """
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,2], [3,4]]))
        >>> (lm.to_sparse() + 2 * lm) == 3 * lm
        True
        >>> np.all((lm - 1).matrix == np.arange(4).reshape(2,2))
        True
        >>> 1 - lm == - (lm - 1)
        True
        >>> np.all((lm - 1).matrix == lm.matrix - 1)
        True
        >>> (1 - lm).matrix
        array([[ 0, -1],
               [-2, -3]])
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,0], [3,4]]))
        >>> 1 + lm == lm + 1
        True
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,2], [3,4]]))
        >>> lm += lm
        >>> lm.matrix
        array([[2, 4],
               [6, 8]])
        """
        if np.isscalar(other):
            return self._scalar_add(other)
        if isinstance(other, LabeledMatrix):
            return self.add(other)
        raise ValueError(f'type of `other` is not understood : {type(other)}')

    __radd__ = __add__

    __iadd__ = __add__

    def __sub__(self, other):
        """
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,2], [3,4]]))
        >>> np.all((lm - lm).matrix == 0)
        True
        >>> np.all((2 * lm - lm - lm).matrix == 0)
        True
        >>> np.all((-lm + lm).matrix == 0)
        True
        """
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__sub__(other).__neg__()

    def _multiply(self, other: LabeledMatrix) -> LabeledMatrix:
        """
        Point-wise multiply, assuming that labels are the same
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1,2], [3,4]]))
        >>> aeq(lm._multiply(lm).matrix, np.array([[ 1,  4], [ 9, 16]]))
        True
        >>> aeq(lm.to_sparse()._multiply(lm.to_sparse()).matrix.toarray(), np.array([[1, 4], [9, 16]]))
        True
        >>> aeq(lm.to_sparse()._multiply(lm).matrix.toarray(), np.array([[1, 4], [9, 16]]))
        True
        >>> aeq(lm._multiply(lm.to_sparse()).matrix.toarray(), np.array([[1, 4], [9, 16]]))
        True
        """
        matrix = safe_multiply(self.matrix, other.matrix)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def multiply(self, other: LabeledMatrix) -> LabeledMatrix:
        """
        >>> lm1 = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,4]]))
        >>> lm2 = LabeledMatrix(2*[['a', 'c']], np.array([[1,2], [3,4]]))
        >>> mlm = lm2.multiply(lm1)
        >>> mlm.matrix
        array([[16]])
        >>> mlm.label
        (['c'], ['c'])
        >>> np.all(mlm.matrix ==
        ...       lm2.to_sparse().multiply(lm1.to_sparse()).matrix.toarray())
        True
        """
        aligned_self, aligned_other = self.align(other, axes=[(0, 0, False), (1, 1, False)])
        return aligned_self._multiply(aligned_other)

    def _scalar_multiply(self, scalar) -> LabeledMatrix:
        """
        >>> lm1 = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,4]]))
        >>> (lm1 * 2).matrix
        array([[2, 4],
               [6, 8]])
        """
        return LabeledMatrix(self.label, self.matrix * scalar, deco=self.deco)

    def __mul__(self, other):
        """
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,2], [3,4]]))
        >>> aeq((2 * lm.to_sparse()).matrix, np.array([[2, 4], [6, 8]]))
        True
        >>> aeq((2 * lm.to_sparse() * lm).matrix, np.array([[ 2,  8], [18, 32]]))
        True
        """
        if np.isscalar(other):
            return self._scalar_multiply(other)
        if isinstance(other, LabeledMatrix):
            return self.multiply(other)
        raise ValueError(f'type of `other` is not understood : {type(other)}')

    def __rmul__(self, other):
        return self.__mul__(other)

    def _divide(self, other: LabeledMatrix) -> LabeledMatrix:
        """
        element-wise division, assuming labels are the same
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1,0], [3,4]]))
        >>> d = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[0.5,2], [3,4]]))
        >>> d._divide(lm).matrix
        array([[0.5, 0. ],
               [1. , 1. ]])
        >>> d.to_sparse()._divide(lm.to_sparse()).matrix.toarray()
        array([[0.5, 0. ],
               [1. , 1. ]], dtype=float32)
        >>> d._divide(lm.to_sparse()).matrix.toarray()
        array([[0.5, 0. ],
               [1. , 1. ]], dtype=float32)
        >>> d.to_sparse()._divide(lm).matrix.toarray()
        array([[0.5, 0. ],
               [1. , 1. ]], dtype=float32)
        """
        matrix = safe_multiply(self.matrix, pseudo_element_inverse(other.matrix))
        return LabeledMatrix(self.label, matrix,
                             deco=zipmerge(self.deco, other.deco))

    def divide(self, other: LabeledMatrix) -> LabeledMatrix:
        """
        >>> lm1 = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,2.]]))
        >>> lm2 = LabeledMatrix(2*[['a', 'c']], np.array([[1,2], [3,4]]))
        >>> mlm = lm1.divide(lm2)
        >>> mlm.matrix
        array([[0.5]])
        >>> mlm.label
        (['c'], ['c'])
        >>> np.all(mlm.matrix == lm1.to_sparse().divide(lm2.to_sparse()).matrix.toarray())
        True
        """
        aligned_self, aligned_other = self.align(other, axes=[(0, 0, False), (1, 1, False)])
        return aligned_self._divide(aligned_other)

    def _scalar_divide(self, scalar):
        """
        >>> lm1 = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,4]]))
        >>> lm1._scalar_divide(2.0).matrix
        array([[0.5, 1. ],
               [1.5, 2. ]])
        >>> aeq(lm1._scalar_divide(2).matrix, np.array([[ 0.5,  1. ],[ 1.5,  2.]]))
        True
        >>> aeq(lm1.to_sparse()._scalar_divide(2).matrix, np.array([[ 0.5,  1. ], [ 1.5, 2.]]))
        True
        """
        return LabeledMatrix(self.label, self.matrix / float(scalar), deco=self.deco)

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        """
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,2], [3,4]]))
        >>> aeq((lm.to_sparse() / lm).matrix, np.ones((2,2)))
        True
        >>> (lm.to_sparse() / 2.0).matrix.toarray()
        array([[0.5, 1. ],
               [1.5, 2. ]], dtype=float32)
        >>> aeq((lm.to_sparse() / lm.to_sparse()).matrix, np.ones((2,2)))
        True
        >>> (lm / lm).matrix
        array([[1., 1.],
               [1., 1.]])
        >>> (lm / 2.0).matrix
        array([[0.5, 1. ],
               [1.5, 2. ]])
        """
        if np.isscalar(other):
            return self._scalar_divide(other)
        if isinstance(other, LabeledMatrix):
            return self.divide(other)
        raise ValueError(f'type of `other` is not understood : {type(other)}')

    def _dot(self,
             other: LabeledMatrix,
             top: Optional[int] = None,
             mask: Optional[LabeledMatrix] = None) -> LabeledMatrix:
        """
        Calculate matrix dot operation: resulting row labels come from left matrix and column labels from right one
        :param other: right matrix in dot product
        :param top: take only top largest elements when calculating scalar product between two vectors
        :param mask: LabeledMatrix indicating which labels to keep in the result
        >>> lm = LabeledMatrix((['x', 'y'], ['a', 'b']), np.array([[1,2], [3,4]]))
        >>> lm._dot(lm).matrix
        array([[ 7, 10],
               [15, 22]])
        >>> np.all(lm._dot(lm).matrix ==
        ...        lm.to_sparse()._dot(lm.to_sparse()).matrix.toarray())
        True
        """
        if top is not None:
            return LabeledMatrix((self.row, other.column),
                                 truncated_dot(self.matrix, other.matrix, nb=top),
                                 deco=(self.row_deco, other.column_deco))
        if mask is not None:
            return LabeledMatrix((self.row, other.column),
                                 safe_dot(self.matrix, other.matrix, mask.matrix),
                                 deco=(self.row_deco, other.column_deco))
        return LabeledMatrix((self.row, other.column),
                             safe_dot(self.matrix, other.matrix),
                             deco=(self.row_deco, other.column_deco))

    def dot(self,
            other: LabeledMatrix,
            top: Optional[int] = None,
            mask: Optional[LabeledMatrix] = None) -> LabeledMatrix:
        """
        Calculate matrix dot operation: resulting row labels come from left matrix and column labels from right one
        If left matrix column labels and right matrix row labels are distinct, then we restrict so that they match.
        :param other: right matrix in dot product
        :param top: take only top largest elements when calculating scalar product between two vectors
        :param mask: LabeledMatrix indicating which labels to keep in the result
        >>> lm1 = LabeledMatrix((['x', 'y'], ['a', 'b', 'c']), np.arange(6).reshape(2,3))
        >>> lm2 = LabeledMatrix((['a', 'b', 'd'], ['w']), np.arange(3).reshape(3,1))
        >>> lm1.set_deco(row_deco={'x': 'X', 'y': 'Y'})
        >>> lm2.set_deco(column_deco={'w': 'W'})
        >>> slm = lm1.dot(lm2)
        >>> slm.matrix
        array([[1],
               [4]])
        >>> slm.label
        (['x', 'y'], ['w'])
        >>> np.all(slm.matrix == lm1.to_sparse().dot(lm2.to_sparse()).matrix.toarray())
        True
        >>> aeq(lm1.to_sparse().dot(lm2).matrix, np.array([[1],[4]]))
        True

        >>> lm1.dot(lm2).display_reco('x')  #doctest: +NORMALIZE_WHITESPACE
          entry_key  nb_nonzero  total_score  min_score deco
        0         x           1            1          1    X
        ------------------------------------------------------------
          reco  score deco_entry_key deco_reco
        0    w      1              X         W

        >>> mask = slm.restrict_row(['y'])
        >>> aeq(lm1.dot(lm2, mask=mask).matrix, np.array([[4]]))
        True
        """
        if (mask is not None) and (top is not None):
            raise ValueError('Cannot provide both `mask` and `top` in dot')
        aligned_self, aligned_other = self.align(other, axes=[(0, 1, False)])
        if mask is None:
            return aligned_self._dot(aligned_other, top=top)
        else:
            aligned_self, mask = aligned_self.align(mask, axes=[(1, 1, False)])
            aligned_other, mask = aligned_other.align(mask, axes=[(0, 0, False)])
            return aligned_self._dot(aligned_other, mask=mask)

    def _maximum(self, other: LabeledMatrix) -> LabeledMatrix:
        return LabeledMatrix(self.label, safe_maximum(self.matrix, other.matrix),
                             deco=self.deco)

    def _minimum(self, other: LabeledMatrix) -> LabeledMatrix:
        return LabeledMatrix(self.label, safe_minimum(self.matrix, other.matrix),
                             deco=self.deco)

    def maximum(self, other: LabeledMatrix) -> LabeledMatrix:
        """
        Take coordinate-maximum between self and other (for each pair of labels take the largest value)
        >>> lm1 = LabeledMatrix(2*[['b', 'c', 'd']],
        ...                     np.array([[1,0,5], [0,0,0], [0,1,0]]))
        >>> lm2 = LabeledMatrix(2*[['b', 'c', 'd']],
        ...                     np.array([[1,-2,2], [0,3,0],[0,2,0]]))
        >>> aeq(lm1.maximum(lm2.to_sparse()).matrix, np.array([[1, 0, 5], [0, 3, 0], [0, 2, 0]]))
        True
        """
        aligned_self, aligned_other = self.align(other, axes=[(0, 0, True), (1, 1, True)])
        return aligned_self._maximum(aligned_other)

    def minimum(self, other: LabeledMatrix) -> LabeledMatrix:
        """
        Take coordinate-minimum between self and other (for each pair of labels take the smallest value)
        >>> lm1 = LabeledMatrix(2*[['b', 'c', 'd']],
        ...                     np.array([[1,0,5], [0,0,0], [0,1,0]]))
        >>> lm2 = LabeledMatrix(2*[['b', 'c', 'd']],
        ...                     np.array([[1,-2,2], [0,3,0],[0,2,0]]))
        >>> aeq(lm1.minimum(lm2.to_sparse()).matrix,
        ...                 np.array([[1, -2,  2],[0,  0,  0], [0,  1, 0]]))
        True
        >>> lm1.minimum(lm2 * 1.0).matrix
        array([[ 1., -2.,  2.],
               [ 0.,  0.,  0.],
               [ 0.,  1.,  0.]])
        >>> aeq(lm1.minimum(lm2.to_sparse()).matrix, np.array([[1, -2,  2], [0,  0,  0], [0, 1, 0]]))
        True
        >>> lm1.to_sparse().minimum(lm2.to_sparse()) == lm1.minimum(lm2 * 1.0)
        True
        """
        aligned_self, aligned_other = self.align(other, axes=[(0, 0, True), (1, 1, True)])
        return aligned_self._minimum(aligned_other)

    def max(self, axis: Optional[int] = 1) -> Union[Number, LabeledMatrix]:
        """
        Calculate the maximum along given axis, return result as a diagonal square LabeledMatrix
        >>> lm = LabeledMatrix((['c', 'b'], ['x', 'y', 'z']) ,
        ...                    np.array([[0.3, 0.6, 0], [0.75, 0.5, 0]]))
        >>> lm.max().sort().matrix.toarray()
        array([[0.75, 0.  ],
               [0.  , 0.6 ]], dtype=float32)
        >>> lm.max().sort().label
        (['b', 'c'], ['b', 'c'])
        >>> lm.max(axis=0).sort().matrix.toarray()
        array([[0.75, 0.  , 0.  ],
               [0.  , 0.6 , 0.  ],
               [0.  , 0.  , 0.  ]], dtype=float32)
        >>> lm.max(axis=0).sort().label
        (['x', 'y', 'z'], ['x', 'y', 'z'])
        >>> aeq(lm.max(axis=0).matrix, lm.to_sparse().max(axis=0).matrix.toarray())
        True
        >>> aeq(lm.max().matrix, lm.to_sparse().max().matrix.toarray())
        True
        >>> aeq(lm.to_sparse().max(axis=0).matrix.toarray(), lm.max(axis=0).matrix)
        True
        """
        values = safe_max(self.matrix, axis=axis)
        if axis is None:
            return values
        return LabeledMatrix.from_diagonal(self.label[co_axis(axis)], values, self.deco[co_axis(axis)])

    def min(self, axis: Optional[int] = 1) -> Union[Number, LabeledMatrix]:
        """
        Calculate the minimum along given axis, return result as a diagonal square LabeledMatrix
        >>> lm = LabeledMatrix((['a', 'b', 'c'], ['x', 'y', 'z']),
        ...                    np.arange(9).reshape(3, 3) + 1)
        >>> lm.min().sort().matrix.toarray().astype(np.int64)
        array([[1, 0, 0],
               [0, 4, 0],
               [0, 0, 7]])
        >>> lm.min().sort().label
        (['a', 'b', 'c'], ['a', 'b', 'c'])
        >>> lm.min(axis=0).sort().matrix.toarray().astype(np.int64)
        array([[1, 0, 0],
               [0, 2, 0],
               [0, 0, 3]])
        >>> lm.min(axis=0).sort().label
        (['x', 'y', 'z'], ['x', 'y', 'z'])
        >>> aeq(lm.min(axis=0).matrix, lm.to_sparse().min(axis=0).matrix.toarray())
        True
        >>> aeq(lm.min().matrix.toarray(), lm.to_sparse().min().matrix.toarray())
        True
        >>> aeq(lm.to_sparse().min(axis=0).matrix.toarray(), lm.min(axis=0).matrix.toarray())
        True
        """
        values = safe_min(self.matrix, axis=axis)
        if axis is None:
            return values
        return LabeledMatrix.from_diagonal(self.label[co_axis(axis)], values, self.deco[co_axis(axis)])

    def sum(self, axis: Optional[int] = 1) -> Union[Number, LabeledMatrix]:
        """
        Calculate the sum along given axis, return result as a diagonal square LabeledMatrix
        >>> lm = LabeledMatrix((['a', 'b'], ['x']), np.array([[5], [25]]))
        >>> aeq(lm.sum().matrix.toarray(), np.array([[5, 0], [0, 25]]))
        True
        >>> lm.sum().label
        (['a', 'b'], ['a', 'b'])
        >>> aeq(lm.sum(axis=0).matrix.toarray(), np.array([[30]]))
        True
        >>> lm.sum(axis=0).label
        (['x'], ['x'])
        >>> aeq(lm.sum(axis=0).matrix, lm.to_sparse().sum(axis=0).matrix)
        True
        >>> aeq(lm.sum().matrix, lm.to_sparse().sum().matrix.toarray())
        True
        """
        values = safe_sum(self.matrix, axis=axis)
        if axis is None:
            return values
        return LabeledMatrix.from_diagonal(self.label[co_axis(axis)], values, self.deco[co_axis(axis)])

    def mean(self, axis: Optional[int] = 1) -> Union[Number, LabeledMatrix]:
        """
        Calculate the average along given axis, return result as a diagonal square LabeledMatrix
        >>> lm = LabeledMatrix((['a', 'b'], ['x', 'y']), np.array([[5, 5], [25, 75]])).sort()
        >>> lm.mean().matrix.toarray()
        array([[ 5.,  0.],
               [ 0., 50.]], dtype=float32)
        >>> lm.mean().label
        (['a', 'b'], ['a', 'b'])
        >>> lm.mean(axis=0).sort().matrix.toarray()
        array([[15.,  0.],
               [ 0., 40.]], dtype=float32)
        >>> lm.mean(axis=0).sort().label
        (['x', 'y'], ['x', 'y'])
        >>> aeq(lm.mean(axis=0).matrix, lm.to_sparse().mean(axis=0).matrix.toarray())
        True
        >>> aeq(lm.mean().matrix, lm.to_sparse().mean().matrix.toarray())
        True
        """
        values = safe_mean(self.matrix, axis=axis)
        if axis is None:
            return values
        return LabeledMatrix.from_diagonal(self.label[co_axis(axis)], values, self.deco[co_axis(axis)])

    def normalize(self, axis: Optional[int] = 1, norm: str = 'l1') -> LabeledMatrix:
        """
        Normalize the matrix along given axis
        """
        trans_matrix = normalize(self.matrix.copy(), axis=axis, norm=norm)
        return LabeledMatrix(self.label, trans_matrix, deco=self.deco)

    def apply_numpy_function(self,
                             function: Callable[[Number], Number],
                             function_args: Optional[list] = None) -> LabeledMatrix:
        """
        Apply a pointwise (numpy) method on each value of the matrix
        :param function: callable that takes one number as input and that returns another number
        :param function_args: optional list of args to pass into the function call
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1.345,2.234], [3.890,4.678]]))
        >>> lm.apply_numpy_function(np.round, [1]).matrix
        array([[1.3, 2.2],
               [3.9, 4.7]])
        >>> lm.apply_numpy_function(np.log).matrix
        array([[0.29639401, 0.8037937 ],
               [1.35840916, 1.54287067]])
        >>> lm.to_sparse().apply_numpy_function(np.log).matrix.toarray()
        array([[0.29639402, 0.80379367],
               [1.3584092 , 1.5428706 ]], dtype=float32)
        >>> np.all(lm.apply_numpy_function(np.log).apply_numpy_function(np.exp).matrix == lm.matrix)
        True
        """
        if function_args is None:
            function_args = []
        if self.is_sparse:
            matrix = self.matrix.apply_pointwise_function(function, function_args)
        else:
            matrix = self.matrix.copy().astype(np.float64)
            matrix[matrix.nonzero()] = function(matrix[matrix.nonzero()], *function_args)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def power(self, power: int) -> LabeledMatrix:
        return self.apply_numpy_function(np.power, [power])

    def __pow__(self, power: int):
        return self.power(power)

    def abs(self) -> LabeledMatrix:
        """
        Take absolute value of each matrix entry
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[0.5, -0.5], [-0.25, 0]]))
        >>> lm.abs().matrix
        array([[0.5 , 0.5 ],
               [0.25, 0.  ]])
        >>> lm.to_sparse().abs().matrix.toarray()
        array([[0.5 , 0.5 ],
               [0.25, 0.  ]], dtype=float32)
        """
        return LabeledMatrix(self.label, abs(self.matrix), deco=self.deco)

    def __abs__(self):
        return self.abs()

    def sign(self) -> LabeledMatrix:
        """
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[0.5, -0.5], [-0.25, 0]]))
        >>> lm.sign().matrix
        array([[ 1., -1.],
               [-1.,  0.]])
        >>> np.all(lm.to_sparse().sign().matrix.toarray() == lm.sign().matrix)
        True
        """
        if self.is_sparse:
            return LabeledMatrix(self.label, self.matrix.sign(), deco=self.deco)
        return LabeledMatrix(self.label, np.sign(self.matrix), deco=self.deco)

    def inverse(self, scalar: float = 1.0) -> LabeledMatrix:
        """
        Inverse each matrix value and multiply by given scalar
        >>> lm1 = LabeledMatrix(2*[['b', 'c']], np.array([[1.,2.], [5., 0.]]))
        >>> lm1.inverse(2.0).matrix
        array([[2. , 1. ],
               [0.4, 0. ]])
        >>> lm1.to_sparse().inverse(2.0).matrix.toarray()
        array([[2. , 1. ],
               [0.4, 0. ]], dtype=float32)
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,2], [5,4]]))
        >>> lm.to_sparse().inverse(2.0).matrix.toarray()
        array([[2. , 1. ],
               [0.4, 0.5]], dtype=float32)
        >>> lm.inverse(2.).matrix
        array([[2. , 1. ],
               [0.4, 0.5]])
        """
        return LabeledMatrix(self.label,
                             pseudo_element_inverse(self.matrix, scalar),
                             deco=self.deco)

    def log1p(self) -> LabeledMatrix:
        """
        Apply np.log1p function on each matrix value
        """
        return self.apply_numpy_function(np.log1p)

    def log(self) -> LabeledMatrix:
        """
        Apply np.log function on each matrix value
        """
        return self.apply_numpy_function(np.log)

    def exp(self) -> LabeledMatrix:
        """
        Apply np.exp function on each matrix value
        """
        return self.apply_numpy_function(np.exp)

    def round(self) -> LabeledMatrix:
        """
        Apply np.round function on each matrix value
        """
        return self.apply_numpy_function(np.round)

    def trunc(self) -> LabeledMatrix:
        """
        Apply np.trunc function on each matrix value
        """
        return self.apply_numpy_function(np.trunc)

    def clip(self, lower: Optional[float] = None, upper: Optional[float] = None) -> LabeledMatrix:
        """
        Apply np.clip function with arguments lower and upper on each matrix value
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[0.5, -0.5], [-0.25, 0]]))
        >>> lm.clip(0.1, 0.25).matrix
        array([[0.25, 0.1 ],
               [0.1 , 0.  ]])
        >>> lm.to_sparse().clip(0.1, 0.25).matrix.toarray()
        array([[0.25, 0.1 ],
               [0.1 , 0.  ]], dtype=float32)
        """
        return self.apply_numpy_function(np.clip, [lower, upper])

    def logistic(self, shift: float, coef: float = 1.) -> LabeledMatrix:
        """
        Apply sigmoid function centered at shift and rescaled by coef on each matrix value
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y', 'z']),
        ...                    np.array([[3, 6, 9], [7, 5, 10]]))
        >>> lm.logistic(6).matrix
        array([[0.04742587, 0.5       , 0.95257413],
               [0.73105858, 0.26894142, 0.98201379]])
        >>> lm.to_sparse().logistic(6).matrix.toarray()
        array([[0.04742587, 0.5       , 0.95257413],
               [0.7310586 , 0.26894143, 0.98201376]], dtype=float32)
        """
        return self.apply_numpy_function(sigmoid, [shift, coef])

    def truncate(self,
                 cutoff=None,
                 nb=None,
                 cum_h=None, cum_v=None,
                 nb_h=None, nb_v=None,
                 to_optimal_format=False) -> LabeledMatrix:
        """
        TODO docstring
        :param cutoff:
        :param nb:
        :param cum_h:
        :param cum_v:
        :param nb_h:
        :param nb_v:
        :param to_optimal_format:
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'z', 'y']), np.array([[4, 6, 5], [7, 9, 8]]))
        >>> lm.truncate(cutoff=6).matrix
        array([[0, 6, 0],
               [7, 9, 8]])
        >>> lm.truncate(nb_h=2).matrix
        array([[0, 6, 5],
               [0, 9, 8]])
        >>> lm.truncate(nb_v=1).matrix
        array([[0, 0, 0],
               [7, 9, 8]])
        >>> lm.truncate(nb=2).matrix
        array([[0, 0, 0],
               [0, 9, 8]])
        """
        matrix = self.matrix.copy()
        if (nb_h == 0) or (nb_v == 0) or (nb == 0):
            return self.zeros()
        if cum_h is not None:
            if not 0 < cum_h < 1:
                raise LabeledMatrixException(f'`cum_h` must be between 0 and 1, got {cum_h}')
            matrix = truncate_by_cumulative(matrix, per=cum_h, axis=1)
        if cum_v is not None:
            if not 0 < cum_v < 1:
                raise LabeledMatrixException(f'`cum_v` must be between 0 and 1, got {cum_v}')
            matrix = truncate_by_cumulative(matrix, per=cum_v, axis=0)
        if cutoff:
            matrix = truncate_with_cutoff(matrix, cutoff)
        if nb is not None:
            matrix = truncate_by_count(matrix, nb, axis=None)
        if nb_h is not None and nb_h < matrix.shape[1]:  # top elements of each row
            matrix = truncate_by_count(matrix, nb_h, axis=1)
        if nb_v is not None and nb_v < matrix.shape[0]:
            matrix = truncate_by_count(matrix, nb_v, axis=0)

        lm = LabeledMatrix(self.label, matrix, deco=self.deco)
        if to_optimal_format:
            lm = lm.to_optimal_format()
        return lm

    def truncate_by_budget(self, density: LabeledMatrix, volume: Number) -> LabeledMatrix:
        """
        Return a sparse matrix for which sum by row of non-zero elements is bigger or equal to given volume
        Typically, self can be considered as similarity matrix
        and truncation is used to define neighborhoods of a given point with the minimal size equal to volume
        :param density: diagonal LabeledMatrix with labels from self.column.
                        It will be used to give different weights to columns when calculating the sum of a row
        :param volume: we want each row to have at least given volume.
                       In the example above it can be considered as minimal size of a point's neighbourhood.
        >>> sim = LabeledMatrix((['b', 'c'], ['x', 'y', 'z']), np.array([[2, 1, 0], [0, 1, 2]]))
        >>> density = LabeledMatrix((['x', 'y', 'z'], ['x', 'y', 'z']),
        ...                         np.diag(np.arange(1,4)))
        >>> aeq(sim.truncate_by_budget(density, 1.5).matrix, np.array([[2, 1, 0], [0, 0, 2]]))
        True
        """
        if not density.is_square:
            raise ValueError('density matrix should be squared')
        density = density.align(self, axes=[(1, 0, None), (0, 0, None)])[0]
        matrix = truncate_by_budget(self.matrix, density.matrix, volume)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def truncate_by_count(self, max_rank: Union[dict, int], axis: Optional[int]) -> LabeledMatrix:
        """
        Return LabeledMatrix truncated to max_rank maximal elements
        :param max_rank: int or dict: maximal rank to keep (can be dictionary with labels of given axis)
        :param axis: 0/1/None
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'z', 'y']), np.array([[4, 6, 5], [7, 9, 8]]))
        >>> lm.truncate_by_count({'b': 1, 'c': 2}, axis=1).matrix
        array([[0, 6, 0],
               [0, 9, 8]])
        """
        if isinstance(max_rank, dict):
            max_rank = np.array([max_rank.get(label, 0) for label in self.label[co_axis(axis)]],
                                dtype=np.int64)
        elif not is_integer(max_rank):
            raise ValueError('max_rank must be integer or dict')
        if np.all(max_rank == 0):
            return self.zeros()
        return LabeledMatrix(self.label, truncate_by_count(self.matrix, max_rank, axis=axis),
                             deco=self.deco)

    def anomaly(self, skepticism: float = 1.) -> LabeledMatrix:
        """
        Calculate anomaly measure of each value to appear in matrix compared to marginal distribution of those values
        FIXME Link exact description of the formulas
        :param skepticism: float, the bigger it is more likely we will consider some value an anomaly
        >>> v = np.array([[7, 0, 3],
        ...               [5, 0, 5],
        ...               [3, 3, 4],
        ...               [0, 10, 0],
        ...               [0, 9, 1]])
        >>> lm = LabeledMatrix((range(5), range(3)), v).to_dense()
        >>> lm.anomaly().matrix.round(3)
        array([[ 0.588, -1.085,  0.   ],
               [ 0.   , -1.085,  0.084],
               [ 0.   ,  0.   ,  0.   ],
               [-0.716,  1.012, -0.603],
               [-0.716,  0.647,  0.   ]])
        """
        return LabeledMatrix(self.label, anomaly(self.matrix, skepticism), self.deco)

    def dict_argmax(self) -> dict:
        """
        >>> mat = np.array([[4, 5, 0],
        ...                 [0, 3, 0],
        ...                 [0, 0, 0],
        ...                 [1, 0, 0]])
        >>> lm = LabeledMatrix((['a', 'b', 'c', 'd'], ['x', 'y', 'z']), mat)
        >>> lm.dict_argmax()
        {'a': 'y', 'b': 'y', 'd': 'x'}
        >>> lm.to_sparse().dict_argmax() == lm.dict_argmax()
        True
        """
        lm_nonzero = self.without_zeros()
        asort = lm_nonzero.matrix.argmax(axis=1)
        my_dict = {x: lm_nonzero.column[asort[i]] for i, x in enumerate(lm_nonzero.row)}
        return my_dict

    def dict_max(self, axis: int = 1) -> Dict[Any, Number]:
        """
        >>> mat = np.array([[4, 5, 0],
        ...                 [0, 3, 0],
        ...                 [0, 0, 0],
        ...                 [1, 0, 0]])
        >>> lm = LabeledMatrix((['a', 'b', 'c', 'd'], ['x', 'y', 'z']), mat)
        >>> lm.dict_max()
        {'a': 5, 'b': 3, 'd': 1}
        >>> lm.to_sparse().dict_max() == lm.dict_max()
        True
        """
        lm_nonzero = self.without_zeros()
        return dict(zip(lm_nonzero.label[co_axis(axis)], lm_nonzero.matrix.max(axis=axis)))

    def similarity(self, other: Optional[LabeledMatrix] = None, cutoff: float = 0.005, nb_keep: int = 200,
                   top: int = 1000, cumtop: float = 0.02):
        """
        Return Labeled matrix with similarity between row vectors, keeping only the most similar pairs
        FIXME Link an article or describe how exactly similarity is calculated
        :param other: optional LabeledMatrix to calculate similarities between row vectors from self and from other
        :param cutoff: in resulting matrix keep only similarities larger than `cutoff`
        :param nb_keep: in resulting matrix keep only `nb_keep` most similar pairs for each row label
        :param top: consider only `top` largest coordinates in each row vector when calculating similarity
        :param cumtop: consider only `cumtop` part of each row vector's l1-norm (keeping largest coordinates)
        :return: matrix of the shape (len(self.row), len(other.row)) or
                                     (len(self.row), len(other.row)) if other is not specified
        >>> mat = np.array([[0.4, 0.5, 0, 0.1, 0],
        ...                 [0, 0.1, 0.4, 0.5, 0],
        ...                 [0, 0, 0, 0, 0],
        ...                 [0.1, 0, 0.5, 0.4, 0]])
        >>> lm = LabeledMatrix((range(4), range(5)),  mat).to_sparse()
        >>> lm.similarity(nb_keep=2).matrix.toarray()
        array([[1. , 0.2, 0. , 0. ],
               [0. , 1. , 0. , 0.8],
               [0. , 0. , 0. , 0. ],
               [0. , 0.8, 0. , 1. ]], dtype=float32)
        >>> lm.similarity().label
        ([0, 1, 2, 3], [0, 1, 2, 3])
        >>> res = lm.similarity(other=lm.restrict_row([1, 2]), nb_keep=2)
        >>> res.matrix.toarray()
        array([[0.2],
               [1. ],
               [0.8]], dtype=float32)
        >>> res.label
        ([0, 1, 3], [1])
        """
        if other is None:
            return LabeledMatrix((self.row, self.row),
                                 buddies_matrix(self.matrix, cutoff=cutoff, nb_keep=nb_keep,
                                                top=top, cumtop=cumtop),
                                 deco=(self.row_deco, self.row_deco))
        lm, other = self.align(other, axes=[(0, 0, False)])
        lm = lm.without_zeros(axis=1)
        other = other.without_zeros(axis=1)
        return LabeledMatrix((lm.row, other.row),
                             pairwise_buddy(lm.matrix, other.matrix,
                                            cutoff=cutoff, nb_keep=nb_keep),
                             deco=(lm.row_deco, other.row_deco))

    def jaccard(self) -> LabeledMatrix:
        """
        LM is viewed as occurence matrix (item, user) and the jaccard similarity is
        computed between items.
        see http://en.wikipedia.org/wiki/Jaccard_index
        Let u1 = set(User(items1)) and u2 = set(User(items2)), then
            J(item1, item2) = len(u1 & u2) / len(u1 | u2)
        >>> matrix = np.array([[1, 0, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]])
        >>> lm = LabeledMatrix((['a', 'b', 'c'], range(4)), matrix)
        >>> aeq(lm.jaccard().matrix,
        ...     np.array([[ 1., .5,  .33333333], [.5, 1., .5], [ 0.33333333, .5,  1.]]))
        True
        """
        source = self.nonzero_mask()
        inter = source._dot(source.transpose())
        total = source.sum()
        union = inter.nonzero_mask()._dot(total) \
            ._add(total._dot(inter.nonzero_mask()))._add(-inter)
        return 1. * inter / union

    def _relative_count(self, top: Union[float, int], axis: int) -> int:
        """
        :param top: int with number of lines/columns or float with proportion of lines/columns
        :param axis: in [0, 1]
        :return: top if it is already and int >= 1,
                 otherwise number of lines/columns (depending on axis) needed to match proportion given by top
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1, 0], [3, 4]]))
        >>> lm._relative_count(100, axis=0)
        100
        >>> lm._relative_count(0.5, axis=1)
        1
        """
        if 0 <= top < 1:
            top = int(top * self.shape[axis])
        if not is_integer(top):
            raise ValueError(f'top argument must be a float in [0, 1) or an integer, got {type(top)} instead')
        return top

    def top_values_similarity(self, other: Optional[LabeledMatrix] = None, top: Union[float, int] = 0.1,
                              axis: int = 0, renorm: bool = True, potential: bool = False) -> LabeledMatrix:
        """
        Calculating similarity between matrix columns (axis=0) or rows (axis=1) based on common top values.
        For example for axis=0:
        * in each column keep only top largest values
        * for each pair of columns i, j overlap is equal to number of coordinates corresponding
        to top values between column i and column j

        :param other: optional other matrix to calculate similarity between columns in self and in other
                      default: None, so we calculate similarities between columns in self
        :param top: can be either number of top values to consider, either float in [0, 1] as % of total values
                    default: 10% of top values
        :param axis: 0 or 1, default is 0
        :param renorm: boolean to normalize resulting row with l1 norm:
                       we consider overlap as % of number of top values, default is True
        :param potential: boolean whether to use matrix values as weights when calculating intersection weight or
                          just count intersecting non-zero values, default False (counting non-zero values)
        >>> matrix = np.array([[10, 1, 2], [2, 5, 3], [5, 6, 6], [1, 3, 5]])
        >>> lm = LabeledMatrix((range(4), ['a', 'b', 'c']), matrix)
        >>> np.array(lm.top_values_similarity(2).matrix)
        array([[1. , 0.5, 0.5],
               [0.5, 1. , 0.5],
               [0.5, 0.5, 1. ]], dtype=float32)
        >>> np.array(lm.top_values_similarity(lm, 2).matrix)
        array([[1. , 0.5, 0.5],
               [0.5, 1. , 0.5],
               [0.5, 0.5, 1. ]], dtype=float32)
        >>> np.array(lm.top_values_similarity(3).matrix)
        array([[1.       , 0.6666667, 0.6666667],
               [0.6666667, 1.       , 1.       ],
               [0.6666667, 1.       , 1.       ]], dtype=float32)
        >>> np.array(lm.top_values_similarity(0.8, renorm=False).matrix, dtype=np.int64)
        array([[3, 2, 2],
               [2, 3, 3],
               [2, 3, 3]])
        >>> np.array(lm.top_values_similarity(0.8, axis=1, renorm=False).matrix, dtype=np.int64)
        array([[2, 1, 1, 1],
               [1, 2, 2, 2],
               [1, 2, 2, 2],
               [1, 2, 2, 2]])
        >>> lm.top_values_similarity(0.8, axis=1, renorm=False).is_square
        True
        >>> lm.top_values_similarity(0.8, axis=1, renorm=False).row
        [0, 1, 2, 3]
        """
        if axis != 0:
            return self.transpose().top_values_similarity(other.transpose() if other else None, top, axis=co_axis(axis),
                                                          renorm=renorm, potential=potential)
        top = self._relative_count(top, axis=0)

        top_lm = self.truncate(nb_v=top)
        top_lm_mask = top_lm.nonzero_mask()
        if other:
            other_top_mask = other.truncate(nb_v=top).nonzero_mask()
        else:
            other_top_mask = top_lm_mask
        if potential:
            overlap_lm = top_lm.transpose().dot(other_top_mask)
        else:
            overlap_lm = top_lm_mask.transpose().dot(other_top_mask)
        if renorm:
            overlap_lm /= top
        return overlap_lm

    def _check_dispatch_params(self, max_ranks=None, max_volumes=None):
        nb_topic = len(self.column)

        if max_volumes is None:
            max_volumes = self.matrix.shape[0]
        if is_integer(max_volumes):
            max_volumes = np.full(nb_topic, max_volumes, dtype=np.int64)
        elif isinstance(max_volumes, dict):
            max_volumes = np.array([max_volumes.get(topic, 0) for topic in self.column])
        else:
            raise ValueError('max_volumes must be integer or dict')

        if max_ranks is None:
            max_ranks = self.matrix.shape[0]
        if is_integer(max_ranks):
            max_ranks = np.full(nb_topic, max_ranks, dtype=np.int64)
        elif isinstance(max_ranks, dict):
            max_ranks = np.array([max_ranks[topic] for topic in self.column])
        else:
            raise ValueError('max_ranks must be integer or dict')

        if np.min(max_volumes) < 0:
            raise ValueError('max_volumes must be positive or 0')

        if np.min(max_ranks) < 0:
            raise ValueError('max_ranks must be positive or 0')

        return max_ranks, max_volumes

    def _round_robin_allocation(self,
                                maximum_pressure: int,
                                max_ranks: Union[list, np.ndarray],
                                max_volumes: Union[list, np.ndarray]) -> LabeledMatrix:
        """
        WARNING: works only on nonzero scores
        Return LabeledMatrix with allocation user-topic
        :param maximum_pressure: maximal number of times each user can appear in allocation
        :param max_ranks: arraylike: maximal rank of user in extract each topic can take
        :param max_volumes: arraylike: maximal value of population volume each topic can be allocated to
        >>> matrix = np.array([[10, 1, 3], [2, 5, 3], [5, 6, 6], [1, 3, 5]])
        >>> lm = LabeledMatrix((range(4), ['a', 'b', 'c']), matrix)
        >>> lm._round_robin_allocation(1, [4, 4, 4], [4, 4, 4])\
        ...     .to_flat_dataframe('user_id', 'topic_id', 'score') #doctest: +NORMALIZE_WHITESPACE
           user_id topic_id  score
        0        0        a   10.0
        1        1        b    5.0
        2        2        b    6.0
        3        3        c    5.0

        >>> lm._round_robin_allocation(2, [1, 4, 2], [1, 2, 2])\
        ...     .to_flat_dataframe('user_id', 'topic_id', 'score') #doctest: +NORMALIZE_WHITESPACE
           user_id topic_id  score
        0        0        a   10.0
        1        1        b    5.0
        2        2        b    6.0
        3        2        c    6.0
        4        3        c    5.0
        """
        choice = rank_dispatch(self.matrix, maximum_pressure, np.asarray(max_ranks), np.asarray(max_volumes))
        return LabeledMatrix(self.label, choice, self.deco)

    def round_robin_allocation(self,
                               maximum_pressure: int = 1,
                               max_ranks: Optional[Union[int, dict]] = None,
                               max_volumes: Optional[Union[int, dict]] = None) -> LabeledMatrix:
        """
        Return LabeledMatrix with allocation of row labels to 1 or many column labels in round-robin manner.
        WARNING: zero values are ignored and not allocated
        If we consider matrix as user preferences to some topic and we want to assign each user to 1 or many topics:
        * each topic (column) will look at users (rows) in order given by their preference (matrix value)
        * we will assign users (row labels) to topics (columns) iterating over topics (columns)
            and picking most-interested user (row) - https://en.wikipedia.org/wiki/Round-robin_item_allocation

        Optionally we may add more constraints
        :param maximum_pressure: maximal number of times each row can appear in allocation, default 1
        :param max_ranks: int/dict: maximal rank of user (row) each topic (column) can take
        :param max_volumes: int/dict: maximal number of rows that can be allocated to each column
        :return: LabeledMatrix of the same shape and with the same values, keeping only allocated pairs
        >>> matrix = np.array([[10, 1, 3], [2, 5, 3], [5, 6, 6], [1, 3, 5]])
        >>> lm = LabeledMatrix((range(4), ['a', 'b', 'c']), matrix)
        >>> lm.round_robin_allocation(1).to_flat_dataframe('user_id', 'topic_id', 'score') #doctest: +NORMALIZE_WHITESPACE
           user_id topic_id  score
        0        0        a   10.0
        1        1        b    5.0
        2        2        b    6.0
        3        3        c    5.0
        >>> lm.round_robin_allocation(1, 1, 1)\
        ...     .to_flat_dataframe('user_id', 'topic_id', 'score') #doctest: +NORMALIZE_WHITESPACE
           user_id topic_id  score
        0        0        a   10.0
        1        2        b    6.0
        >>> lm.round_robin_allocation(2, {'a': 2, 'b': 1, 'c': 3}, 2)\
        ...     .to_flat_dataframe('user_id', 'topic_id', 'score') #doctest: +NORMALIZE_WHITESPACE
           user_id topic_id  score
        0        0        a   10.0
        1        2        b    6.0
        2        2        c    6.0
        3        3        c    5.0
        """
        max_ranks, max_volumes = self._check_dispatch_params(max_ranks, max_volumes)
        return self._round_robin_allocation(maximum_pressure, max_ranks, max_volumes)

    def argmax_allocation(self,
                          maximum_pressure: int,
                          max_ranks: Optional[Union[int, dict]] = None,
                          max_volumes: Optional[Union[int, dict]] = None) -> LabeledMatrix:
        """
        Return LabeledMatrix with allocation of row labels to 1 or many column labels based on matrix value.
        WARNING: zero values are ignored and not allocated
        If we consider matrix as user preferences to some topic and we want to assign each user to 1 or many topics:
        * we order pairs user—topic (row-column) in preference descending order
        * iterating over pairs in this order we will check if it can be allocated wrt constraints:
        :param maximum_pressure: maximal number of times each row can appear in allocation, default 1
        :param max_ranks: int/dict: maximal rank of user (row) each topic (column) can take
        :param max_volumes: int/dict: maximal number of rows that can be allocated to each column
        :return: LabeledMatrix of the same shape and with the same values, keeping only allocated pairs
        """
        max_ranks, max_volumes = self._check_dispatch_params(max_ranks, max_volumes)

        choice = argmax_dispatch(self.matrix, maximum_pressure, max_ranks, max_volumes)
        return LabeledMatrix(self.label, choice, self.deco)

    def tail_clustering(self, weight: LabeledMatrix, n_clusters: int, min_density: float = 0.) -> LabeledMatrix:
        """
        :param weight: weight LabeledMatrix, sum of weights for each row will be used to weight samples
        :param n_clusters: number of clusters to construct
        :param min_density:
        >>> v = np.array([[7, 0, 3],
        ...               [5, 0, 5],
        ...               [3, 3, 4],
        ...               [0, 10, 0],
        ...               [0, 9, 1]])
        >>> lm = LabeledMatrix((range(5), range(3)), v).to_dense()
        >>> weight = LabeledMatrix.from_diagonal(np.arange(5), [1, 2, 4, 4, 1])
        >>> lm.tail_clustering(weight, 2).dict_argmax()
        {0: 2, 1: 2, 2: 2, 3: 3, 4: 3}
        >>> lm.tail_clustering(weight, 2).to_sparse().dict_argmax()
        {0: 2, 1: 2, 2: 2, 3: 3, 4: 3}
        """
        mults = weight.sum(axis=1).align(self, axes=[(1, 1, None), (0, 1, None)])[0] \
            .matrix.diagonal()
        if self.is_sparse:
            labels = sparse_tail_clustering(self.matrix, mults, n_clusters, min_density)
        else:
            labels = tail_clustering(self.matrix, mults, n_clusters)
        lm = LabeledMatrix.from_zip_occurrence(self.row, take_indices(self.row, labels))
        lm.set_deco(self.row_deco, self.row_deco)
        return lm

    def hierarchical_clustering(self, weight: Optional[LabeledMatrix], n_clusters: int) -> LabeledMatrix:
        """
        Return LabeledMatrix with clusters' characteristic vectors for each row using Ward's method
        https://en.wikipedia.org/wiki/Ward%27s_method
        :param weight: weight LabeledMatrix, sum of weights for each row will be used to weight samples
        :param n_clusters: number of clusters to construct
        >>> v = np.array([[7, 0, 3],
        ...               [5, 0, 5],
        ...               [3, 3, 4],
        ...               [0, 10, 0],
        ...               [0, 9, 1]])
        >>> lm = LabeledMatrix((range(5), range(3)), v)
        >>> weights = LabeledMatrix.from_diagonal(np.arange(5), [1, 2, 4, 4, 1])
        >>> lm.hierarchical_clustering(weights, 2).dict_argmax()
        {0: 2, 1: 2, 2: 2, 3: 3, 4: 3}
        """
        if weight is not None:
            weight = weight.sum(axis=1).align(self, axes=[(1, 1, None), (0, 1, None)])[0].matrix.diagonal()

        labels = WardTree(np.asarray(self.matrix), weights=weight, n_clusters=n_clusters).build_labels()
        lm = LabeledMatrix.from_zip_occurrence(self.row, take_indices(self.row, labels))
        lm.set_deco(self.row_deco, self.row_deco)

        return lm

    def connected_components(self, connection='weak') -> LabeledMatrix:
        """
        Return LabeledMatrix with clusters' characteristic vectors for each row
        using scipy.sparse.csgraph.connected_components
        :param connection: the type of connection to use in scipy.sparse.csgraph.connected_components
        >>> mat = np.array([[1, 1, 0, 0, 0],
        ...                 [0, 1, 0, 0, 1],
        ...                 [0, 0, 1, 0, 0],
        ...                 [0, 0, 1, 1, 0],
        ...                 [0, 1, 0, 0, 1]])
        >>> lm = LabeledMatrix([range(5)]*2, mat)
        >>> lm.connected_components().label
        ([0, 1, 2, 3, 4], [0, 1])
        >>> lm.connected_components().matrix.toarray().astype(np.int64)
        array([[1, 0],
               [1, 0],
               [0, 1],
               [0, 1],
               [1, 0]])
        >>> lm.to_sparse().connected_components('strong').sort().matrix.toarray().astype(np.int64)
        array([[0, 1, 0, 0],
               [1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [1, 0, 0, 0]])
        """
        if not self.is_square:
            raise LabeledMatrixException('Matrix must be square')
        matrix = self.matrix.to_scipy_sparse(copy=False) if self.is_sparse else self.matrix
        n_components, lab = connected_components(matrix, connection=connection)
        print(f'Number of components: {n_components}')
        lm = LabeledMatrix.from_zip_occurrence(self.row, lab)
        lm.set_deco(row_deco=self.row_deco)
        return lm

    def affinity_clusters(self, preference: Optional[Union[str, Number]] = None, max_iter: int = 200):
        """
        :param preference: 'mean', 'median' or float
        :param max_iter:
        >>> similarity = np.array([[3, 5, 1, 1],
        ...                        [5, 2, 2, 1],
        ...                        [1, 1, 2, 6],
        ...                        [1, 2, 4, 3]])
        >>> lm = LabeledMatrix((['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd']), similarity)
        >>> lm_ac = lm.affinity_clusters().sort()
        >>> lm_ac.label
        (['a', 'b', 'c', 'd'], ['a', 'd'])
        >>> lm_ac.matrix.toarray().astype(np.int64)
        array([[1, 0],
               [1, 0],
               [0, 1],
               [0, 1]])
        """
        if not self.is_square:
            raise LabeledMatrixException('Works only on squared matrix')
        labels = affinity_propagation(self.matrix, preference, max_iter=max_iter)
        lm = LabeledMatrix.from_zip_occurrence(self.row, take_indices(self.row, labels))
        lm.set_deco(*self.deco)
        return lm

    def spectral_clusters(self, n_clusters: int = 10):
        """
        Return LabeledMatrix with clusters' characteristic vectors for each row
        using sklearn.cluster.SpectralClustering
        :param n_clusters: number of clusters to generate
        """
        clust = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_neighbors=3)
        lab = clust.fit_predict(self.matrix)
        lm = LabeledMatrix.from_zip_occurrence(self.row, lab)
        lm.set_deco(row_deco=self.row_deco)
        return lm

    def co_clustering(self, ranks, max_iter=120, nb_preruns=30, pre_iter=4) -> Tuple[LabeledMatrix, LabeledMatrix]:
        """
        >>> matrix = np.array([[5, 5, 5, 0, 0, 0],
        ...                    [5, 5, 5, 0, 0, 0],
        ...                    [0, 0, 0, 5, 5, 5],
        ...                    [0, 0, 0, 5, 5, 5],
        ...                    [4, 4, 0, 4, 4, 4],
        ...                    [4, 4, 4, 0, 4, 4]])
        >>> lm = LabeledMatrix((range(6), range(6)), matrix)
        >>> w, h = lm.co_clustering((3,2))
        >>> w = w.sort()
        >>> h = h.sort()
        >>> w.label == ([0, 1, 2, 3, 4, 5], [0, 1, 2])
        True
        >>> h.label == ([0, 1, 2, 3, 4, 5], [0, 1])
        True
        >>> w.dict_argmax()[0] == w.dict_argmax()[1]
        True
        >>> w.dict_argmax()[2] == w.dict_argmax()[3]
        True
        >>> w.dict_argmax()[4] == w.dict_argmax()[5]
        True
        >>> h.dict_argmax()[4] == h.dict_argmax()[5] == h.dict_argmax()[3]
        True
        >>> h.dict_argmax()[0] == h.dict_argmax()[1] == h.dict_argmax()[2]
        True
        """
        w, h = co_clustering(self.matrix, ranks=ranks, max_iter=max_iter, nb_preruns=nb_preruns, pre_iter=pre_iter)
        lmw = LabeledMatrix.from_zip_occurrence(self.row, w)
        lmw.set_deco(row_deco=self.row_deco)
        lmh = LabeledMatrix.from_zip_occurrence(self.column, h)
        lmw.set_deco(column_deco=self.column_deco)
        return lmw, lmh

    def svd(self, rank: int, randomized: bool = True) -> Tuple[LabeledMatrix, LabeledMatrix]:
        """
        Partial SVD matrix factorization
        :param rank: number of largest singular values to keep
        :param randomized: False corresponds to scipy.sparse.linalg.svds solver
                           True faster but not exact algorithm from http://arxiv.org/pdf/0909.4061.pdf will be used
        :return: two matrices, U & V each multiplied by sqrt of singular values matrix S.
        >>> lm = LabeledMatrix((range(5), range(4)), np.arange(20).reshape(5,4))
        >>> u, w = lm.svd(2, randomized=False)
        >>> u.matrix  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        array([[ 1.2323054 ,  0.4517649 ],
               [ 0.79740864,  1.5830183 ],
               [ 0.3625115 ,  2.7142718 ],
               [-0.07238551,  3.8455253 ],
               [-0.5072828 ,  4.9767785 ]], dtype=float32)
        >>> u.label
        ([0, 1, 2, 3, 4], [0, 1])
        >>> w.label
        ([0, 1, 2, 3], [0, 1])
        >>> np.allclose(u.dot(w.transpose()).matrix, lm.matrix, atol=1e-4)
        True
        >>> u, w = lm.svd(2)
        >>> u.matrix
        array([[-0.45176491, -1.23232132],
               [-1.58301848, -0.79741909],
               [-2.71427205, -0.36251687],
               [-3.84552562,  0.07238536],
               [-4.97677919,  0.50728759]])
        >>> u.dot(w.transpose()) == lm
        True
        """
        method = randomized_svd if randomized else svds
        if randomized:
            self_matrix = self.matrix
        else:
            if self.is_sparse:
                self_matrix = self.matrix.to_scipy_sparse(copy=False)
            else:
                self_matrix = KarmaSparse(self.matrix).to_scipy_sparse(copy=False)
        u, s, w = method(self_matrix, rank)
        s = np.diag(np.sqrt(s))
        u = u.dot(s)
        w = s.dot(w).transpose()
        lm_u = LabeledMatrix((self.row, list(range(rank))), u, (self.row_deco, {}))
        lm_w = LabeledMatrix((self.column, list(range(rank))), w, (self.column_deco, {}))
        return lm_u, lm_w

    def svd_appromximation(self, rank: int, randomized: bool = True) -> LabeledMatrix:
        """
        Return reconstructed matrix after SVD factorization: lower rank approximation of the initial matrix
        :param rank: number of largest singular values to keep
        :param randomized: False corresponds to scipy.sparse.linalg.svds solver
                           True faster but not exact algorithm from http://arxiv.org/pdf/0909.4061.pdf will be used
        >>> lm = LabeledMatrix((range(5), range(4)), np.arange(20).reshape(5,4))
        >>> lm.svd_appromximation(2) == lm
        True
        >>> aeq(lm.svd_appromximation(1).matrix, lm.svd_appromximation(2).matrix)
        False
        """
        if isinstance(rank, int):
            ranks = (rank,)
        else:
            ranks = rank
        uu, ww = [], []
        method = randomized_svd if randomized else svds
        if randomized:
            self_matrix = self.matrix
        else:
            if self.is_sparse:
                self_matrix = self.matrix.to_scipy_sparse(copy=False)
            else:
                self_matrix = KarmaSparse(self.matrix).to_scipy_sparse(copy=False)
        for rank_ in ranks:
            u, s, w = method(self_matrix, rank_)
            uu.append(u.dot(np.diag(s)))
            ww.append(w)
        matrix = 1. * safe_dot(np.hstack(uu), np.vstack(ww),
                               mat_mask=self.matrix) / len(ranks)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def sort_by_hierarchical_clustering(self) -> LabeledMatrix:
        """
        Return LabeledMatrix with rows and columns reordered wrt the Ward tree leaves labels
        https://en.wikipedia.org/wiki/Ward%27s_method
        >>> v = np.array([[7, 0, 3],
        ...               [5, 0, 5],
        ...               [3, 3, 4],
        ...               [0, 10, 0],
        ...               [0, 9, 1]])
        >>> lm = LabeledMatrix((range(5), range(3)), v)
        >>> lm.sort_by_hierarchical_clustering().matrix #doctest: +NORMALIZE_WHITESPACE
        array([[ 0, 10,  0],
               [ 0,  9,  1],
               [ 3,  3,  4],
               [ 5,  0,  5],
               [ 7,  0,  3]])
        """
        lm = self.sort()
        order_coding = clustering_dispatcher(np.asarray(lm.matrix))
        idx = np.asarray(sorted(list(range(len(lm.row))), key=lambda i: order_coding[i]))

        if lm.is_square:
            return lm._take_on_row(idx)._take_on_column(idx)
        return lm._take_on_row(idx)

    @UseSeed()
    def nmf(self, rank: Optional[Union[int, List[int]]],
            max_model_rank: int = 60,
            max_iter: int = 150,
            svd_init: bool = False) -> Tuple[LabeledMatrix, LabeledMatrix]:
        """
        Return non-negative matrix factorization
        :param rank:
        :param max_model_rank:
        :param max_iter:
        :param svd_init:
        >>> m = LabeledMatrix.from_random(seed=12, sparse=False)
        >>> w, h = m.dot(m.transpose()).nmf(3, max_iter=20)
        >>> (w.dot(h.transpose()) - m.dot(m.transpose())).abs().matrix.sum() < 0.05
        True
        """
        if (isinstance(rank, int)) or (rank is None):
            ranks = (rank,)
        else:
            ranks = rank
        ww, hh = [], []

        if not self.is_sparse and keep_sparse(self.matrix):
            matrix = self.to_sparse().matrix
        else:
            matrix = self.matrix

        for rank_ in ranks:
            w, h = nmf(matrix, rank=rank_, max_model_rank=max_model_rank, max_iter=max_iter, svd_init=svd_init)
            ww.append(w)
            hh.append(h)
        www, hhh = np.hstack(ww), np.hstack(hh)
        lmw = LabeledMatrix((self.row, list(range(www.shape[1]))), www,
                            deco=(self.row_deco, {}))
        lmh = LabeledMatrix((self.column, list(range(hhh.shape[1]))), hhh,
                            deco=(self.column_deco, {}))
        return lmw, lmh

    def nmf_fold(self, right_factor: LabeledMatrix, max_iter: int = 30) -> LabeledMatrix:
        """
        Find left factor in non-negative matrix factorization for a given matrix `self` and right factor
        :param right_factor: already obtained right factor
        :param max_iter: number of iterations
        >>> w = LabeledMatrix.from_random((10, 5), seed=100, sparse=False)
        >>> h = LabeledMatrix.from_random((7, 5), seed=100)
        >>> matrix = w.dot(h.transpose())
        >>> ww = matrix.nmf_fold(h, 200)
        >>> ww.label == w.label
        True
        >>> diff = w - ww
        >>> diff.abs().max(axis=None) < 0.1
        True
        """
        source, right_factor = self.align(right_factor, axes=[(0, 1, True)])
        left_factor_matrix = nmf_fold(source.matrix, right_factor.matrix.transpose(),
                                      max_iter)
        return LabeledMatrix((source.row, right_factor.column), left_factor_matrix,
                             deco=(source.row_deco, right_factor.column_deco))

    def nmf_approximation(self,
                          rank: Optional[Union[int, List[int]]],
                          max_iter: int = 120,
                          svd_init: bool = True) -> LabeledMatrix:
        """
        Return reconstructed matrix after NMF factorization: lower rank approximation of the initial matrix
        :param rank:
        :param max_iter:
        :param svd_init:
        >>> m = LabeledMatrix.from_random(seed=12)
        >>> lm = m.dot(m.transpose())
        >>> (lm - lm.nmf_approximation(3)).abs().matrix.sum() < 0.1
        True
        >>> (lm - lm.nmf_approximation([3, 4, 3])).abs().matrix.sum() < 0.1
        True
        """
        if (isinstance(rank, int)) or (rank is None):
            ranks = (rank,)
        else:
            ranks = rank
        ww, hh = [], []

        for rank_ in ranks:
            w, h = nmf(self.matrix, rank=rank_, max_iter=max_iter, svd_init=svd_init)
            ww.append(w)
            hh.append(h)
        matrix = 1. * safe_dot(np.hstack(ww), np.hstack(hh).transpose(),
                               mat_mask=self.matrix) / len(ranks)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def _reco_df(self, row: Optional[Any] = None, nb: int = 20):
        """
        >>> mat = np.array([[4, 5, 0, 1],
        ...                 [0, 0, 0, 0]])
        >>> lm = LabeledMatrix((['a', 'b'], range(4)), mat)
        >>> lm.display_reco('b') #doctest: +NORMALIZE_WHITESPACE
        All entries are zeros
          entry_key  nb_nonzero
        0         b           0
        ------------------------------------------------------------
        Empty DataFrame
        Columns: [reco, score]
        Index: []
        """
        if row is None:
            row = self.rand_row()

        if row not in self.row:
            print('Unknown label')
            return pd.DataFrame(), pd.DataFrame()
        lm_loc = self.restrict_row([row])

        try:
            lm_loc = lm_loc.without_zeros()
        except LabeledMatrixException:
            print('All entries are zeros')
            return (pd.DataFrame([[row, 0]], columns=['entry_key', 'nb_nonzero']),
                    pd.DataFrame([], columns=['reco', 'score']))

        my_df = lm_loc.truncate(nb_h=nb) \
            .to_flat_dataframe('entry_key', 'reco', 'score') \
            .sort_values('score', ascending=False)

        # head
        nb_nonzero = len(lm_loc.column)
        head = {'entry_key': [row], 'nb_nonzero': [nb_nonzero]}

        if nb_nonzero > 0:
            head['total_score'] = [safe_sum(lm_loc.matrix)]
            head['min_score'] = [safe_min(lm_loc.matrix)]
        if getattr(self, 'row_deco', None):
            head['deco'] = [self.row_deco.get(row, '')]
        return pd.DataFrame(head).reset_index(drop=True), my_df.reset_index(drop=True)

    def display_reco(self, row: Optional[Any] = None, nb: int = 15, reco_only: bool = False):
        """
        For a given row show column labels of nb largest elements.
        If we consider current matrix as some preference score (for example rows as users,
                columns as items to recommend) this method show top-nb recommendation.
        :param row: chosen row label to get the reco. If not specified, random row is chosen
        :param nb: number of recommendations to show
        :param reco_only: boolean to skip header part
        >>> mat = np.array([[4, 5, 0, 1],
        ...                 [5, 4, 1, 0],
        ...                 [0, 1, 4, 5],
        ...                 [1, 0, 3, 4]])
        >>> lm = LabeledMatrix([['w', 'x', 'y', 'z'], ['a', 'b', 'c', 'd']], mat)
        >>> lm.display_reco('w') #doctest: +NORMALIZE_WHITESPACE
          entry_key  nb_nonzero  total_score  min_score
        0         w           3           10          1
        ------------------------------------------------------------
          reco  score
        0    b      5
        1    a      4
        2    d      1
        >>> lm.set_deco(row_deco={'w':'q'}, column_deco={'a': 'h'})
        >>> lm.display_reco('w') #doctest: +NORMALIZE_WHITESPACE
          entry_key  nb_nonzero  total_score  min_score deco
        0         w           3           10          1    q
        ------------------------------------------------------------
          reco  score deco_entry_key deco_reco
        0    b      5              q      None
        1    a      4              q         h
        2    d      1              q      None
        """
        from IPython.display import display
        head, reco = self._reco_df(row, nb)
        reco = reco.loc[:, reco.columns != 'entry_key']
        if not reco_only:
            display(head)
            print('-' * 60)
        display(reco)

    def plot_as_clustermap(self, **kwargs):
        """
        Plot current matrix as a sns.clustermap
        :param kwargs: kwargs to propagate into sns.clustermap
        :return: matplotlib Axes object
        """
        import seaborn as sns
        kwargs = {'figsize': (30, 30), 'dendrogram_ratio': 0.05, 'cmap': sns.color_palette('RdYlGn_r', 100), **kwargs}
        return sns.clustermap(self.to_vectorial_dataframe(), **kwargs)

    def plot_as_heatmap(self, ordering: Optional[str] = None, **kwargs):
        """
        Plot current matrix as a sns.heatmap
        :param ordering: optional ordering of labels. Available values 'hierarchical' and 'naive' (for sort on labels)
        :param kwargs: kwargs to propagate into sns.heatmap
        :return: matplotlib Axes object
        """
        import seaborn as sns
        if ordering == 'hierarchical':
            lm = self.sort_by_hierarchical_clustering()
        elif ordering == 'naive':
            lm = self.sort()
        else:
            lm = self
        with sns.axes_style('white'):
            kwargs = {'figsize': (30, 30), 'cmap': sns.color_palette('RdYlGn_r', 100), 'annot': True, **kwargs}
            ax = sns.heatmap(lm.to_dense().to_vectorial_dataframe(), square=True, **kwargs)
        return ax

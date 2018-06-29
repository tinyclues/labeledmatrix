#
# Copyright tinyclues, All rights reserved
#
from itertools import izip

from cytoolz.dicttoolz import keymap
from collections import OrderedDict
import random

from scipy.sparse.linalg import svds
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.linear_model.logistic import LogisticRegression
from glove import Glove

from karma.core.column import Column, create_column_from_data
from karma.core.dataframe import DataFrame
from karma.core.instructions import Map, Dot, KroneckerVector
from karma.core.karmacode import KarmaCode
from karma.core.utils import is_integer, use_seed
from karma.core.method import enforce_lib_closure

from cyperf.clustering.hierarchical import WardTree
from cyperf.indexing.indexed_list import IndexedList
from cyperf.matrix.karma_sparse import KarmaSparse, is_karmasparse, ks_diag
from cyperf.tools import logit, logit_inplace, take_indices
from cyperf.tools.getter import apply_python_dict

from karma.core.utils.collaborative_tools import keep_sparse
from karma.learning.affinity_propagation import affinity_propagation
from karma.learning.co_clustering import co_clustering
from karma.learning.hierarchical import clustering_dispatcher
from karma.learning.matrix_utils import *
from karma.learning.nmf import nmf, nmf_fold
from karma.learning.randomize_svd import randomized_svd
from karma.learning.sparse_tail_clustering import sparse_tail_clustering
from karma.learning.tail_clustering import tail_clustering

from karma.runtime import KarmaSetup

from karma.core.labeledmatrix.utils import (lm_occurence,
                                            co,
                                            lmdiag,
                                            zipmerge,
                                            dict_merge,
                                            aeq)


__all__ = ["LabeledMatrix", "LabeledMatrixException"]


def to_flat_dataframe(lm_collection, row='col0', col='col1'):
    """
    Returns a flat dataframe with non-zero values of all LabeledMatrix contained in lm_collection.
    If lm_collection is a list, then values of corresponding LMs will be put into 'sim0', 'sim1', ... columns
    If lm_collection is a dict, then dict keys will used as column names for value columns
    If a pair (row, column) is found in one LM and not in another, missing values will be filled with zeros
    Args:
        lm_collection: list or dict of LabeledMatrix
        row: name of column that corresponds to rows of LabeledMatrix
        col: name of column that corresponds to columns of LabeledMatrix
    """
    if not isinstance(lm_collection, dict):
        lm_collection = OrderedDict([('sim' + str(j), lm_) for (j, lm_) in enumerate(lm_collection)])
    names = lm_collection.keys()

    # First of all we need to align all LabeledMatrices to then extract same indices from each of them
    row_names = lm_collection[names[0]].row
    column_names = lm_collection[names[0]].column

    for k in names[1:]:
        row_names = row_names.union(lm_collection[k].row)[0]
        column_names = column_names.union(lm_collection[k].column)[0]

    align_lm = LabeledMatrix((row_names, column_names), np.zeros((len(row_names), len(column_names)), dtype=np.uint8))

    all_row_idx, all_col_idx = np.zeros(0, np.int64), np.zeros(0, np.int64)
    lm_aligned_collection = {}
    for k in names:
        lm_aligned_collection[k] = lm_collection[k].align(align_lm, axes=[(0, 0, None), (1, 1, None)], self_only=True)
        row_indices, col_indices = lm_aligned_collection[k].matrix.nonzero()
        # here we have two arrays of nonzero row and column indices of a matrix such that
        # (row_indices[j], column_indices[j]) corresponds to jth non zero value of matrix
        # then we need to check if a pair (row_indices[j], column_indices[j]) is already in (all_row_idx, all_col_idx)
        # and if not we memorise its index and then append to (all_row_idx, all_col_idx)
        missing_idx = np.where(np.logical_not(np.logical_and(np.in1d(row_indices, all_row_idx),
                                                             np.in1d(col_indices, all_col_idx))))[0]
        all_row_idx = np.concatenate((all_row_idx, row_indices[missing_idx]))
        all_col_idx = np.concatenate((all_col_idx, col_indices[missing_idx]))

    data = DataFrame()
    data[row] = create_column_from_data(take_indices(row_names, all_row_idx))
    data[col] = create_column_from_data(take_indices(column_names, all_col_idx))
    for k in names:
        # we fill missing values with zeros (via standard LM alignment process),
        # but we can recheck if a pair (row, column) is present in current LM, and put Missing if not
        data[k] = create_column_from_data(lm_aligned_collection[k].matrix[all_row_idx, all_col_idx])

    return data


class LabeledMatrixException(Exception):
    pass


class LabeledMatrix(object):
    """
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
    ...             column_deco={'x': "X", 'z': "Z", 'y': "Y"})
    >>> lm.reco('c') #doctest: +NORMALIZE_WHITESPACE
    -------------------------------------------------------
    entry_key | nb_nonzero | total_score | min_score | deco
    -------------------------------------------------------
    c           3            24            7           C
    -------------------
    reco | score | deco
    -------------------
    z      9       Z
    y      8       Y
    x      7       X
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
    def __init__(self, (row, column), matrix, deco=({}, {})):
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

    @property
    def shape(self):
        """
        >>> matrix = np.array([[4, 6, 5], [7, 9, 8], [1, 3, 2]])
        >>> row, column = ['b', 'c', 'a'], ['x', 'z', 'y']
        >>> lm = LabeledMatrix((row, column), matrix.copy())
        >>> lm.shape
        (3, 3)
        """
        return self.matrix.shape

    @property
    def dtype(self):
        """
        Returns underlying matrix dtype
        >>> matrix = np.array([[4, 6, 0], [0, 9, 8], [0, 0, 2]])
        >>> row, column = ['b', 'c', 'a'], ['x', 'z', 'y']
        >>> LabeledMatrix((row, column), matrix.astype(np.int16)).dtype
        dtype('int16')
        >>> LabeledMatrix((row, column), matrix.astype(np.float32)).dtype
        dtype('float32')
        >>> LabeledMatrix((row, column), matrix).to_sparse().dtype
        dtype('float64')
        """
        return self.matrix.dtype

    @staticmethod
    def check_format((row, column), matrix):
        """
        Used to check a number of assertion on the content of a LabeledMatrix.

        :param row: row labels, labels should be unique and of the correct
                    shape, with respect to the given matrix
        :param column: column labels, labels should be unique and of the correct
                       shape, with respect to the given matrix
        :param matrix: a numpy or scipy two dimensional array
        :return: None
        :raise: LabeledMatrixException

        Exemple: ::

            >>> LabeledMatrix.check_format((['a','b','c'], ['1', '2']),
            ...                            [[0, 1, 0],[1, 0, 1]])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            LabeledMatrixException: Unacceptable matrix type: <type 'list'>
            >>> LabeledMatrix.check_format((['a','b','c'], ['1', '2']),
            ...                            np.array([2]))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            LabeledMatrixException: Wrong number of dimension: 1
            >>> LabeledMatrix.check_format((['a','b','c'], ['1']),
            ...                            np.array([[0, 1, 0],[1, 0, 1]]))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            LabeledMatrixException: Number of rows 3 should corresponds to matrix.shape[0]=2
            >>> LabeledMatrix.check_format((['a', 'c'], ['1', '2']),
            ...                            np.array([[0, 1, 0],[1, 0, 1]]))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            LabeledMatrixException: Number of columns 2 should corresponds to matrix.shape[1]=3
            >>> LabeledMatrix.check_format((['a', 'b'], ['1', '2', '3']),
            ...                            np.array([[0, 1, 0], [1, 0, 1]]))  # doctest: +ELLIPSIS

        """

        if not (is_scipysparse(matrix) or is_karmasparse(matrix) or isinstance(matrix, np.ndarray)):
            raise LabeledMatrixException("Unacceptable matrix type: %s" % (type(matrix),))
        if matrix.ndim != 2:
            raise LabeledMatrixException("Wrong number of dimension: %s" % (matrix.ndim,))
        # get default labels
        if len(row) != matrix.shape[0]:
            raise LabeledMatrixException(
                "Number of rows %s should corresponds to matrix.shape[0]=%s" % (len(row),
                                                                                matrix.shape[0]))
        if len(column) != matrix.shape[1]:
            raise LabeledMatrixException(
                "Number of columns %s should corresponds to matrix.shape[1]=%s" % (len(column),
                                                                                   matrix.shape[1]))

    def __repr__(self):
        repr_ = "<LabeledMatrix with properties :"
        if self.is_sparse:
            repr_ += "\n * sparse of format {}".format(self.matrix.format)
        else:
            repr_ += "\n * dense numpy"
        repr_ += "\n * dtype {}".format(self.matrix.dtype)
        repr_ += "\n * dimension {}".format(self.matrix.shape)
        nnz = self.nnz()
        repr_ += "\n * number of non-zero elements {} ".format(nnz)
        repr_ += "\n * density of non-zero elements {} ".format(round(self.density(), 7))
        if nnz:
            min_ = np.min(self.matrix.data) if self.is_sparse else np.min(self.matrix)
            repr_ += "\n * min element {}".format(min_)
            max_ = np.max(self.matrix.data) if self.is_sparse else np.max(self.matrix)
            repr_ += "\n * max element {}".format(max_)
            repr_ += "\n * sum elements {}".format(self.matrix.sum())
        return unicode(repr_ + ">", encoding="utf-8", errors='replace').encode("utf-8")

    def __getitem__(self, *labs):
            """
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
            if len(labs[0]) == 2:
                return self.matrix[self.row.index(labs[0][0]), self.column.index(labs[0][1])]\
                    if (labs[0][0] in self.row) and (labs[0][1] in self.column) else 0
            else:
                raise LabeledMatrixException("Dimension of input should be 2.")

    def is_square(self):
        """
        >>> lm1 = LabeledMatrix((['b', 'a'], ['d', 'b']), np.arange(4).reshape(2, 2))
        >>> lm1.is_square()
        False
        >>> lm1.symmetrize_label().is_square()
        True
        """
        return self.row == self.column

    def has_sorted_row(self):
        return self.row.is_sorted()

    def has_sorted_column(self):
        return self.column.is_sorted()

    def nnz(self):
        """
        nnz as non-zero! Returns the number of non-zero element contained in
        the matrix

        :return: the number of non-zero element in the matrix
        """
        if not hasattr(self, "_nnz"):
            self._nnz = number_nonzero(self.matrix)
        return self._nnz

    def density(self):
        return self.nnz() * 1. / np.product(self.matrix.shape)

    def copy(self):
        return LabeledMatrix(self.label, self.matrix.copy(), deco=self.deco)

    def rand_row(self):
        return random.choice(self.row)

    def rand_column(self):
        return random.choice(self.column)

    def rand_label(self):
        return (self.rand_row(), self.rand_column())

    def set_deco(self, row_deco=None, column_deco=None):
        def _check_dict(deco):
            if not isinstance(deco, dict):
                raise ValueError('Decoration should be a dict')
            else:
                return deco

        if row_deco is not None:
            self.row_deco = _check_dict(row_deco)
        if column_deco is not None:
            self.column_deco = _check_dict(column_deco)
        self.deco = (self.row_deco, self.column_deco)

    def to_dense(self):
        """
        Returns only a view if self is dense
        >>> import scipy.sparse as sp
        >>> lm = LabeledMatrix([['b', 'c'], ['a', 'd', 'e']],
        ...                    sp.rand(2, 3, 0.5, format="csr"))
        >>> lm.to_dense().to_sparse() == lm
        True
        """
        if self.is_sparse:
            return LabeledMatrix(self.label, self.matrix.toarray(), deco=self.deco)
        else:
            return self

    def to_sparse(self):
        """
        Returns only a view if self is sparse
        >>> lm = LabeledMatrix([['b', 'c'], ['a', 'd']], np.array([[1,2], [3,0]]))
        >>> lm.to_sparse().to_dense() == lm
        True
        """
        if self.is_sparse:
            return self
        else:
            return LabeledMatrix(self.label, KarmaSparse(self.matrix), deco=self.deco)

    def align(self, other, axes=[(0, 0, False), (1, 1, False)], self_only=False):
        """
        Aligns two LabeledMatrix according to provided axis
        For instance, if axes = [axis] and axis = (0, 1, True) means that
            * self.column (axis[0]=0) will be aligned with other.column (axis[1]=1)
            * by taking the union of the labels (axis[2]=True)

        The third parameter in each element of axes` corresponds to outer/inner join in SQL logic.
            * if axis[2] == False, the default value, the intersection (inner) of labels will be taken
            * if axis[2] == True, the union (outer) of labels will be taken
            * if axis[2] == None, the labels of other matrix will be taken

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
            label_self = self_copy.label[co(ax_self)]
            label_other = other_copy.label[co(ax_other)]
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
        if self_only:
            return self_copy
        else:
            return self_copy, other_copy

    def symmetrize_label(self, restrict=False):
        """
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
        if self.is_square():
            return self
        elif restrict:
            lm = self.restrict_row(self.column).restrict_column(self.row)
        else:
            lm = self.extend_row(self.column).extend_column(self.row)
        return lm.align(lm, [(0, 1, None)])[0]

    def _take_on_row(self, indices=None):
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

    def sort_row(self):
        if not self.has_sorted_row():
            row, argsort_row = self.row.sorted()
            return LabeledMatrix((row, self.column), self.matrix[argsort_row], self.deco)
        else:
            return self

    def sort_column(self):
        if not self.has_sorted_column():
            column, argsort_column = self.column.sorted()
            return LabeledMatrix((self.row, column), self.matrix[:, argsort_column],
                                 self.deco)
        else:
            return self

    def sort_by_deco(self):
        """
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

    def sort(self):
        """
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

    def extend_row(self, rows, deco={}):
        """
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
                             deco=(dict_merge(self.row_deco, deco), self.column_deco))

    def extend_column(self, columns, deco={}):
        """
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
                             deco=(self.row_deco, dict_merge(self.column_deco, deco)))

    def extend(self, label, deco=({}, {})):
        """
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,4]]))
        >>> lm.extend(2*[['a']]).matrix
        array([[0, 0, 0],
               [0, 1, 2],
               [0, 3, 4]])
        >>> lm.to_sparse().extend((['a'], ['a'])).matrix.toarray().astype(np.int)
        array([[0, 0, 0],
               [0, 1, 2],
               [0, 3, 4]])
        >>> lm.to_sparse().extend(2*[['a', 'd', 'c']]).matrix.toarray().astype(np.int)
        array([[1, 2, 0, 0],
               [3, 4, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
        >>> lm.to_sparse().extend(2*[['a', 'd', 'c']]).label
        (['b', 'c', 'a', 'd'], ['b', 'c', 'a', 'd'])
        >>> lm = LabeledMatrix([['b', 'c'], ['x']], np.array([[1], [3]]))
        >>> lm.extend((["a", "b"], ["y"])).matrix
        array([[1, 0],
               [3, 0],
               [0, 0]])
        >>> lm.extend((["a", "b"], ["y"])).label
        (['b', 'c', 'a'], ['x', 'y'])
        >>> lm.to_sparse().extend((["a", "b"], ["y"])).matrix.toarray().astype(np.int)
        array([[1, 0],
               [3, 0],
               [0, 0]])
        """
        if label[0] and label[1]:
            return self.extend_column(label[1]).extend_row(label[0])
        elif label[0]:
            return self.extend_row(label[0])
        elif label[1]:
            return self.extend_column(label[1])
        else:
            return self.copy()

    def restrict_row(self, rows):
        common_rows, arg_row, _ = self.row.intersection(rows)
        if KarmaSetup.verbose:
            print 'keeping {} rows out of {}.'.format(len(common_rows), len(self.row))
        if len(common_rows):
            return LabeledMatrix((common_rows, self.column),
                                 align_along_axis(self.matrix, arg_row, 1, False),
                                 deco=self.deco)
        else:
            raise LabeledMatrixException('restrict has returned an empty labeled matrix')

    def restrict_column(self, columns):
        common_columns, arg_column, _ = self.column.intersection(columns)
        if KarmaSetup.verbose:
            print 'keeping {} columns out of {}.'.format(len(common_columns),
                                                         len(self.column))
        if len(common_columns):
            return LabeledMatrix((self.row, common_columns),
                                 align_along_axis(self.matrix, arg_column, 0, False),
                                 deco=self.deco)
        else:
            raise LabeledMatrixException('restrict has returned an empty labeled matrix')

    def restrict(self, label):
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
        >>> rlm.sort().matrix.toarray().astype(np.int)
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
        elif label[0] is None:
            return self.restrict_column(label[1])
        elif label[1] is None:
            return self.restrict_row(label[0])
        else:
            return self.copy()

    def exclude_row(self, rows):
        keep_row, arg_row = self.row.difference(rows)
        if len(keep_row):
            return LabeledMatrix((keep_row, self.column),
                                 align_along_axis(self.matrix, arg_row, 1, False),
                                 deco=self.deco)
        else:
            raise LabeledMatrixException('exclude has returned an empty labeled matrix')

    def exclude_column(self, columns):
        keep_column, arg_column = self.column.difference(columns)
        if len(keep_column):
            return LabeledMatrix((self.row, keep_column),
                                 align_along_axis(self.matrix, arg_column, 0, False),
                                 deco=self.deco)
        else:
            raise LabeledMatrixException('exclude has returned an empty labeled matrix')

    def exclude(self, label):
        """
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
        elif label[0] is None:
            return self.exclude_column(label[1])
        elif label[1] is None:
            return self.exclude_row(label[0])
        else:
            return self.copy()

    def sample_rows(self, p):
        """
        >>> lm = LabeledMatrix((['b', 'c', 'd'], ['x', 'y', 'z']),
        ...                    np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]]))
        >>> lm.sample_rows(12) #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: sample larger than population
        >>> lm.sample_rows(0.2) #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        LabeledMatrixException: p should be > 0, currently is 0.
        >>> sample1 = lm.sample_rows(2)
        >>> sample1.matrix.shape
        (2, 3)
        >>> sample2 = lm.sample_rows(0.8)
        >>> sample2.matrix.shape
        (2, 3)
        """
        if 0 < p < 1:
            p = len(self.row) * p
        p = int(p)
        if p <= 0:
            raise LabeledMatrixException('p should be > 0, currently is {}.'.format(p))
        return self.restrict_row(random.sample(self.row, p))

    def sample_columns(self, p):
        """
        >>> lm = LabeledMatrix((['b', 'c', 'd'], ['x', 'y', 'z']),
        ...                    np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]]))
        >>> lm.sample_columns(12) #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: sample larger than population
        >>> lm.sample_rows(0.2) #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        LabeledMatrixException: p should be > 0, currently is 0.
        >>> sample1 = lm.sample_columns(2)
        >>> sample1.matrix.shape
        (3, 2)
        >>> sample2 = lm.sample_columns(0.4)
        >>> sample2.matrix.shape
        (3, 1)
        """
        if 0 < p < 1:
            p = len(self.column) * p
        p = int(p)
        if p <= 0:
            raise LabeledMatrixException('p should be > 0, currently is {}.'.format(p))
        return self.restrict_column(random.sample(self.column, p))

    def transpose(self):
        return LabeledMatrix((self.column, self.row), self.matrix.transpose(),
                             deco=(self.deco[1], self.deco[0]))

    def without_zeros(self, axis=None, min_nonzero=1):
        """
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
            return self.without_zeros(axis=1, min_nonzero=min_nonzero)\
                       .without_zeros(axis=0, min_nonzero=min_nonzero)
        index = np.where(safe_sum(nonzero_mask(self.matrix), axis=axis) >= min_nonzero)[0]
        if len(index) == self.matrix.shape[co(axis)]:
            return self
        if len(index) == 0:
            raise LabeledMatrixException('without_zeros has returned an empty labeled matrix')
        to_keep = self.label[co(axis)].select(index)
        if axis == 1:
            return LabeledMatrix((to_keep, self.column),
                                 self.matrix[index], deco=self.deco)
        if axis == 0:
            return LabeledMatrix((self.row, to_keep),
                                 self.matrix[:, index], deco=self.deco)

    def nonzero_mask(self):
        """
        Same as matrix_utils.nonzero_mask():
        replace each non-zero entry by 1.
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[1,0], [3,0]]))
        >>> aeq(lm.nonzero_mask().matrix, np.array([[1, 0], [1, 0]]))
        True
        >>> aeq(lm.to_sparse().nonzero_mask().matrix, np.array([[1, 0], [1, 0]]))
        True
        """
        return LabeledMatrix(self.label, nonzero_mask(self.matrix), deco=self.deco)

    def rank(self, axis=1, reverse=False):
        """
        Returns ranks of entries along given axis in ascending order (descending if reverse is True)
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

    def diagonal(self):
        """
        >>> lm = LabeledMatrix((['b', 'c'], ['b', 'c']), np.array([[4, 6], [7, 9]]))
        >>> aeq(lm.to_sparse().diagonal().matrix, np.array([[4, 0],[0, 9]]))
        True
        >>> lm.diagonal().matrix
        array([[4, 0],
               [0, 9]])
        """
        if not self.is_square():
            raise Exception("diagonal() works only on square matrices.")
        if self.is_sparse:
            diag_matrix = ks_diag(self.matrix.diagonal(), format=self.matrix.format)
        else:
            diag_matrix = np.diagflat(self.matrix.diagonal())
        return LabeledMatrix(self.label, diag_matrix, deco=self.deco)

    def without_diagonal(self):
        """
        >>> lm = LabeledMatrix((['b', 'c'], ['b', 'c']), np.array([[4, 6], [7, 9]]))
        >>> lm.to_sparse().without_diagonal().matrix.toarray()
        array([[0., 6.],
               [7., 0.]])
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

    def without_mask(self, mask):
        """
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

    def zeros(self):
        """
        >>> lm = LabeledMatrix([['b', 'c']] * 2, np.array([[1,2], [3,4]]))
        >>> aeq(lm.zeros().matrix, np.zeros((2,2)))
        True
        >>> lm.zeros().label == lm.label
        True
        >>> aeq(lm.to_sparse().zeros().matrix.toarray(), np.zeros((2,2)))
        True
        """
        if self.is_sparse:
            matrix_zeros = KarmaSparse(self.matrix.shape)
        else:
            matrix_zeros = np.zeros(self.matrix.shape, dtype=self.matrix.dtype)
        return LabeledMatrix(self.label, matrix_zeros, deco=self.deco)

    def truncate(self, cutoff=None, nb=None, cum_h=None, cum_v=None, nb_h=None, nb_v=None):
        """
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
            if not (0 < cum_h < 1):
                raise LabeledMatrixException("truncate cum_h {}".format(cum_h))
            matrix = truncate_by_cumulative(matrix, per=cum_h, axis=1)
        if cum_v is not None and (0 < cum_v < 1):
            if not (0 < cum_v < 1):
                raise LabeledMatrixException("Top cum_v {}".format(cum_v))
            matrix = truncate_by_cumulative(matrix, per=cum_v, axis=0)
        if cutoff:
            matrix = truncate_with_cutoff(matrix, cutoff)
        if nb is not None:
            matrix = truncate_by_count(matrix, nb, axis=None)
        if nb_h is not None and nb_h < matrix.shape[1]:  # top elements of each row
            matrix = truncate_by_count(matrix, nb_h, axis=1)
        if nb_v is not None and nb_v < matrix.shape[0]:
            matrix = truncate_by_count(matrix, nb_v, axis=0)

        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def truncate_by_budget(self, density, volume):
        """
        self - typically a similarity matrix
        density - diagonal matrix (~ self.column)
        volume - min_volume of neighbourhood we wish w.r.t. given density

        >>> sim = LabeledMatrix((['b', 'c'], ['x', 'y', 'z']), np.array([[2, 1, 0], [0, 1, 2]]))
        >>> density = LabeledMatrix((['x', 'y', 'z'], ['x', 'y', 'z']),
        ...                         np.diag(np.arange(1,4)))
        >>> aeq(sim.truncate_by_budget(density, 1.5).matrix, np.array([[2, 1, 0], [0, 0, 2]]))
        True
        """
        assert density.is_square(), "Density matrix should be squared"
        density = density.align(self, axes=[(1, 0, None), (0, 0, None)])[0]
        matrix = truncate_by_budget(self.matrix, density.matrix, volume)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def truncate_by_count(self, max_rank, axis):
        """
        Returns LabeledMatrix truncated to max_rank maximal elements
        Args:
            max_rank: int or dict: maximal rank per axis
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'z', 'y']), np.array([[4, 6, 5], [7, 9, 8]]))
        >>> lm.truncate_by_count({'b': 1, 'c': 2}, axis=1).matrix
        array([[0, 6, 0],
               [0, 9, 8]])
        """
        if isinstance(max_rank, dict):
            max_rank = np.array([max_rank.get(label, 0) for label in self.label[co(axis)]],
                                dtype=np.int)
        elif not is_integer(max_rank):
            raise ValueError('max_rank must be integer or dict')
        if np.all(max_rank == 0):
            return self.zeros()
        return LabeledMatrix(self.label, truncate_by_count(self.matrix, max_rank, axis=axis),
                             deco=self.deco)

    def _divide(self, lm):
        """
        element-wise division, assuming labels are the same
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1,0], [3,4]]))
        >>> d = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[0.5,2], [3,4]]))
        >>> d._divide(lm).matrix
        array([[0.5, 0. ],
               [1. , 1. ]])
        >>> d.to_sparse()._divide(lm.to_sparse()).matrix.toarray()
        array([[0.5, 0. ],
               [1. , 1. ]])
        >>> d._divide(lm.to_sparse()).matrix.toarray()
        array([[0.5, 0. ],
               [1. , 1. ]])
        >>> d.to_sparse()._divide(lm).matrix.toarray()
        array([[0.5, 0. ],
               [1. , 1. ]])
        """
        matrix = safe_multiply(self.matrix, pseudo_element_inverse(lm.matrix))
        return LabeledMatrix(self.label, matrix,
                             deco=zipmerge(self.deco, lm.deco))

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

    def inverse(self, scalar=1.0):
        """
        >>> lm1 = LabeledMatrix(2*[['b', 'c']], np.array([[1.,2.], [5., 0.]]))
        >>> lm1.inverse(2.0).matrix
        array([[2. , 1. ],
               [0.4, 0. ]])
        >>> lm1.to_sparse().inverse(2.0).matrix.toarray()
        array([[2. , 1. ],
               [0.4, 0. ]])
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,2], [5,4]]))
        >>> lm.to_sparse().inverse(2.0).matrix.toarray()
        array([[2. , 1. ],
               [0.4, 0.5]])
        >>> lm.inverse(2.).matrix
        array([[2. , 1. ],
               [0.4, 0.5]])
        """
        return LabeledMatrix(self.label,
                             pseudo_element_inverse(self.matrix, scalar),
                             deco=self.deco)

    def divide(self, other):
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
        s, o = self.align(other, axes=[(0, 0, False), (1, 1, False)])
        return s._divide(o)

    def _multiply(self, lm):
        """
        Point-wise multiply, assuming that labels are the same
        >>> lm = LabeledMatrix((['b', 'c'], ["x", "y"]), np.array([[1,2], [3,4]]))
        >>> aeq(lm._multiply(lm).matrix, np.array([[ 1,  4], [ 9, 16]]))
        True
        >>> aeq(lm.to_sparse()._multiply(lm.to_sparse()).matrix.toarray(), np.array([[1, 4], [9, 16]]))
        True
        >>> aeq(lm.to_sparse()._multiply(lm).matrix.toarray(), np.array([[1, 4], [9, 16]]))
        True
        >>> aeq(lm._multiply(lm.to_sparse()).matrix.toarray(), np.array([[1, 4], [9, 16]]))
        True
        """
        matrix = safe_multiply(self.matrix, lm.matrix)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def multiply(self, other):
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
        s, o = self.align(other, axes=[(0, 0, False), (1, 1, False)])
        return s._multiply(o)

    def _scalar_multiply(self, scalar):
        """
        >>> lm1 = LabeledMatrix(2*[['b', 'c']], np.array([[1,2], [3,4]]))
        >>> (lm1 * 2).matrix
        array([[2, 4],
               [6, 8]])
        """
        return LabeledMatrix(self.label, self.matrix * scalar, deco=self.deco)

    def _scalar_add(self, scalar):
        """
        >>> lm = LabeledMatrix((['b', 'c'], ["x", "y"]), np.array([[1,0], [0,4]]))
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
        else:
            matrix = self.matrix.copy().astype(max(self.matrix.dtype, type(scalar)))
            matrix[matrix.nonzero()] += scalar
            return LabeledMatrix(self.label, matrix, deco=self.deco)

    def _add(self, other):
        """
        Point-wise sum, assuming that labels are the same
        >>> lm = LabeledMatrix((['b', 'c'], ["x", "y"]), np.array([[1,2], [3,4]]))
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

    def add(self, other):
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
        s, o = self.align(other, axes=[(0, 0, True), (1, 1, True)])
        return s._add(o)

    def _dot(self, lm, top=None, mask=None):
        """
        >>> lm = LabeledMatrix((["x", "y"], ['a', 'b']), np.array([[1,2], [3,4]]))
        >>> lm._dot(lm).matrix
        array([[ 7, 10],
               [15, 22]])
        >>> np.all(lm._dot(lm).matrix ==
        ...        lm.to_sparse()._dot(lm.to_sparse()).matrix.toarray())
        True
        """
        if top is not None:
            return LabeledMatrix((self.row, lm.column),
                                 truncated_dot(self.matrix, lm.matrix, nb=top),
                                 deco=(self.row_deco, lm.column_deco))
        elif mask is not None:
            return LabeledMatrix((self.row, lm.column),
                                 safe_dot(self.matrix, lm.matrix, mask.matrix),
                                 deco=(self.row_deco, lm.column_deco))
        else:
            return LabeledMatrix((self.row, lm.column),
                                 safe_dot(self.matrix, lm.matrix),
                                 deco=(self.row_deco, lm.column_deco))

    def dot(self, other, top=None, mask=None):
        """
        If self and other labels are distinct, then dot restrict columns and rows
        so that they match.
        >>> lm1 = LabeledMatrix((["x", "y"], ['a', 'b', 'c']), np.arange(6).reshape(2,3))
        >>> lm2 = LabeledMatrix((["a", "b", "d"], ['w']), np.arange(3).reshape(3,1))
        >>> lm1.set_deco(row_deco={"x": "X", "y": "Y"})
        >>> lm2.set_deco(column_deco={"w": "W"})
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
        >>> lm1.dot(lm2).reco('x')  #doctest: +NORMALIZE_WHITESPACE
        -------------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score | deco
        -------------------------------------------------------
        x           1            1             1           X
        -------------------
        reco | score | deco
        -------------------
        w      1       W

        >>> mask = slm.restrict_row(['y'])
        >>> aeq(lm1.dot(lm2, mask=mask).matrix, np.array([[4]]))
        True
        """
        if (mask is not None) and (top is not None):
            raise LabeledMatrixException('Cannot provide both "mask" and "top" in dot')
        s, o = self.align(other, axes=[(0, 1, False)])
        if mask is None:
            return s._dot(o, top=top)
        else:
            s, mask = s.align(mask, axes=[(1, 1, False)])
            o, mask = o.align(mask, axes=[(0, 0, False)])
            return s._dot(o, mask=mask)

    def compatibility_renormalization(self, lmcat1, lmcat2, same_factor, not_same_factor):
        """
        >>> from karma.core.labeledmatrix.utils import lm_from_dict
        >>> lm = LabeledMatrix([['a', 'b', 'c'], ['e', 'f', 'r']],
        ...                    np.arange(9).reshape(3,3))
        >>> cat1 = {'a': 'x', 'c': 'y', 'b': 'y'}
        >>> lmcat1 = lm_from_dict(cat1)
        >>> cat2 = {'e': 'x', 'f': 'x', 'r': 'y'}
        >>> lmcat2 = lm_from_dict(cat2)
        >>> lm.compatibility_renormalization(lmcat1, lmcat2, 1, 0).matrix.toarray()
        array([[0., 1., 0.],
               [0., 0., 5.],
               [0., 0., 8.]])
        >>> lm.compatibility_renormalization(lmcat1, lmcat2, 0, 1).matrix.toarray()
        array([[0., 0., 2.],
               [3., 4., 0.],
               [6., 7., 0.]])
        """
        lmcat1 = lmcat1.align(self, axes=[(1, 1, None)])[0]
        lmcat2 = lmcat2.align(self, axes=[(1, 0, None)])[0]
        lmcat1, lmcat2 = lmcat1.align(lmcat2, axes=[(0, 0, False)])
        return LabeledMatrix(self.label,
                             safe_compatibility_renormalization(self.matrix,
                                                                lmcat1.matrix,
                                                                lmcat2.matrix,
                                                                same_factor,
                                                                not_same_factor),
                             deco=self.deco)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

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
        elif isinstance(other, LabeledMatrix):
            return self.add(other)
        else:
            raise LabeledMatrixException("type of 'other' is not understood : {}"
                                         .format(type(other)))

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
        elif isinstance(other, LabeledMatrix):
            return self.multiply(other)
        else:
            raise LabeledMatrixException("type of other is not understood : {}"
                                         .format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1,2], [3,4]]))
        >>> aeq((lm.to_sparse() / lm).matrix, np.ones((2,2)))
        True
        >>> (lm.to_sparse() / 2.0).matrix.toarray()
        array([[0.5, 1. ],
               [1.5, 2. ]])
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
        elif isinstance(other, LabeledMatrix):
            return self.divide(other)
        else:
            raise LabeledMatrixException("type of other is not understood : {}"
                                         .format(type(other)))

    def __eq__(self, other):
        return isinstance(other, LabeledMatrix) \
            and (self.row == other.row) and (self.column == other.column) \
            and aeq(self.matrix, other.matrix)

    def _maximum(self, other):
        return LabeledMatrix(self.label, safe_maximum(self.matrix, other.matrix),
                             deco=self.deco)

    def _minimum(self, other):
        return LabeledMatrix(self.label, safe_minimum(self.matrix, other.matrix),
                             deco=self.deco)

    def maximum(self, other):
        """
        >>> lm1 = LabeledMatrix(2*[['b', 'c', 'd']],
        ...                     np.array([[1,0,5], [0,0,0], [0,1,0]]))
        >>> lm2 = LabeledMatrix(2*[['b', 'c', 'd']],
        ...                     np.array([[1,-2,2], [0,3,0],[0,2,0]]))
        >>> aeq(lm1.maximum(lm2.to_sparse()).matrix, np.array([[1, 0, 5], [0, 3, 0],[0, 2, 0]]))
        True
        """
        s, o = self.align(other, axes=[(0, 0, True), (1, 1, True)])
        return s._maximum(o)

    def minimum(self, other):
        """
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
        s, o = self.align(other, axes=[(0, 0, True), (1, 1, True)])
        return s._minimum(o)

    def max(self, axis=1):
        """
        >>> lm = LabeledMatrix((['c', 'b'], ['x', 'y', 'z']) ,
        ...                    np.array([[0.3, 0.6, 0], [0.75, 0.5, 0]]))
        >>> lm.max().sort().matrix.toarray()
        array([[0.75, 0.  ],
               [0.  , 0.6 ]])
        >>> lm.max().sort().label
        (['b', 'c'], ['b', 'c'])
        >>> lm.max(axis=0).sort().matrix.toarray()
        array([[0.75, 0.  , 0.  ],
               [0.  , 0.6 , 0.  ],
               [0.  , 0.  , 0.  ]])
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
        return lmdiag(self.label[co(axis)], values, sdeco=self.deco[co(axis)],
                      dense_output=False)

    def min(self, axis=1):
        """
        >>> lm = LabeledMatrix((['a', 'b', 'c'], ['x', 'y', 'z']),
        ...                    np.arange(9).reshape(3, 3) + 1)
        >>> lm.min().sort().matrix.toarray().astype(np.int)
        array([[1, 0, 0],
               [0, 4, 0],
               [0, 0, 7]])
        >>> lm.min().sort().label
        (['a', 'b', 'c'], ['a', 'b', 'c'])
        >>> lm.min(axis=0).sort().matrix.toarray().astype(np.int)
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
        return lmdiag(self.label[co(axis)], values, sdeco=self.deco[co(axis)],
                      dense_output=False)

    def sum(self, axis=1):
        """
        Computes sum along axis, returns a diagonal matrix (easier to use in matrix product)
        >>> lm = LabeledMatrix((['a', 'b'], ["x"]), np.array([[5], [25]]))
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
        return lmdiag(self.label[co(axis)], values, sdeco=self.deco[co(axis)],
                      dense_output=False)

    def mean(self, axis=1):
        """
        >>> lm = LabeledMatrix((['a', 'b'], ['x', 'y']), np.array([[5, 5], [25, 75]])).sort()
        >>> lm.mean().matrix.toarray()
        array([[ 5.,  0.],
               [ 0., 50.]])
        >>> lm.mean().label
        (['a', 'b'], ['a', 'b'])
        >>> lm.mean(axis=0).sort().matrix.toarray()
        array([[15.,  0.],
               [ 0., 40.]])
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
        return lmdiag(self.label[co(axis)], values, sdeco=self.deco[co(axis)],
                      dense_output=False)

    def abs(self):
        """
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[0.5, -0.5], [-0.25, 0]]))
        >>> lm.abs().matrix
        array([[0.5 , 0.5 ],
               [0.25, 0.  ]])
        >>> lm.to_sparse().abs().matrix.toarray()
        array([[0.5 , 0.5 ],
               [0.25, 0.  ]])
        """
        return LabeledMatrix(self.label, self.matrix.__abs__(), deco=self.deco)

    def sign(self):
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
        else:
            return LabeledMatrix(self.label, np.sign(self.matrix), deco=self.deco)

    def apply_lambda(self, lambda_):
        """
        >>> mat = np.array([[0.4, 0.5, 0, 0],
        ...                 [0, 0.4, 0, 0],
        ...                 [0, 0, 0.4, 0.5],
        ...                 [0.1, 0, 0, 0.4]])
        >>> lm = LabeledMatrix(2*[['a', 'b', 'c', 'd']], mat)
        >>> bilambda = lambda x: int(x[1] > x[0])
        >>> lm.to_sparse().apply_lambda(bilambda).matrix.toarray()
        array([[0. , 0.5, 0. , 0. ],
               [0. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0.5],
               [0. , 0. , 0. , 0. ]])
        >>> lm.apply_lambda(bilambda).matrix
        array([[0. , 0.5, 0. , 0. ],
               [0. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0.5],
               [0. , 0. , 0. , 0. ]])
        """
        x, y = self.matrix.nonzero()
        factor = np.array([lambda_((self.row[i], self.column[j])) for i, j in izip(x, y)])
        if self.is_sparse:
            data = self.matrix.data * factor
            matrix = KarmaSparse((data, (x, y)), shape=self.matrix.shape)
        else:
            matrix = self.matrix.copy()
            matrix[(x, y)] = matrix[(x, y)] * factor
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def apply_numpy_function(self, function, function_args=None):
        """
        >>> lm = LabeledMatrix(2*[['a', 'b']], np.array([[1.345,2.234], [3.890,4.678]]))
        >>> lm.apply_numpy_function(np.round, [1]).matrix
        array([[1.3, 2.2],
               [3.9, 4.7]])
        >>> lm.apply_numpy_function(np.log).matrix
        array([[0.29639401, 0.8037937 ],
               [1.35840916, 1.54287067]])
        >>> lm.to_sparse().apply_numpy_function(np.log).matrix.toarray()
        array([[0.29639401, 0.8037937 ],
               [1.35840916, 1.54287067]])
        >>> np.all(lm.apply_numpy_function(np.log).apply_numpy_function(np.exp).matrix == lm.matrix)
        True
        """
        if function_args is None:
            function_args = []
        if self.is_sparse:
            matrix = self.matrix.apply_pointwise_function(function, function_args)
        else:
            matrix = self.matrix.copy().astype(np.float)
            matrix[matrix.nonzero()] = function(matrix[matrix.nonzero()], *function_args)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def power(self, power):
        return self.apply_numpy_function(np.power, [power])

    def log1p(self):
        return self.apply_numpy_function(np.log1p)

    def log(self):
        return self.apply_numpy_function(np.log)

    def exp(self):
        return self.apply_numpy_function(np.exp)

    def round(self):
        return self.apply_numpy_function(np.round)

    def trunc(self):
        return self.apply_numpy_function(np.trunc)

    def soft_cutoff(self, threshold):
        """
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y', 'z']),
        ...                    np.array([[3, -6, 9], [7, 5, 10]]))
        >>> lm.soft_cutoff(100).matrix
        array([[0.02955447, 0.        , 0.08606881],
               [0.06760618, 0.04877058, 0.09516258]])
        """
        def np_soft_cutoff(x, threshold):
            return np.maximum(1. - np.exp(-x / threshold), 0)
        return self.apply_numpy_function(np_soft_cutoff, [threshold])

    def clip(self, lower=None, upper=None):
        """
        >>> lm = LabeledMatrix(2*[['b', 'c']], np.array([[0.5, -0.5], [-0.25, 0]]))
        >>> lm.clip(0.1, 0.25).matrix
        array([[0.25, 0.1 ],
               [0.1 , 0.  ]])
        >>> lm.to_sparse().clip(0.1, 0.25).matrix.toarray()
        array([[0.25, 0.1 ],
               [0.1 , 0.  ]])
        """
        return self.apply_numpy_function(np.clip, [lower, upper])

    def logistic(self, shift, coef=1.):
        """
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y', 'z']),
        ...                    np.array([[3, 6, 9], [7, 5, 10]]))
        >>> lm.logistic(6).matrix
        array([[0.04742587, 0.5       , 0.95257413],
               [0.73105858, 0.26894142, 0.98201379]])
        >>> lm.to_sparse().logistic(6).matrix.toarray()
        array([[0.04742587, 0.5       , 0.95257413],
               [0.73105858, 0.26894142, 0.98201379]])
        """
        return self.apply_numpy_function(logit, [shift, coef])

    def reco(self, my_row=None, nb=15, reco_only=False):
        """
        Argmax horizontal columns sorting
        >>> mat = np.array([[4, 5, 0, 1],
        ...                 [5, 4, 1, 0],
        ...                 [0, 1, 4, 5],
        ...                 [1, 0, 3, 4]])
        >>> lm = LabeledMatrix([['w', 'x', 'y', 'z'], ['a', 'b', 'c', 'd']], mat)
        >>> lm.reco("w") #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        w           3            10            1
        ------------
        reco | score
        ------------
        b      5
        a      4
        d      1
        >>> lm.set_deco(row_deco={"w":'q'}, column_deco={"a": "h"})
        >>> lm.reco("w") #doctest: +NORMALIZE_WHITESPACE
        -------------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score | deco
        -------------------------------------------------------
        w           3            10            1           q
        -------------------
        reco | score | deco
        -------------------
        b      5
        a      4       h
        d      1
        """
        h, r = self.reco_df(my_row, nb)
        if not reco_only:
            h.preview()
        r.preview(exclude="entry_key", rows=nb)

    def reco_df(self, my_row=None, nb=20):
        """
        >>> mat = np.array([[4, 5, 0, 1],
        ...                 [0, 0, 0, 0]])
        >>> lm = LabeledMatrix((['a', 'b'], range(4)), mat)
        >>> lm.reco('b') #doctest: +NORMALIZE_WHITESPACE
        All entries are zeros
        ----------------------
        entry_key | nb_nonzero
        ----------------------
        b           0
        ------------
        reco | score
        ------------
        """
        if my_row is None:
            my_row = self.rand_row()
        if my_row not in self.row:
            print "Unknown label"
            return DataFrame(), DataFrame()
        lm_loc = self.restrict_row([my_row])
        try:
            lm_loc = lm_loc.without_zeros()
        except LabeledMatrixException:
            print "All entries are zeros"
            return DataFrame([[my_row, 0]], columns=['entry_key', 'nb_nonzero']), \
                DataFrame(columns=['reco', 'score'])
        my_df = lm_loc.truncate(nb_h=nb)\
                      .to_flat_dataframe('entry_key', 'reco', 'score')\
                      .sort_by('score', reverse=True)
        if self.column_deco:
            deco_lambda = lambda x: self.column_deco.get(x, "")
            my_df['deco'] = my_df.build_callable_column(deco_lambda, 'reco')
        # head
        head = DataFrame()
        head['entry_key'] = Column([my_row])
        nb_nonzero = len(lm_loc.column)
        head['nb_nonzero'] = Column([nb_nonzero])
        if nb_nonzero > 0:
            head['total_score'] = Column([safe_sum(lm_loc.matrix)])
            head['min_score'] = Column([safe_min(lm_loc.matrix)])
        if self.row_deco:
            head['deco'] = Column([self.row_deco.get(my_row, "")])
        return head, my_df

    def dict_argmax(self):
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

    def dict_max(self, axis=1):
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
        return dict(izip(lm_nonzero.label[co(axis)], lm_nonzero.matrix.max(axis=axis)))

    @enforce_lib_closure
    def to_karmacode(self, inp, output, default="zero"):
        """
        >>> lm = LabeledMatrix((['a', 'b'], ['x', 'y', 'z']), np.arange(6).reshape(2,3))
        >>> df = DataFrame({'col': ['c', 'b', 'b', 'a']})
        >>> kc_dense = lm.to_dense().to_karmacode('col', 'dense_vector')
        >>> kc_sparse = lm.to_sparse().to_karmacode('col', 'sparse_vector')
        >>> df += kc_dense
        >>> df += kc_sparse
        >>> aeq(df['dense_vector'][:], df['sparse_vector'][:])
        True
        >>> df['dense_vector'][:].shape
        (4, 3)
        >>> df['dense_vector'].coordinates
        ['x', 'y', 'z']
        >>> df['dense_vector'][:]
        array([[0., 0., 0.],
               [3., 4., 5.],
               [3., 4., 5.],
               [0., 1., 2.]])
        >>> type(kc_dense[0].mapping[0])
        <type 'list'>
        """
        kc, inputs, coordinates = KarmaCode([]), (inp,), self.column.list
        if not self.is_sparse:
            kc += Map(inputs, output, (self.row.list, self.matrix),
                        default_row(self.matrix, default), coordinates=coordinates)
        else:
            if default != 'zero':
                raise ValueError('Sparse LabeledMatrix supports only *zero* default values')

            vector_size = len(self.row)
            kc += Map(inputs, '_indices', mapping=self.row._index.copy(), default=vector_size)
            kc += KroneckerVector(('_indices',), '_one_hot_vector',
                                  vector_size=vector_size, output_format='sparse',
                                  coordinates=self.row.list)
            kc += Dot(('_one_hot_vector',), output,
                      rightfactor=self.matrix, coordinates=coordinates)
        return kc.with_outputs(output)

    def to_flat_dataframe(self, row="col0", col="col1", dist="similarity", **kwargs):
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
        >>> lm.to_sparse().to_flat_dataframe().sort_by('similarity', reverse=True)\
                          .preview() #doctest: +NORMALIZE_WHITESPACE
        ------------------------
        col0 | col1 | similarity
        ------------------------
        a      y      5.0
        a      x      4.0
        b      y      3.0
        c      z      2.0
        d      x      1.0
        """
        deco_row = kwargs.get('deco_row', None)
        deco_col = kwargs.get('deco_col', None)

        row_indices, col_indices = self.matrix.nonzero()
        if not self.is_sparse:
            values = self.matrix[row_indices, col_indices]
        else:
            values = self.matrix.data.copy()  # for sparse this is fast since order is the same
        data = DataFrame()
        data[row] = Column(take_indices(self.row, row_indices))
        data[col] = Column(take_indices(self.column, col_indices))
        data[dist] = Column(values)
        if deco_row:
            data = data\
                .with_instruction_column(deco_row, Map((row,), '', self.row_deco, ""))
        if deco_col:
            data = data\
                .with_instruction_column(deco_col, Map((col,), '', self.column_deco, ""))
        return data

    def to_list_dataframe(self, col="col", prefix="list_of_", exclude=False):
        """
        Return a DataFrame with columns col, list_of_col.
        For each row, sort the non zero values and return the column labels as list_of_col
        >>> mat = np.array([[4, 0, 0, 0],
        ...                 [5, 4, 1, 0],
        ...                 [0, 1, 4, 5],
        ...                 [1, 0, 5, 4]])
        >>> lm = LabeledMatrix(2*[['a', 'b', 'c', 'd']], mat)
        >>> lm.to_list_dataframe(exclude=True).preview() #doctest: +NORMALIZE_WHITESPACE
        -----------------
        col | list_of_col
        -----------------
        b     ['a', 'c']
        c     ['d', 'b']
        d     ['c', 'a']
        >>> lm.to_sparse().to_list_dataframe(exclude=True).preview()
        ...             #doctest: +NORMALIZE_WHITESPACE
        -----------------
        col | list_of_col
        -----------------
        b     ['a', 'c']
        c     ['d', 'b']
        d     ['c', 'a']
        >>> lm = LabeledMatrix([['a', 'b', 'c', 'd'], ['w', 'x', 'y', 'z']], mat)
        >>> lm.to_list_dataframe(exclude=True).preview() #doctest: +NORMALIZE_WHITESPACE
        ---------------------
        col | list_of_col
        ---------------------
        a     ['w']
        b     ['w', 'x', 'y']
        c     ['z', 'y', 'x']
        d     ['y', 'z', 'w']
        """
        if self.is_sparse:
            matrix = self.matrix.tocsr()
            double = [[x, [self.column[y]
                           for y in matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]
                      [np.argsort(matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]])[::-1]]
                      if not (exclude * (x == self.column[y]))]]
                      for i, x in enumerate(self.row)]
        else:
            asort = self.matrix.argsort(axis=1)[:, ::-1]
            double = [[x, [self.column[y] for y in asort[i] if
                      self.matrix[i, y] and not (exclude * (x == self.column[y]))]]
                      for i, x in enumerate(self.row)]
        return DataFrame([x for x in double if x[1]],
                         columns=[col, prefix + col]).sort_by(col)

    def to_vectorial_dataframe(self, col="col0", vector="col1", deco=False):
        """
        >>> mat = np.array([[4, 5, 0],
        ...                 [0, 3, 0],
        ...                 [0, 0, 0],
        ...                 [1, 0, 0]])
        >>> lm = LabeledMatrix((['a', 'b', 'c', 'd'], ['x', 'y', 'z']), mat, deco=({'a': 'AA'}, {'y': 'YY'}))
        >>> lm.to_vectorial_dataframe().preview() #doctest: +NORMALIZE_WHITESPACE
        -------------------------------
        col0 | col1:x | col1:y | col1:z
        -------------------------------
        a      4        5        0
        b      0        3        0
        c      0        0        0
        d      1        0        0
        >>> lm.to_vectorial_dataframe(deco=True).preview() #doctest: +NORMALIZE_WHITESPACE
        --------------------------------
        col0 | col1:x | col1:YY | col1:z
        --------------------------------
        AA     4        5         0
        b      0        3         0
        c      0        0         0
        d      1        0         0
        >>> lm.to_sparse().to_vectorial_dataframe().preview() #doctest: +NORMALIZE_WHITESPACE
        ---------------------
        col0 | col1
        ---------------------
        a      y: 5.0, x: 4.0
        b      y: 3.0
        c
        d      x: 1.0
        """
        if deco:
            columns = [self.column_deco.get(col_, col_) for col_ in self.column.list]
            rows = [self.row_deco.get(row_, row_) for row_ in self.row.list]
        else:
            columns = self.column.list  # to copy ?
            rows = self.row.list
        df = DataFrame(OrderedDict([(col, rows), (vector, self.matrix)]))
        df[vector].coordinates = columns
        return df

    def normalize(self, axis=1, norm="l1"):
        trans_matrix = normalize(self.matrix.copy(), axis=axis, norm=norm)
        return LabeledMatrix(self.label, trans_matrix, deco=self.deco)

    # todo: remove the mixpow parameter
    # how about naming it 'bayesian_normalization'?
    def transition(self, norm="l1", invpow=1, invlog=0., threshold=10, width=2., mixpow=0,
                   renorm=True):
        """
        Very special (like bayesian) normalization for square LM.
        Warning: self - self.diagonal() should be made before.
        >>> lm = LabeledMatrix(2*[['b', 'c', 'd']], np.array([[4, 4, 1], [2, 5, 2], [2, 1., 3]]))
        >>> np.round((lm - lm.diagonal()).transition(threshold=1., width=0.2).matrix, 3)
        array([[0.   , 0.706, 0.294],
               [0.429, 0.   , 0.571],
               [0.714, 0.286, 0.   ]])
        """
        matrix = normalize(self.matrix, axis=0, norm=norm, invpow=invpow, invlog=invlog,
                           threshold=threshold, width=width)
        if renorm:
            matrix = normalize(matrix, norm=norm, axis=1)
        if mixpow != 0:
            matrix = safe_multiply(self.matrix ** mixpow, matrix)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def connected_components(self, connection="weak"):
        """
        >>> mat = np.array([[1, 1, 0, 0, 0],
        ...                 [0, 1, 0, 0, 1],
        ...                 [0, 0, 1, 0, 0],
        ...                 [0, 0, 1, 1, 0],
        ...                 [0, 1, 0, 0, 1]])
        >>> lm = LabeledMatrix([range(5)]*2, mat)
        >>> KarmaSetup.verbose = False
        >>> lm.connected_components().label
        ([0, 1, 2, 3, 4], [0, 1])
        >>> lm.connected_components().matrix.toarray().astype(np.int)
        array([[1, 0],
               [1, 0],
               [0, 1],
               [0, 1],
               [1, 0]])
        >>> lm.to_sparse().connected_components("strong").sort().matrix.toarray().astype(np.int)
        array([[0, 1, 0, 0],
               [1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [1, 0, 0, 0]])
        """
        if not self.is_square():
            raise LabeledMatrixException("Matrix must be squared")
        matrix = self.matrix.to_scipy_sparse(copy=False) if self.is_sparse else self.matrix
        nn, lab = connected_components(matrix, directed=True, connection=connection)
        if KarmaSetup.verbose:
            print "Number of components: {}".format(nn)
        lm = lm_occurence(self.row, lab)
        lm.set_deco(row_deco=self.row_deco)
        return lm

    def affinity_clusters(self, preference=None, max_iter=200):
        """
        >>> similarity = np.array([[3, 5, 1, 1],
        ...                        [5, 2, 2, 1],
        ...                        [1, 1, 2, 6],
        ...                        [1, 2, 4, 3]])
        >>> lm = LabeledMatrix((['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd']), similarity)
        >>> lm_ac = lm.affinity_clusters().sort()
        >>> lm_ac.label
        (['a', 'b', 'c', 'd'], ['a', 'd'])
        >>> lm_ac.matrix.toarray().astype(np.int)
        array([[1, 0],
               [1, 0],
               [0, 1],
               [0, 1]])
        """
        if not self.is_square:
            raise LabeledMatrixException('Works only on squared matrix')
        labels = affinity_propagation(self.matrix, preference, max_iter=max_iter, damping=0.6)
        lm = lm_occurence(self.row, take_indices(self.row, labels))
        lm.set_deco(*self.deco)
        return lm

    def spectral_clusters(self, k=10):
        clust = SpectralClustering(n_clusters=k, n_init=10, affinity="precomputed",
                                   n_neighbors=3, assign_labels='kmeans')
        lab = clust.fit_predict(self.matrix)
        lm = lm_occurence(self.row, lab)
        lm.set_deco(row_deco=self.row_deco)
        return lm

    def similarity(self, other=None, cutoff=0.005, nb_keep=200, top=1000, cumtop=0.02):
        """
        See matrix_utils.buddies_matrix.
        >>> mat = np.array([[0.4, 0.5, 0, 0.1, 0],
        ...                 [0, 0.1, 0.4, 0.5, 0],
        ...                 [0, 0, 0, 0, 0],
        ...                 [0.1, 0, 0.5, 0.4, 0]])
        >>> lm = LabeledMatrix((range(4), range(5)),  mat).to_sparse()
        >>> lm.similarity(nb_keep=2).matrix.toarray()
        array([[1. , 0.2, 0. , 0. ],
               [0. , 1. , 0. , 0.8],
               [0. , 0. , 0. , 0. ],
               [0. , 0.8, 0. , 1. ]])
        >>> lm.similarity().label
        ([0, 1, 2, 3], [0, 1, 2, 3])
        >>> res = lm.similarity(other=lm.restrict_row([1, 2]), nb_keep=2)
        >>> res.matrix.toarray()
        array([[0.2],
               [1. ],
               [0.8]])
        >>> res.label
        ([0, 1, 3], [1])
        """
        if other is None:
            return LabeledMatrix((self.row, self.row),
                                 buddies_matrix(self.matrix, cutoff=cutoff, nb_keep=nb_keep,
                                                top=top, cumtop=cumtop),
                                 deco=(self.row_deco, self.row_deco))
        else:
            lm, other = self.align(other, axes=[(0, 0, False)])
            lm = lm.without_zeros(axis=1)
            other = other.without_zeros(axis=1)
            return LabeledMatrix((lm.row, other.row),
                                 pairwise_buddy(lm.matrix, other.matrix,
                                                cutoff=cutoff, nb_keep=nb_keep),
                                 deco=(lm.row_deco, other.row_deco))

    def similarity_query(self, my_row=None, nb=20, top=5000):
        """
        TODO : this method needs rewritting
        >>> lm1 = LabeledMatrix((['a', 'b', 'c'], ["x", "y"]),
        ...                     np.array([[5, 5], [7, 3], [3, 7]]))
        >>> lm1.similarity_query('b') #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        b           3            24.0          6.0
        ------------
        reco | score
        ------------
        b      10.0
        a      8.0
        c      6.0
        >>> lm1.to_sparse().similarity_query('c') #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        c           3            24.0          6.0
        ------------
        reco | score
        ------------
        c      10.0
        a      8.0
        b      6.0
        """
        if my_row is None:
            my_row = self.rand_row()
        if my_row not in self.row:
            print "Unknown label"
        else:
            self.restrict_row([my_row]).similarity(self, top=top).reco(my_row, nb=nb)

    def jaccard(self):
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
        total = source.sum(axis=1)
        union = inter.nonzero_mask()._dot(total)\
                     ._add(total._dot(inter.nonzero_mask()))._add(-inter)
        return 1. * inter / union

    def buddies(self, rows, batch_size=100, max_iter=False):
        """
        >>> from scipy.sparse import diags
        >>> lm = LabeledMatrix((range(5), range(5)),
        ...                     diags([np.ones(5), np.ones(4), 1.1 * np.ones(4)],
        ...                              [0, -1, 1]).toarray()).normalize()
        >>> list(lm.buddies([1])) #doctest: +ELLIPSIS
        [(2, 0.354...), (1, 0.322...), (0, 0.322...), (3, 0.104...), (4, 0.010...)]

        # testing asymmetric case
        >>> [buddy for buddy, similarity in lm.restrict_row([1, 0, 2]).buddies([1])]
        [2, 0, 1, 3]
        >>> list(lm.restrict_column([1, 0, 2, 3]).buddies([1])) #doctest: +ELLIPSIS
        [(2, 0.354...), (0, 0.322...), (1, 0.322...), (3, 0.104...)]
        >>> mat = np.repeat(np.atleast_2d(np.arange(10)), 10, axis=0)
        >>> lm = LabeledMatrix([range(10), range(10)], mat)
        >>> [buddy for buddy, similarity in lm.buddies([1], batch_size=2)]
        [9, 8, 7, 6, 5, 4, 3, 2, 1]
        """
        source = self.symmetrize_label()
        matrix = source.matrix
        vector = np.atleast_2d(source.restrict_row(rows).matrix.mean(axis=0))

        def extend_vector(trunc_size, vector, matrix):
            vector_trunc = truncate_by_count(vector, axis=1, max_rank=trunc_size)
            old_min = np.min(vector_trunc[vector_trunc.nonzero()])
            return safe_maximum(vector, normalize(safe_dot(vector_trunc, matrix), axis=None, norm="linf") * old_min)

        def nonzero_argsort(batch_size, vector):
            batch_indices = np.argsort(vector)[-batch_size:][::-1]
            batch_indices = batch_indices[vector[batch_indices] != 0]
            return batch_indices

        def ordered_merge(new_array, old_array):
            return new_array[~np.in1d(new_array, old_array)]
        # Q.? : Should we first return the passed rows and next lock for any extension?
        total_indices = nonzero_argsort(batch_size, np.squeeze(vector))
        for i, col in enumerate(source.column.select(total_indices)):
            yield col, vector[0, total_indices[i]]

        it = 0
        current_batch_size = batch_size
        current_indices = np.array([], dtype=np.int32)
        while not max_iter or (it < max_iter):
            it += 1
            total_indices = np.hstack([total_indices, current_indices])
            vector = extend_vector(current_batch_size, vector, matrix)
            current_batch_size = min(current_batch_size + batch_size * it, vector.shape[1])
            current_indices = ordered_merge(nonzero_argsort(current_batch_size,
                                                            np.squeeze(vector)),
                                            total_indices)
            if len(current_indices) == 0:
                break
            for i, col in enumerate(source.column.select(current_indices)):
                yield col, vector[0, current_indices[i]]

    @use_seed()
    def nmf(self, rank, max_model_rank=None, max_iter=None, svd_init=False):
        """
        >>> from karma.synthetic.labeledmatrix import rand
        >>> m = rand(seed=12, density=0.5).to_dense()
        >>> w, h = m.dot(m.transpose()).nmf(3)
        >>> (w.dot(h.transpose()) - m.dot(m.transpose())).abs().matrix.sum() < 0.05
        True
        """
        if (type(rank) is int) or (rank is None):
            ranks = (rank,)
        else:
            ranks = rank
        ww, hh = [], []

        if not self.is_sparse and keep_sparse(self.matrix):
            matrix = self.to_sparse().matrix
        else:
            matrix = self.matrix

        for rank in ranks:
            w, h = nmf(matrix, rank=rank, max_model_rank=max_model_rank, max_iter=max_iter, svd_init=svd_init)
            ww.append(w)
            hh.append(h)
        www, hhh = np.hstack(ww), np.hstack(hh)
        lmw = LabeledMatrix((self.row, range(www.shape[1])), www,
                            deco=(self.row_deco, {}))
        lmh = LabeledMatrix((self.column, range(hhh.shape[1])), hhh,
                            deco=(self.column_deco, {}))
        return lmw, lmh

    def nmf_fold(self, right_factor, max_iter=30):
        """
        >>> from karma.synthetic.labeledmatrix import rand
        >>> w = rand((10, 5), density=0.5, seed=100).to_dense()
        >>> h = rand((7, 5), density=0.5, seed=100)

        >>> from karma.core.labeledmatrix.utils import hstack
        >>> w, h = hstack([w]), hstack([h]) # to get default column labels
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

    def co_clustering(self, ranks, max_iter=120, nb_preruns=30, pre_iter=4):
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
        lmw = lm_occurence(self.row, w)
        lmw.set_deco(row_deco=self.row_deco)
        lmh = lm_occurence(self.column, h)
        lmw.set_deco(column_deco=self.column_deco)
        return lmw, lmh

    def nmf_proxy(self, rank, max_iter=120, svd_init=True):
        """
        >>> from karma.synthetic.labeledmatrix import rand
        >>> m = rand(seed=12, density=0.5)
        >>> lm = m.dot(m.transpose())
        >>> (lm - lm.nmf_proxy(3)).abs().matrix.sum() < 0.1
        True
        >>> (lm - lm.nmf_proxy([3, 4, 3])).abs().matrix.sum() < 0.1
        True
        """
        if (type(rank) is int) or (rank is None):
            ranks = (rank,)
        else:
            ranks = rank
        ww, hh = [], []

        for rank in ranks:
            w, h = nmf(self.matrix, rank=rank, max_iter=max_iter, svd_init=svd_init)
            ww.append(w)
            hh.append(h)
        matrix = 1. * safe_dot(np.hstack(ww), np.hstack(hh).transpose(),
                               mat_mask=self.matrix) / len(ranks)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def svd(self, rank, randomized=True):
        """
        If randomized=False, scipy.sparse.linalg.svds solver will be used
        For randomized=True, the result will be not exact but obtained by computing shorter time.
        For randomized version, see:
        http://arxiv.org/pdf/0909.4061.pdf
        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        >>> lm = LabeledMatrix((range(5), range(4)), np.arange(20).reshape(5,4))
        >>> u, w = lm.svd(2, randomized=False)
        >>> u.matrix
        array([[ 1.23232132,  0.45176491],
               [ 0.79741909,  1.58301848],
               [ 0.36251687,  2.71427205],
               [-0.07238536,  3.84552562],
               [-0.50728759,  4.97677919]])
        >>> u.label
        ([0, 1, 2, 3, 4], [0, 1])
        >>> w.label
        ([0, 1, 2, 3], [0, 1])
        >>> u.dot(w.transpose()) == lm
        True
        >>> u, w = lm.svd(2, randomized=True)
        >>> u.matrix
        array([[-0.45176491, -1.23232132],
               [-1.58301848, -0.79741909],
               [-2.71427205, -0.36251687],
               [-3.84552562,  0.07238536],
               [-4.97677919,  0.50728759]])
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
        lm_u = LabeledMatrix((self.row, range(rank)), u, (self.row_deco, {}))
        lm_w = LabeledMatrix((self.column, range(rank)), w, (self.column_deco, {}))
        return lm_u, lm_w

    def svd_proxy(self, rank, randomized=True):
        """
        >>> lm = LabeledMatrix((range(5), range(4)), np.arange(20).reshape(5,4))
        >>> lm.svd_proxy(2) == lm
        True
        >>> aeq(lm.svd_proxy(1).matrix, lm.svd_proxy(2).matrix)
        False
        """
        if type(rank) is int:
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
        for rank in ranks:
            u, s, w = method(self_matrix, rank)
            uu.append(u.dot(np.diag(s)))
            ww.append(w)
        matrix = 1. * safe_dot(np.hstack(uu), np.vstack(ww),
                               mat_mask=self.matrix) / len(ranks)
        return LabeledMatrix(self.label, matrix, deco=self.deco)

    def tail_clustering(self, weight, k, min_density=0.):
        """
        >>> v = np.array([[7, 0, 3],
        ...               [5, 0, 5],
        ...               [3, 3, 4],
        ...               [0, 10, 0],
        ...               [0, 9, 1]])
        >>> lm = LabeledMatrix((range(5), range(3)), v).to_dense()
        >>> weight = lmdiag(range(5), [1, 2, 4, 4, 1])
        >>> lm.tail_clustering(weight, 2).dict_argmax()
        {0: 2, 1: 2, 2: 2, 3: 3, 4: 3}
        >>> lm.tail_clustering(weight, 2).to_sparse().dict_argmax()
        {0: 2, 1: 2, 2: 2, 3: 3, 4: 3}
        """
        mults = weight.sum(axis=1).align(self, axes=[(1, 1, None), (0, 1, None)])[0]\
                      .matrix.diagonal()
        if self.is_sparse:
            labels = sparse_tail_clustering(self.matrix, mults, k, min_density)
        else:
            labels = tail_clustering(self.matrix, mults, k)
        lm = lm_occurence(self.row, take_indices(self.row, labels))
        lm.set_deco(self.row_deco, self.row_deco)
        return lm

    def hierarchical_clustering(self, weight, k):
        """
        Returns LabeledMatrix with clusters' characteristic vectors for each row
        >>> v = np.array([[7, 0, 3],
        ...               [5, 0, 5],
        ...               [3, 3, 4],
        ...               [0, 10, 0],
        ...               [0, 9, 1]])
        >>> lm = LabeledMatrix((range(5), range(3)), v)
        >>> weights = lmdiag(range(5), [1, 2, 4, 4, 1])
        >>> lm.hierarchical_clustering(weights, 2).dict_argmax()
        {0: 2, 1: 2, 2: 2, 3: 3, 4: 3}
        """
        if weight is not None:
            weight = weight.sum(axis=1).align(self, axes=[(1, 1, None), (0, 1, None)])[0].matrix.diagonal()

        labels = WardTree(np.asarray(self.matrix), weights=weight, n_clusters=k).build_labels()
        lm = lm_occurence(self.row, take_indices(self.row, labels))
        lm.set_deco(self.row_deco, self.row_deco)

        return lm

    def sort_by_hierarchical_clustering(self):
        """
        Returns LabeledMatrix with rows and columns reordered in respect to the Ward tree leaves labels
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
        idx = np.asarray(sorted(range(len(lm.row)), key=lambda i: order_coding[i]))

        if lm.is_square():
            return lm._take_on_row(idx)._take_on_column(idx)
        else:
            return lm._take_on_row(idx)

    def anomaly(self, skepticism=1.):
        """
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

    def deduplicate_by(self, exclude, nb=1, max_iter=20):
        """
        exclude is assumed to be square LM
        >>> matp = np.array([[10, 3, 4, 5, 2, 1, 1, 2]])
        >>> lmp = LabeledMatrix([['matelas'], ['matelas', 'drap1', 'drap2', 'drap3', 'fifa', 'gtaIV',
        ...                       'gtaV', 'xbox']], matp)
        >>> groups = {'matelas': 'matelas',  'drap1': "drap", 'drap2': "drap", 'drap3': "drap",
        ...           'xbox': "xbox", 'gtaV': "game", 'gtaIV': "game", 'fifa': "game"}
        >>> from karma.core.labeledmatrix.utils import lm_block
        >>> lmg = lm_block(groups)
        >>> lmp.reco('matelas') #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        matelas     8            28            1
        ---------------
        reco    | score
        ---------------
        matelas   10
        drap3     5
        drap2     4
        drap1     3
        xbox      2
        fifa      2
        gtaV      1
        gtaIV     1

        >>> lmp.deduplicate_by(lmg).reco('matelas') #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        matelas     4            19.0            2.0
        ---------------
        reco    | score
        ---------------
        matelas   10.0
        drap3     5.0
        xbox      2.0
        fifa      2.0

        >>> lmp.to_sparse().deduplicate_by(lmg, nb=2).reco('matelas') #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        matelas     6            24.0          1.0
        ---------------
        reco    | score
        ---------------
        matelas   10.0
        drap3     5.0
        drap2     4.0
        xbox      2.0
        fifa      2.0
        gtaV      1.0
        >>> lmp.deduplicate_by(lmg, nb=3).reco_df('matelas')[1]['reco'] == \
            lmp.reco_df('matelas')[1]['reco']
        True
        >>> mat = np.array([[0.4, 0.5, 0, 0.1, 0],
        ...                 [0.1, 0, 0.5, 0.4, 0]])
        >>> lm = LabeledMatrix((range(2), range(5)),  mat)
        >>> groups = LabeledMatrix((range(5), range(5)),  np.arange(25).reshape(5, 5) % 2).to_sparse()
        >>> np.array(lm.deduplicate_by(groups, nb=1).matrix)
        array([[0. , 0.5, 0. , 0.1, 0. ],
               [0.1, 0. , 0.5, 0. , 0. ]])
        """
        exclude = exclude.align(self, axes=[(1, 0, None)])[0].nonzero_mask()
        rest, result = self.copy(), self.zeros()
        it = 0
        while rest.nnz() and it < max_iter:
            it += 1
            current_leader = rest.truncate(nb_h=1)
            rest = rest - current_leader
            to_remove = rest * current_leader.nonzero_mask()._dot(exclude)
            to_remove = to_remove - to_remove.truncate(nb_h=nb-1)
            rest = rest - to_remove
            result = result + current_leader
        return result + rest

    def category_deduplication(self, category, nb_keep=1):
        """
        "category" should be LM with (0,1) values such that category.row ~ self.column
        and category.column ~ category to which it belongs
        >>> from karma.core.labeledmatrix.utils import lm_from_dict
        >>> lmp = LabeledMatrix([['matelas'], ['matelas', 'drap1', 'drap2', 'drap3', 'fifa', 'gtaV',
        ...                       'gtaIV', 'xbox']], np.array([[10, 3, 4, 5, 2, 1, 1, 2]]))
        >>> groups = {'matelas': 'matelas',  'drap1': "drap", 'drap2': "drap", 'drap3': "drap",
        ...           'gtaV': "game", 'gtaIV': "game", 'fifa': "game"}
        >>> lm_cat = lm_from_dict(groups)
        >>> lmp.category_deduplication(lm_cat).reco() #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        matelas     4            19.0          2.0
        ---------------
        reco    | score
        ---------------
        matelas   10.0
        drap3     5.0
        xbox      2.0
        fifa      2.0
        """
        category = category.align(self, axes=[(1, 0, None)])[0]
        category_list = KarmaSparse(category.matrix, copy=False)\
            .argmax(axis=1, only_nonzero=True)
        new_matrix = KarmaSparse(self.matrix, copy=False)\
            .truncate_by_count_by_groups(category_list, nb_keep)
        return LabeledMatrix(self.label, new_matrix, deco=self.deco)

    def substitute_eliminating(self, dissimilarity, max_iter=50, p=1):
        """
        Multiplicative version
        >>> matp = np.array([[6, 4, 3, 2]])
        >>> lm = LabeledMatrix([['matelas'], ['drap1', 'drap2', 'drap3', 'drap4']], matp)
        >>> dis_mat = np.array([[0, 0.5, 0.1, 0],
        ...                     [0, 0, 0.7, 0],
        ...                     [0, 0, 0, 0.2],
        ...                     [0.1, 0, 0, 0]])
        >>> groups = LabeledMatrix(2 * [['drap1', 'drap2', 'drap3', 'drap4']], dis_mat).to_sparse()
        >>> lm.substitute_eliminating(groups).reco() #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        matelas     4            10.21         0.21
        -------------
        reco  | score
        -------------
        drap1   6.0
        drap4   2.0
        drap2   2.0
        drap3   0.21

        """
        dissim_mat = dissimilarity.align(self, axes=[(1, 0, None), (0, 0, None)])[0]\
                                  .to_sparse().matrix.copy()
        dissim_mat = dissim_mat ** p
        rest = self.matrix.copy()
        result = 0 * rest
        if is_scipysparse(result):
            result.eliminate_zeros()
        it = 0
        # 0 dissimilarity for sparse matrix will considered as 1.
        while it < max_iter and (rest.nnz if self.is_sparse else np.count_nonzero(rest)):
            it += 1
            current_leader = truncate_by_count(rest, max_rank=1, axis=1)
            result = result + current_leader
            rest = rest - current_leader
            cl_mask = nonzero_mask(current_leader)
            rest = rest - safe_multiply(rest, safe_dot(cl_mask, nonzero_mask(dissim_mat),
                                        dense_output=False))\
                + safe_multiply(rest, safe_dot(cl_mask, dissim_mat, dense_output=False))
        return LabeledMatrix(self.label, result + rest, deco=self.deco)

    def soft_group_concurence(self, group, decay=0.5, max_period=4, max_group=15, groupcutoff=0):
        """
        >>> matp = np.array([[6, 1, 1, 1.3, 1.2, 2, 3]])
        >>> lmp = LabeledMatrix([['matelas'],
        ...                     ['matelas', 'drap1', 'drap2', 'drap3',
        ...                      'drap4', 'xbox', 'gtaV']], matp)
        >>> lmp.reco() #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        matelas     7            15.5          1.0
        ---------------
        reco    | score
        ---------------
        matelas   6.0
        gtaV      3.0
        xbox      2.0
        drap3     1.3
        drap4     1.2
        drap2     1.0
        drap1     1.0

        >>> from karma.core.labeledmatrix.utils import lm_from_dict
        >>> group = lm_from_dict({'matelas': 'matelas',  'drap1': "drap", 'drap2': "drap",
        ...                       'drap3': "drap", 'drap4': "drap", 'fifa': "game",
        ...                       'xbox': "xbox", 'gtaV': "game", 'gtaIV': "game"})
        >>> lmp.soft_group_concurence(group, max_group=3, decay=0.5).reco() #doctest: +NORMALIZE_WHITESPACE
        ------------------------------------------------
        entry_key | nb_nonzero | total_score | min_score
        ------------------------------------------------
        matelas     6            17.4375       0.5625
        ----------------
        reco    | score
        ----------------
        matelas   6.0
        drap3     4.5
        gtaV      3.0
        drap4     2.25
        drap1     1.125
        drap2     0.5625

        """
        # this alignment can be simplified
        group = group.align(self.without_zeros(), axes=[(1, 0, None)])[0].without_zeros()
        self_group = self.dot(group).truncate(nb_h=max_group,
                                              cutoff=groupcutoff).without_zeros()
        group = group.align(self_group, axes=[(0, 0, None)])[0].without_zeros()
        cut_self = self.align(self_group, axes=[(1, 1, None)])[0]
        cut_self = cut_self.align(group, axes=[(0, 1, None)])[0]

        self_iter = cut_self.matrix.copy()
        partial_sum = 0. * self_iter

        for p in xrange(max_period):
            group_rest = self_group.matrix.copy()
            for it in xrange(max_group):
                if number_nonzero(group_rest) == 0:
                    break
                group_leader = truncate_by_count(group_rest, axis=1, max_rank=1)
                group_rest = group_rest - group_leader
                current_leader = safe_multiply(self_iter,
                                               safe_dot(nonzero_mask(group_leader),
                                                        group.matrix.transpose()))
                current_leader = truncate_by_count(current_leader, axis=1, max_rank=1)
                self_iter = self_iter - current_leader
                partial_sum += decay ** min(p, 10) * safe_multiply(safe_dot(group_leader, group.matrix.transpose()),
                                                                   nonzero_mask(current_leader))

        return LabeledMatrix(cut_self.label, partial_sum, deco=cut_self.deco)

    def logistic_regression(self, response, C=1e8, penalty='l2', max0=100000, max1=5000):
        """
        >>> mat = np.array([[-3., -2.],
        ...                 [ 1.,  1.],
        ...                 [ 4.,  1.],
        ...                 [ 1.,  6.],
        ...                 [ 0., -1.],
        ...                 [ 4.,  4.],
        ...                 [ 0.,  3.],
        ...                 [-2.,  1.],
        ...                 [ 3.,  1.],
        ...                 [ 0., -1.]])
        >>> y = np.array([[1, 0.0, 0.0, 1, 0.0, 1, 0.0, 0.0, 1]])
        >>> yy = np.vstack([y, 1 - y]).transpose()
        >>> lm = LabeledMatrix((range(10), ['a', 'b']), mat)
        >>> response = LabeledMatrix((range(9), ['y', 'ybar']), yy)
        >>> lm_beta, lm_intercept = lm.to_sparse().logistic_regression(response)
        >>> lm_pred = lm.apply_logistic_model(lm_beta, lm_intercept)
        >>> lm_pred.matrix
        array([[0.23668904, 0.76331096],
               [0.40982753, 0.59017247],
               [0.42858583, 0.57141417],
               [0.69164498, 0.30835502],
               [0.2974724 , 0.7025276 ],
               [0.60249609, 0.39750391],
               [0.51964895, 0.48035105],
               [0.39132797, 0.60867203],
               [0.42230742, 0.57769258],
               [0.2974724 , 0.7025276 ]])
        >>> lm_beta.matrix
        array([[ 0.02568505,  0.23449988],
               [-0.02568505, -0.23449988]])
        >>> lm_intercept.matrix
        array([[-0.62486334],
               [ 0.62486334]])
        >>> lm_pred_dense = lm.apply_logistic_model(*lm.logistic_regression(response))
        >>> lm_pred_dense == lm_pred
        True
        """
        # setting model
        lr = LogisticRegression(C=C, penalty=penalty, tol=1e-5, fit_intercept=True)

        source, response = self.align(response, ([1, 1, False],))
        response = response.nonzero_mask()

        X = source.matrix.to_scipy_sparse(copy=False) if self.is_sparse else source.matrix
        Y = response.matrix.transpose()
        Y = Y.tocsr() if is_karmasparse(Y) else Y
        beta, intercept = [], []
        # to train each response column separately
        for i in xrange(Y.shape[0]):
            y = Y[i].toarray()[0] if response.is_sparse else Y[i]
            select = np.hstack([np.random.permutation(np.argwhere(y == 0).T[0])[:max0],
                                np.random.permutation(np.argwhere(y != 0).T[0])[:max1]])
            # TODO: unslim coef?
            lr.fit(X[select], y[select])
            beta.append(lr.coef_)
            intercept.append(lr.intercept_)
        beta, intercept = np.vstack(beta), np.vstack([intercept])
        lm_beta = LabeledMatrix((response.column, source.column), beta,
                                deco=(response.column_deco, source.column_deco))
        lm_intercept = LabeledMatrix((response.column, range(1)), intercept)
        return lm_beta, lm_intercept

    def apply_logistic_model(self, lm_beta, lm_intercept):
        assert lm_beta.row == lm_intercept.row
        assert lm_intercept.column == range(1)

        if self.column == lm_beta.column:
            source, lm_beta = self, lm_beta
        else:
            source, lm_beta = self.align(lm_beta, [(1, 1, False)])
        source_ext_matrix = np.hstack([np.ones((len(source.row), 1)),
                                       source.to_dense().matrix])
        coef_matrix = np.hstack([lm_intercept.matrix, lm_beta.to_dense().matrix])
        guessed_matrix = logit_inplace(source_ext_matrix.dot(coef_matrix.transpose()))
        return LabeledMatrix((self.row, lm_beta.row), guessed_matrix,
                             (self.row_deco, lm_beta.row_deco))

    def update_clusters(self, cooccurrence, specificity=0.7, eps=10.):
        """
        self is supposed to be cluster-like LM,
            i.e. row are the entries, columns are clusters,
            and each row of self.matrix is a categorical vector,
            matrix[i, j] = 1 iff row[i] in cluster[j], else 0.

        `cooccurrence` is entries-entries like matrix.

        Return: cluster-like LM that add to self rows that were not yet present in self
        but appear in `cooccurrence.row`

        Example: :
            >>> from karma.core.labeledmatrix.utils import lm_from_dict
            >>> group = lm_from_dict({'matelas': 'matelas',  'drap1': "drap", 'drap2': "drap",
            ...                       'fifa': "game", 'gtaV': "game", 'gtaIV': "game"})
            >>> row = ['fifa13', 'drap3']
            >>> column = ['fifa', 'gtaV', 'drap1', 'matelas']
            >>> mat = np.array([[4, 3, 1, 1],
            ...                 [1, 1, 4, 2]])
            >>> cooccurrence = LabeledMatrix((row, column), mat)
            >>> update = group.update_clusters(cooccurrence)
            >>> update.reco('fifa', reco_only=True)  #doctest: +NORMALIZE_WHITESPACE
            ------------
            reco | score
            ------------
            game   1.0
            >>> update.reco('drap3', reco_only=True)  #doctest: +NORMALIZE_WHITESPACE
            ------------
            reco | score
            ------------
            drap   1.0
            >>> update.row.difference(group.row)[0]
            ['fifa13', 'drap3']
            >>> update.column == group.column
            True
        """
        if len(cooccurrence.row.difference(self.row)[0]) == 0:
            return self

        cogroup = cooccurrence.without_diagonal().dot(self).without_zeros()
        return self + cogroup.exclude_row(self.row).dot(
            (cogroup.sum(axis=0) + eps).power(-specificity)).truncate(nb_h=1).nonzero_mask()

    def pairwise_overlap(self, top, axis=0, renorm=True, potential=False):
        """
        * top can by either int (number) either float in [0, 1]
        * axis in {0, 1}, default is 0
        * renorm = True/False, default is True

        Warning : That takes into account only nonzero scores
            >>> matrix = np.array([[10, 1, 2], [2, 5, 3], [5, 6, 6], [1, 3, 5]])
            >>> lm = LabeledMatrix((range(4), ['a', 'b', 'c']), matrix)
            >>> np.array(lm.pairwise_overlap(2).matrix)
            array([[1. , 0.5, 0.5],
                   [0.5, 1. , 0.5],
                   [0.5, 0.5, 1. ]])
            >>> np.array(lm.pairwise_overlap(3).matrix)
            array([[1.        , 0.66666667, 0.66666667],
                   [0.66666667, 1.        , 1.        ],
                   [0.66666667, 1.        , 1.        ]])
            >>> np.array(lm.pairwise_overlap(0.8, renorm=False).matrix, dtype=np.int)
            array([[3, 2, 2],
                   [2, 3, 3],
                   [2, 3, 3]])
            >>> np.array(lm.pairwise_overlap(0.8, axis=1, renorm=False).matrix, dtype=np.int)
            array([[2, 1, 1, 1],
                   [1, 2, 2, 2],
                   [1, 2, 2, 2],
                   [1, 2, 2, 2]])
            >>> lm.pairwise_overlap(0.8, axis=1, renorm=False).is_square()
            True
            >>> lm.pairwise_overlap(0.8, axis=1, renorm=False).row
            [0, 1, 2, 3]
        """
        if axis != 0:
            return self.transpose().pairwise_overlap(top, axis=co(axis), renorm=renorm, potential=potential)

        if 0 <= top < 1:
            top = int(top * len(self.row))

        top_lm = self.truncate(nb_v=top)
        top_lm_mask = top_lm.nonzero_mask()
        if potential:
            overlap_lm = top_lm.transpose()._dot(top_lm_mask)
        else:
            overlap_lm = top_lm_mask.transpose()._dot(top_lm_mask)
        if renorm:
            return overlap_lm.normalize(axis=1, norm='linf')
        else:
            return overlap_lm

    def _rank_dispatch(self, maximum_pressure, max_ranks, max_volumes):
        """
        WARNING: works only on nonzero scores
        Returns LabeledMatrix with allocation user-topic
        Args:
            maximum_pressure: maximal number of times each user can appear in allocation
            max_ranks: arraylike: maximal rank of user in extract each topic can take
            max_volumes: arraylike: maximal value of population volume each topic can be allocated to
        >>> matrix = np.array([[10, 1, 3], [2, 5, 3], [5, 6, 6], [1, 3, 5]])
        >>> lm = LabeledMatrix((range(4), ['a', 'b', 'c']), matrix)
        >>> lm._rank_dispatch(1, [4, 4, 4], [4, 4, 4]).to_flat_dataframe('user_id', 'topic_id', 'score')\
              .preview() #doctest: +NORMALIZE_WHITESPACE
        --------------------------
        user_id | topic_id | score
        --------------------------
        0         a          10.0
        1         b          5.0
        2         b          6.0
        3         c          5.0

        >>> lm._rank_dispatch(2, [1, 4, 2], [1, 2, 2]).to_flat_dataframe('user_id', 'topic_id', 'score')\
              .preview() #doctest: +NORMALIZE_WHITESPACE
        --------------------------
        user_id | topic_id | score
        --------------------------
        0         a          10.0
        1         b          5.0
        2         b          6.0
        2         c          6.0
        3         c          5.0
        """
        choice = rank_dispatch(self.matrix, maximum_pressure, np.asarray(max_ranks), np.asarray(max_volumes))
        return LabeledMatrix(self.label, choice, self.deco)

    def rank_dispatch(self, maximum_pressure, max_ranks=None, max_volumes=None):
        """
        Return LabeledMatrix with allocation user-topic.

        WARNING: works only on nonzero scores
        Args:
            maximum_pressure: maximal number of times each user can appear in allocation
            max_ranks: int/dict: maximal rank of user in extract each topic can take;
                                           default: maximal rank
            max_volumes: int/dict: maximal value of population volume each topic can be allocated to;
                                             default: all users
        >>> matrix = np.array([[10, 1, 3], [2, 5, 3], [5, 6, 6], [1, 3, 5]])
        >>> lm = LabeledMatrix((range(4), ['a', 'b', 'c']), matrix)
        >>> lm.rank_dispatch(1).to_flat_dataframe('user_id', 'topic_id', 'score')\
              .preview() #doctest: +NORMALIZE_WHITESPACE
        --------------------------
        user_id | topic_id | score
        --------------------------
        0         a          10.0
        1         b          5.0
        2         b          6.0
        3         c          5.0
        >>> lm.rank_dispatch(1, 1, 1).to_flat_dataframe('user_id', 'topic_id', 'score')\
              .preview() #doctest: +NORMALIZE_WHITESPACE
        --------------------------
        user_id | topic_id | score
        --------------------------
        0         a          10.0
        2         b          6.0
        >>> lm.rank_dispatch(2, {'a': 2, 'b': 1, 'c': 3}, 2).to_flat_dataframe('user_id', 'topic_id', 'score')\
              .preview() #doctest: +NORMALIZE_WHITESPACE
        --------------------------
        user_id | topic_id | score
        --------------------------
        0         a          10.0
        2         b          6.0
        2         c          6.0
        3         c          5.0
        """
        nb_topic = len(self.column)
        if max_volumes is None:
            max_volumes = self.matrix.shape[0]
        if is_integer(max_volumes):
            max_volumes = np.full(nb_topic, max_volumes, dtype=np.int)
        elif isinstance(max_volumes, dict):
            max_volumes = np.array([max_volumes.get(topic, 0) for topic in self.column])
        else:
            raise ValueError('max_volumes must be integer or dict')

        if max_ranks is None:
            max_ranks = self.matrix.shape[0]
        if is_integer(max_ranks):
            max_ranks = np.full(nb_topic, max_ranks, dtype=np.int)
        elif isinstance(max_ranks, dict):
            max_ranks = np.array([max_ranks[topic] for topic in self.column])
        else:
            raise ValueError('max_ranks must be integer or dict')

        return self._rank_dispatch(maximum_pressure, max_ranks, max_volumes)

    def argmax_dispatch(self, maximum_pressure, max_ranks=None, max_volumes=None):
        """
        Return LabeledMatrix with allocation user-topic based on score.

        Args:
            maximum_pressure: maximal number of times each user can appear in allocation
            max_ranks: int/dict: maximal rank of user in extract each topic can take;
                                           default: maximal rank
            max_volumes: int/dict: maximal value of population volume each topic can be allocated to;
                                             default: all users
        """
        matrix = self.matrix
        nb_topic = len(self.column)
        if max_volumes is None:
            max_volumes = matrix.shape[0]
        if is_integer(max_volumes):
            max_volumes = np.full(nb_topic, max_volumes, dtype=np.int)
        elif isinstance(max_volumes, dict):
            max_volumes = np.array([max_volumes.get(topic, 0) for topic in self.column])
        else:
            raise ValueError('max_volumes must be integer or dict')

        if max_ranks is None:
            max_ranks = matrix.shape[0]
        if is_integer(max_ranks):
            max_ranks = np.full(nb_topic, max_ranks, dtype=np.int)
        elif isinstance(max_ranks, dict):
            max_ranks = np.array([max_ranks[topic] for topic in self.column])
        else:
            raise ValueError('max_ranks must be integer or dict')

        choice = argmax_dispatch(matrix, maximum_pressure, max_ranks, max_volumes)
        return LabeledMatrix(self.label, choice, self.deco)

    def population_allocation(self, dispatch_mask, norm='l1'):
        """
        Returns population allocation matrix for a given score matrix (self), and dispatch matrix from dispatch_mask
        both self and dispatch_mask should be LabeledMatrices with label (users, topics) and we use standard
        LM's alignment by intersection on both users and topics
        Args:
            dispatch_mask: LabeledMatrix with dispatch
            norm: string or None: norm to use to normalize result's rows, if None raw result will be returned
        >>> score = LabeledMatrix((['u1', 'u2', 'u3'], ['t1', 't2']), np.array([[1, 0.2], [0.5, 0.5], [0.2, 1]]))
        >>> dispatch_mask = LabeledMatrix((['u1', 'u2', 'u3'], ['t1', 't2']), np.array([[1, 0], [0.5, 0], [0., 1]]))
        >>> score.population_allocation(dispatch_mask, norm=None).matrix
        array([[2., 1.],
               [2., 1.]])
        >>> score.population_allocation(dispatch_mask, norm='l1').matrix
        array([[0.66666667, 0.33333333],
               [0.66666667, 0.33333333]])
        """
        pop_allocation = self.nonzero_mask().transpose().dot(dispatch_mask.nonzero_mask())
        if norm is not None:
            pop_allocation = pop_allocation.normalize(norm=norm)

        return pop_allocation

    def potential_allocation(self, dispatch_mask, norm='l1'):
        """
        Returns potential allocation matrix for a given score matrix (self), and dispatch matrix from dispatch_mask
        both self and dispatch_mask should be LabeledMatrices with label (users, topics) and we use standard
        LM's alignment by intersection on both users and topics
        Args:
            dispatch_mask: LabeledMatrix with dispatch
            norm: string or None: norm to use to normalize result's rows, if None raw result will be returned
        >>> score = LabeledMatrix((['u1', 'u2', 'u3'], ['t1', 't2']), np.array([[1, 0.2], [0.5, 0.5], [0.2, 1]]))
        >>> dispatch_mask = LabeledMatrix((['u1', 'u2', 'u3'], ['t1', 't2']), np.array([[1, 0], [0.5, 0], [0., 1]]))
        >>> score.potential_allocation(dispatch_mask, norm=None).matrix
        array([[1.5, 0.2],
               [0.7, 1. ]])
        >>> score.potential_allocation(dispatch_mask, norm='l1').matrix
        array([[0.88235294, 0.11764706],
               [0.41176471, 0.58823529]])
        """
        pot_allocation = self.transpose().dot(dispatch_mask.nonzero_mask())
        if norm is not None:
            pot_allocation = pot_allocation.normalize(norm=norm)

        return pot_allocation

    def heatmap(self, row_name='row', col_name='col', ordering=None, **kwargs):
        """

        :param col_name:
        :param row_name:
        :param ordering: None, hierarchical or naive (sort both row and col)
        :param kwargs:
        :return:

        >>> from karma.core.karmaplot import KarmaPlot
        >>> lm = LabeledMatrix((['a', 'b', 'c', 'd', 'e'], [42]), KarmaSparse(np.arange(5).reshape((5, 1))))
        >>> kplot = lm.heatmap(col_name='toto', renorm='%', backend=None)
        >>> isinstance(kplot, KarmaPlot)
        True
        >>> kplot.data['matrix']
        array([[ 0.],
               [10.],
               [20.],
               [30.],
               [40.]])
        >>> kplot.data['vmax']
        100.0
        """
        if ordering == 'hierarchical':
            lm = self.sort_by_hierarchical_clustering()
        elif ordering == 'naive':
            lm = self.sort()
        else:
            lm = self
        kwargs.pop('column', None)
        return lm.to_dense().to_vectorial_dataframe(row_name, col_name).heatmap(**kwargs)

    def scalar_multiply_rows(self, rows, scalar):
        """
        Multiplies given rows by a scalar
        Args:
            rows: rows to multiply; takes intersection of self.row and given list
            scalar: scalar to multiply on
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1, 0], [3, 4]]))
        >>> lm.scalar_multiply_rows(['c', 'f'], 2).matrix
        array([[1, 0],
               [6, 8]])
        >>> lm.scalar_multiply_rows(['b', 'c'], 0.5).matrix
        array([[0.5, 0. ],
               [1.5, 2. ]])
        >>> lm.scalar_multiply_rows(['a', 'd'], 5).matrix
        array([[1, 0],
               [3, 4]])
        """
        try:
            diff_lm = self.restrict_row(rows)
        except LabeledMatrixException:
            return self

        diff_lm = (1 - scalar) * diff_lm
        diff_lm = diff_lm.align(self, [(0, 0, None), (1, 1, None)], self_only=True)

        return self - diff_lm

    def scalar_multiply_columns(self, columns, scalar):
        """
        Multiplies given columns by a scalar
        Args:
            columns: columns to multiply; takes intersection of self.column and given list
            scalar: scalar to multiply on
        >>> lm = LabeledMatrix((['b', 'c'], ['x', 'y']), np.array([[1, 0], [3, 4]]))
        >>> lm.scalar_multiply_columns(['y', 'z'], 2).matrix
        array([[1, 0],
               [3, 8]])
        >>> lm.scalar_multiply_columns(['x', 'y'], 0.5).matrix
        array([[0.5, 0. ],
               [1.5, 2. ]])
        >>> lm.scalar_multiply_columns(['u', 'z'], 5).matrix
        array([[1, 0],
               [3, 4]])
        """
        return self.transpose().scalar_multiply_rows(columns, scalar).transpose()

    def rename_row(self, prefix='', suffix='', mapping=None):
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
        rows = ['{}{}{}'.format(prefix, r, suffix) for r in apply_python_dict(mapping, self.row, None, keep_same=True)]
        row_deco = keymap(dict(izip(self.row, rows)).get, self.row_deco)
        return LabeledMatrix((rows, self.column), self.matrix, (row_deco, self.column_deco))

    def rename_column(self, prefix='', suffix='', mapping=None):
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

    def glove(self, rank, epochs=30, alpha=0.75, max_count=None, learning_rate=0.05, 
              no_threads=KarmaSetup.open_mp_nb_thread, seed=None):
        """
        Warning : works only on upper triangle matrix
        For params, see https://github.com/maciejkula/glove-python
                        https://nlp.stanford.edu/projects/glove/
        Result is stable only if seed is fixed and no_threads=1
        
        Test on random matrix to show usage:

        >>> with use_seed(122): a = np.random.rand(100, 5)
        >>> lm = LabeledMatrix((range(a.shape[0]), range(a.shape[0])), a.dot(a.T))
        >>> lm_glove = lm.glove(5, no_threads=1)
        >>> lm_glove.shape == a.shape
        True
        """
        assert self.is_square()

        matrix = self.to_sparse().matrix
        if max_count is None:
            max_count = np.percentile(matrix.diagonal(), 80)

        scipy_matrix = matrix.triu().to_scipy_sparse(copy=False).tocoo()

        model = Glove(no_components=rank, alpha=alpha, max_count=max_count,
                      max_loss=10.0, learning_rate=learning_rate, random_state=seed)
        model.fit(scipy_matrix, epochs=epochs, no_threads=no_threads, verbose=KarmaSetup.verbose)
        return LabeledMatrix((self.row, range(rank)), model.word_vectors, deco=(self.row_deco, {}))

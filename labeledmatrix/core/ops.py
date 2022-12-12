from cyperf.matrix.karma_sparse import ks_hstack
from labeledmatrix.core.labeledmatrix import LabeledMatrix
from labeledmatrix.learning.matrix_utils import align_along_axis, safe_add, safe_multiply


def lm_hstack(list_of_lm, dense_output=False):
    """
    >>> from labeledmatrix.core.labeledmatrix import LabeledMatrix
    >>> from labeledmatrix.core.utils import aeq
    >>> import numpy as np
    >>> lm1 = LabeledMatrix((['b', 'a'], ['x']), np.array([[4], [7]])).sort()
    >>> lm2 = LabeledMatrix((['c'], ['x', 'z', 'y']), np.array([[7, 9, 8]])).sort()
    >>> lm3 = LabeledMatrix((['a', 'd'], ['z', 'w', 'x']), np.array([[1, 5, 20], [-1, 4, -20]])).sort()
    >>> lm = lm_hstack([lm1, lm2, lm3])
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


def reduce_sum(list_of_lm):
    """
    >>> from labeledmatrix.core.labeledmatrix import LabeledMatrix
    >>> import numpy as np
    >>> lm1 = LabeledMatrix((['b', 'a'], ['x']), np.array([[4], [7]]))
    >>> lm2 = LabeledMatrix((['c'], ['x', 'z', 'y']), np.array([[7, 9, 8]])).to_sparse()
    >>> lm3 = LabeledMatrix((['a', 'd'], ['z', 'w', 'x']), np.array([[1, 5., 20], [-1, 4, -20]]))
    >>> res = reduce_sum([lm1, lm2, lm3]).to_dense().sort()
    >>> res.label
    (['a', 'b', 'c', 'd'], ['w', 'x', 'y', 'z'])
    >>> res.matrix
    array([[  5.,  27.,   0.,   1.],
           [  0.,   4.,   0.,   0.],
           [  0.,   7.,   8.,   9.],
           [  4., -20.,   0.,  -1.]], dtype=float32)
    """
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
    from labeledmatrix.core.labeledmatrix import LabeledMatrix
    return LabeledMatrix((total_rows, total_columns),
                         result, deco=(row_deco, column_deco))


def reduce_product(list_of_lm):
    """
    >>> from labeledmatrix.core.labeledmatrix import LabeledMatrix
    >>> from labeledmatrix.core.utils import aeq
    >>> import numpy as np
    >>> lm1 = LabeledMatrix((['b', 'a'], ['x']), np.array([[4], [7]])).to_sparse()
    >>> lm2 = LabeledMatrix((['a'], ['x', 'z', 'y']), np.array([[7, 9, 8]]))
    >>> lm3 = LabeledMatrix((['a', 'd'], ['z', 'w', 'x']), np.array([[1, 5, 20], [-1, 4, -20]]))
    >>> res = reduce_product([lm1, lm2, lm3]).to_dense()
    >>> res.label
    (['a'], ['x'])
    >>> aeq(res.matrix, np.array([[980]]))
    True
    """
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

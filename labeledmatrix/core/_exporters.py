import pandas as pd

from cyperf.tools import apply_python_dict, argsort, take_indices


def to_vectorial_dataframe(lm, deco: bool = False) -> pd.DataFrame:
    """
    TODO doc
    :param lm:
    :param deco:
    :return:
    """
    if deco:
        columns = apply_python_dict(lm.column_deco, lm.column.list, None, True)
        rows = apply_python_dict(lm.row_deco, lm.row.list, None, True)
    else:
        rows = lm.row.list
        columns = lm.column.list
    if lm.is_sparse:
        return pd.DataFrame.sparse.from_spmatrix(lm.matrix.to_scipy_sparse(), columns=columns, index=rows)
    return pd.DataFrame(lm.matrix, columns=columns, index=rows)


def to_flat_dataframe(lm, row="col0", col="col1", dist="similarity", **kwargs) -> pd.DataFrame:
    """
    Return a DataFrame with three columns (row, col and dist) from LabeledMatrix.

    kwargs:
        - deco_row: name of the decoration column for row
        - deco_col: name of the decoration column for col
    """
    row_indices, col_indices = lm.matrix.nonzero()
    if not lm.is_sparse:
        values = lm.matrix[row_indices, col_indices]
    else:
        values = lm.matrix.data.copy()  # for sparse this is fast since order is the same

    data = {row: take_indices(lm.row, row_indices),
            col: take_indices(lm.column, col_indices),
            dist: values}
    if 'deco_row' in kwargs:
        data[kwargs['deco_row']] = apply_python_dict(lm.row_deco, data[row], '', False)
    if 'deco_col' in kwargs:
        data[kwargs['deco_col']] = apply_python_dict(lm.column_deco, data[col], '', False)
    return pd.DataFrame(data)


# FIXME add structs with label:value instead of list
# FIXME can we reuse self.to_ragged_tensor or self.to_pyarrow here if removing argsort ?
def to_list_dataframe(lm, col: str = "col", prefix: str = "list_of_", exclude: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame with columns col, list_of_col.
    For each row, sort the non zero values and return the column labels as list_of_col
    """
    if lm.is_sparse:
        matrix = lm.matrix.tocsr()
        double = [[x, [lm.column[y]
                       for y in matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]
                       [argsort(matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]])[::-1]]
                       if not exclude or x != lm.column[y]]]
                  for i, x in enumerate(lm.row)]
    else:
        asort = lm.matrix.argsort(axis=1)[:, ::-1]
        double = [[x, [lm.column[y] for y in asort[i] if
                       lm.matrix[i, y] and (not exclude or x != lm.column[y])]]
                  for i, x in enumerate(lm.row)]
    return pd.DataFrame([x for x in double if x[1]],
                        columns=[col, prefix + col]).sort_values(col)

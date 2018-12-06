from collections import defaultdict

import numpy as np

from karma.core.instructions import INSTRUCTIONS
from karma.core.dataframe import DataFrame
from karma.learning.matrix_utils import coherence, gram_quantiles, safe_max, safe_min
from karma.learning.utils import create_basic_virtual_hstack


def gram_first_quartile(mat):
    return gram_quantiles(mat, q=0.25)[0]


def gram_third_quartile(mat):
    return gram_quantiles(mat, q=0.75)[0]


def is_positive(matrix):
    """
    >>> import numpy as np
    >>> from cyperf.matrix.karma_sparse import KarmaSparse
    >>> is_positive(np.eye(5))
    True
    >>> is_positive(-np.eye(5))
    False
    >>> is_positive(KarmaSparse(np.eye(5, 5)))
    True
    """
    return matrix.min() >= 0


def descriptive_features_benchmark(df, features=None, func_list=None, column_order=None, restrict_nonempty=False):
    """Returns descriptive statistics on features of a dataframe.

    Parameters
    ----------
    df : Karma dataframe
        Dataframe of interest.
    features : list of strings, default to all vectorial and sparse columns
        List of features on which to restrict descriptive statistics computations.
    func_list : list of tuple of two strings or functions
        If given a tuple ('f', 'g', 'n') then we compute g row-wise, use f as an aggregator, and name the resulting
        statistics n, if n is not specified we create the name f_g.
        If given a function, it is supposed to apply on the whole matrix associated to the feature.
    column_order : list of strings, default None
        Allows to force column order of the output
    restrict_nonempty : bool, default False
        Should the computations be restricted to non-empty rows of dataframe
    >>> from karma.core.dataframe import DataFrame
    >>> import numpy as np
    >>> from karma.core.utils import use_seed
    >>> with use_seed(12784):
    ...     df = DataFrame({"col_1": np.ones((5, 2)), "col_2": np.random.randn(5, 2)})
    >>> f_list = [('mean', 'l2_norm', 'm_l2'), is_positive, ('mean', 'l0_norm')]
    >>> descriptive_features_benchmark(df, func_list=f_list).preview() #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    ------------------------------------------------------------------------------------------
    feature | is_sparse | dimension | nonempty_rows_rate | m_l2   | is_positive | mean_l0_norm
    ------------------------------------------------------------------------------------------
    col_2     False       2           1.0                  1.4748   False         2.0
    col_1     False       2           1.0                  1.4142   True          2.0
    >>> with use_seed(42):
    ...    df = DataFrame({"col_1": np.ones((5, 2)), "col_2": np.random.randn(5, 2), "col_3": np.random.randn(5)})
    >>> f_list = [('mean', 'l2_norm', 'm_l2'), coherence, safe_min]
    >>> descriptive_features_benchmark(df, features=df.column_names, func_list=f_list).preview() #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    ------------------------------------------------------------------------------------
    feature | is_sparse | dimension | nonempty_rows_rate | m_l2   | coherence | safe_min
    ------------------------------------------------------------------------------------
    col_2     False       2           1.0                  0.995    0.5752      -0.4695
    col_3     False       1           1.0                  0.9619   0.0         -1.9133
    col_1     False       2           1.0                  1.4142   1.0         1.0
    """
    if features is None:
        features = df.vectorial_column_names + df.sparse_column_names

    result_lists = defaultdict(list)

    result_lists['feature'] = features
    result_lists['is_sparse'] = [df[feature].is_sparse() for feature in features]
    result_lists['dimension'] = [df[feature].safe_dim() for feature in features]

    default_column_order = ['feature', 'is_sparse', 'dimension', 'nonempty_rows_rate']

    if func_list is None:
        func_list = [('min', 'is_one_hot', 'is_one_hot'),
                     is_positive,
                     ('mean', 'l2_norm'),
                     ('standard_deviation', 'l2_norm'),
                     ('mean', 'l0_norm'),
                     gram_first_quartile,
                     gram_third_quartile,
                     coherence,
                     safe_min,
                     safe_max]
        # TODO gram_first_quartile, gram_third_quartile, coherence will materizalize VDP !

    for feature in features:
        data = create_basic_virtual_hstack(df, [feature]).X[0]

        idx_non_empty = np.where(INSTRUCTIONS['is_non_empty']('input', 'output').bulk_call((data,)))[0]
        result_lists['nonempty_rows_rate'].append(float(len(idx_non_empty)) / len(df))

        if restrict_nonempty:
            data = data[idx_non_empty]

        for func in func_list:
            if isinstance(func, tuple):
                if len(func) == 2:
                    func += ('_'.join(func),)
                aggregator, instruction_name, col_name = func

                try:
                    instruction = INSTRUCTIONS[instruction_name]('input', 'output')
                    transformed_data = instruction.bulk_call((data,))
                    agg_val = DataFrame({'values': transformed_data}).aggregate('{}(values)'.format(aggregator))
                except TypeError:
                    agg_val = None
                result_lists[col_name].append(agg_val)
                default_column_order.append(col_name)
            else:
                col_name = func.__name__
                try:
                    func_val = func(data)
                except TypeError:
                    func_val = None
                result_lists[col_name].append(func_val)
                default_column_order.append(col_name)

    if column_order is None:
        return DataFrame(dict(result_lists)).copy(*default_column_order)
    else:
        return DataFrame(dict(result_lists)).copy(*column_order)

from collections import defaultdict
from karma.core.dataframe import DataFrame
from karma.learning.matrix_utils import coherence, gram_quantiles, safe_max, safe_min


def gram_first_quartile(mat):
    return gram_quantiles(mat, q=0.25)

def gram_third_quartile(mat):
    return gram_quantiles(mat, q=0.75)


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
    ----------------------------------------------------------------------------------------------------
    features | is_sparse | dimension | proportion_of_nonempty_rows | m_l2   | is_positive | mean_l0_norm
    ----------------------------------------------------------------------------------------------------
    col_2      False       2           1                             1.4748   False         2.0
    col_1      False       2           1                             1.4142   True          2.0
    """
    if features is None:
        features = df.vectorial_column_names + df.sparse_column_names

    result_lists = defaultdict(list)

    result_lists["features"] = features
    result_lists["is_sparse"] = [df[feature].is_sparse() for feature in features]
    result_lists["dimension"] = [df[feature].vectorial_shape()[0] for feature in features]
    result_lists["proportion_of_nonempty_rows"] = [df.aggregate('sum(is_non_empty({}))'.format(feature)) / len(df)
                                                for feature in features]

    if column_order is None:
        default_column_order = ["features", "is_sparse", "dimension", "proportion_of_nonempty_rows"]

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

    for feature in features:

        if restrict_nonempty:
            df_copy = df.copy()
            nonempty_col_name = 'is_nonempty_' + feature
            df_copy[nonempty_col_name] = df_copy['is_non_empty({})'.format(feature)]
            df_copy = df_copy.where(nonempty_col_name)
        else:
            df_copy = df.copy()

        for func in func_list:
            if isinstance(func, tuple):
                if len(func) == 3:
                    col_name = func[2]
                else:
                    col_name = func[0] + '_' + func[1]
                aggregate_func = func[0] + '(' + func[1] + '({}))'
                result_lists[col_name].append(df_copy.aggregate(aggregate_func.format(feature)))
                default_column_order.append(col_name)
            else:
                col_name = func.__name__
                result_lists[col_name].append(func(df_copy[feature][:]))
                default_column_order.append(col_name)

    if column_order is None:
        return DataFrame(dict(result_lists)).copy(*default_column_order)
    else:
        return DataFrame(dict(result_lists)).copy(*column_order)

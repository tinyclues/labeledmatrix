from collections import OrderedDict
from copy import deepcopy
from functools import partial
from itertools import chain
import numpy as np
from karma.core.dataframe import DataFrame
from sklearn.utils.extmath import cartesian
from karma.macros import squash
from karma.core.column import Column
from karma.learning.utils import CrossValidationWrapper, BasicVirtualHStack
from karma.learning.bayesian_constants import (BAYES_PRIOR_COEF_VAR_NAME, BAYES_POST_INTERCEPT_MEAN_NAME,
                                               BAYES_POST_COEF_MEAN_NAME)
from karma.core.utils.utils import Parallel
from karma.core.utils.utils import coerce_to_tuple_and_check_all_strings
from multiprocessing.pool import ThreadPool

CV_KEYS = ['cv', 'cv_n_splits', 'seed', 'cv_groups']
PRED_COL_NAME = 'predictions'
PENALTY_COL_NAME = 'penalty'
PRINT_PENALTY_COL_NAME = 'condensed_penalty'


class CVSampler(object):
    """Interface around CrossValidationWrapper.

    Easily samples the test error of a lib_symbol given a value of the free parameter. The lib_symbol needs to
    implement cross-validation.

    Parameters
    ----------
    dataframe : dataframe
    features : basestring or array-like of basestrings
    lib_parameters : dict, both training and cv parameters
    penalty_parameter_name : basestring default BAYES_PRIOR_COEF_VAR_NAME
    lib_symbol : default 'logistic_regression'
    metrics : basestring or array-like of basestrings, agregation symbols, default 'auc'

    >>> from karma.core.utils import use_seed
    >>> with use_seed(42):
    ...     random_df = DataFrame({'aa': np.random.randn(20, 9),
    ...                            'b': np.random.randn(20, 3),
    ...                            'group_col': np.random.randint(3, size=20),
    ...                            'strat_col': np.random.randint(3, size=20),
    ...                            'y': np.random.randint(2, size=20)})
    >>> cvs = CVSampler(random_df, {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 2, 'verbose': False})
    >>> cvs.evaluate_and_summarize_cv(0.1, features=['aa', 'b']).copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
    ----------------------------------------------------------------
    fold_type | Count | CountPos | auc    | penalty_hr | features
    ----------------------------------------------------------------
    test        4       3          1.0      [0.1 0.1]    ['aa', 'b']
    test        4       3          -1.0     [0.1 0.1]    ['aa', 'b']
    train       20      13         0.8022   [0.1 0.1]    ['aa', 'b']
    >>> cvs = CVSampler(random_df, {'axis': 'y', 'cv': 0.2, 'verbose': False}, metrics=['auc', 'normalized_log_loss'])
    >>> cvs.evaluate_and_summarize_cv(0.1, features=['aa', 'b']).copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
    --------------------------------------------------------------------------------------
    fold_type | Count | CountPos | auc    | normalized_log_loss | penalty_hr | features
    --------------------------------------------------------------------------------------
    test        4       3          1.0      0.8127                [0.1 0.1]    ['aa', 'b']
    train       20      13         0.8022   0.7824                [0.1 0.1]    ['aa', 'b']
    >>> cvs.evaluate_and_summarize_cv(0.1, features=['aa', 'b'], metric_groups='group_col').copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
    -----------------------------------------------------------------------------------------------
    fold_type | group_col | Count | CountPos | auc | normalized_log_loss | penalty_hr | features
    -----------------------------------------------------------------------------------------------
    test        0           1       1          0.0   nan                   [0.1 0.1]    ['aa', 'b']
    test        1           2       1          1.0   0.9949                [0.1 0.1]    ['aa', 'b']
    test        2           1       1          0.0   nan                   [0.1 0.1]    ['aa', 'b']
    train       0           7       4          1.0   0.6888                [0.1 0.1]    ['aa', 'b']
    train       1           9       5          0.5   0.8749                [0.1 0.1]    ['aa', 'b']
    train       2           4       4          0.0   nan                   [0.1 0.1]    ['aa', 'b']
    >>> cvs.evaluate_and_summarize_cv(0.1, features='b', metric_groups=['group_col', 'strat_col']).copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
    ---------------------------------------------------------------------------------------------------------
    fold_type | group_col | strat_col | Count | CountPos | auc  | normalized_log_loss | penalty_hr | features
    ---------------------------------------------------------------------------------------------------------
    test        0           2           1       1          0.0    nan                   [0.1]        b
    test        1           0           1       1          0.0    nan                   [0.1]        b
    test        1           2           1       0          -1.0   nan                   [0.1]        b
    test        2           0           1       1          0.0    nan                   [0.1]        b
    train       0           0           1       1          0.0    nan                   [0.1]        b
    train       0           1           2       0          -1.0   nan                   [0.1]        b
    train       0           2           4       3          1.0    0.8143                [0.1]        b
    train       1           0           6       4          1.0    0.8415                [0.1]        b
    train       1           1           1       0          -1.0   nan                   [0.1]        b
    train       1           2           2       1          -1.0   1.1288                [0.1]        b
    train       2           0           3       3          0.0    nan                   [0.1]        b
    train       2           1           1       1          0.0    nan                   [0.1]        b
    >>> cvs = CVSampler(random_df, {'axis': 'y', 'cv': 0.2, 'verbose': False}, penalty_parameter_name='pre_lasso_penalty',
    ...                 lib_symbol='bayesian_logistic_regression')
    >>> cvs.evaluate_and_summarize_cv(0.1, features=['aa', 'b']).copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
    -------------------------------------------------------------
    fold_type | Count | CountPos | auc | penalty_hr | features
    -------------------------------------------------------------
    test        4       3          1.0   0.1          ['aa', 'b']
    train       20      13         1.0   0.1          ['aa', 'b']
    """
    def __init__(self, dataframe, lib_parameters, penalty_parameter_name=BAYES_PRIOR_COEF_VAR_NAME,
                 lib_symbol='logistic_regression', metrics='auc'):
        self.dataframe = dataframe

        self.lib_symbol = lib_symbol
        self.penalty_parameter_name = penalty_parameter_name

        self.training_params = deepcopy(lib_parameters)
        for cv_key in CV_KEYS:
            self.training_params.pop(cv_key, None)

        self.cv_wrapper = CrossValidationWrapper.create_cv_from_data_and_params(dataframe, **lib_parameters)
        self.training_params['cv'] = self.cv_wrapper

        self.metrics = coerce_to_tuple_and_check_all_strings(metrics)

        self.kc = None
        self.meta = None

    def evaluate_cv(self, penalty_value, features=None, pred_col_name=PRED_COL_NAME):
        df_copy = self.dataframe.copy()
        if features is None:
            features = df_copy.vectorial_column_names + df_copy.sparse_column_names

        self.training_params[self.penalty_parameter_name] = penalty_value

        self.kc = df_copy.build_karmacode(method=self.lib_symbol,
                                          inputs=features,
                                          outputs=pred_col_name,
                                          parameters=self.training_params)

        self.meta = self.kc[-1].meta
        df_copy += self.kc
        return df_copy

    def evaluate_and_summarize_cv(self, penalty_value, features=None, pred_col_name=PRED_COL_NAME, metric_groups=None):
        # add a check in case the lib does not generate a cv

        if features is None:
            features = self.dataframe.vectorial_column_names + self.dataframe.sparse_column_names

        df_cv_copy = self.evaluate_cv(penalty_value, features=features, pred_col_name=pred_col_name)

        train_metric_results = self._metric_aggregation(df_cv_copy, penalty_value, features, metric_groups)

        train_betas_and_intercept = np.hstack([np.hstack(self.meta[BAYES_POST_COEF_MEAN_NAME]),
                                               self.meta[BAYES_POST_INTERCEPT_MEAN_NAME]])
        full_betas = map(tuple, np.repeat(train_betas_and_intercept.reshape(1, -1), len(train_metric_results), axis=0))
        train_metric_results['full_betas'] = full_betas
        train_metric_results['features'] = [str(features) for _ in xrange(len(train_metric_results))]

        test_metric_results = DataFrame()
        test_betas_and_intercept = map(np.hstack, zip(map(np.hstack, self.meta['cv'].feat_coefs),
                                                      self.meta['cv'].intercepts))
        for i, (fold_test_indices, fold_test_y_hat) in enumerate(self.cv_wrapper.fold_indices_iter):
            predictions_df = self.dataframe[fold_test_indices]
            predictions_df[pred_col_name] = fold_test_y_hat

            test_fold_metric_results = self._metric_aggregation(predictions_df, penalty_value, features, metric_groups)
            full_betas = map(tuple, np.repeat(test_betas_and_intercept[i].reshape(1, -1), len(test_fold_metric_results), axis=0))
            test_fold_metric_results['full_betas'] = full_betas
            test_fold_metric_results['features'] = [str(features) for _ in xrange(len(test_fold_metric_results))]
            test_metric_results = squash(test_metric_results, test_fold_metric_results)

        final_metric_results = squash({'train': train_metric_results.copy(*test_metric_results.column_names),
                                       'test': test_metric_results}, label='fold_type') # if lazy=True it breaks

        fixed_columns = [PENALTY_COL_NAME, 'full_betas', 'fold_type']
        proper_col_order = fixed_columns + [c for c in final_metric_results.column_names if c not in fixed_columns]

        return final_metric_results.copy(*proper_col_order)

    def _metric_aggregation(self, df, penalty_value, features, metric_groups=None): # we could remove the penalty_value arg and use an attribute instead
        metric_agg_tuple = tuple('{0}({1}, {2}) as {0}'.format(metric, PRED_COL_NAME, self.training_params['axis'])
                                 for metric in self.metrics)
        agg_tuple = ('# as Count', 'sum({}) as CountPos'.format(self.training_params['axis']),) + metric_agg_tuple

        df_grouped = df.group_by(metric_groups, agg_tuple)

        if self.penalty_parameter_name in ['C', BAYES_PRIOR_COEF_VAR_NAME]:
            penalty_extended = self.extend_and_format_penalty(penalty_value, features)
            df_grouped = df_grouped.with_column(PENALTY_COL_NAME,
                                                'constant(constant={})'.format(tuple(penalty_extended)))
            partial_collapse = partial(self.collapse_penalty, features=features)
            df_grouped['penalty_hr'] = Column(map(np.array_str, map(partial_collapse, df_grouped[PENALTY_COL_NAME][:])))
        else:
            df_grouped = df_grouped.with_column(PENALTY_COL_NAME,
                                                'constant(constant={})'.format(penalty_value))
            df_grouped['penalty_hr'] = df_grouped[PENALTY_COL_NAME][:]

        return df_grouped

    def extend_and_format_penalty(self, penalty_value, features):
        """Utility function to provide common formatting for penalties.
        >>> df = DataFrame({"aa": np.random.randn(20, 9), "b": np.random.randn(20, 3), "y": np.random.randint(2, size=20)})
        >>> cv_sampler = CVSampler(df, {'axis': 'y', 'cv': 0.2})
        >>> cv_sampler.extend_and_format_penalty(0.1, ['aa', 'b'])
        array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        >>> cv_sampler.extend_and_format_penalty((0.1, 1), ['aa', 'b'])
        array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1. , 1. , 1. ])
        """
        if isinstance(features, basestring):
            df_bvhs = BasicVirtualHStack([self.dataframe[features].safe_dim()])
        else:
            df_bvhs = BasicVirtualHStack([self.dataframe[feat].safe_dim() for feat in features])
        return df_bvhs.adjust_array_to_total_dimension(penalty_value)

    def collapse_penalty(self, penalty_value, features):
        """Utility function to provide minimal description for penalties.
        >>> df = DataFrame({"aa": np.random.randn(20, 9), "b": np.random.randn(20, 3), "y": np.random.randint(2, size=20)})
        >>> cv_sampler = CVSampler(df, {'axis': 'y', 'cv': 0.2})
        >>> cv_sampler.collapse_penalty((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), ['aa', 'b'])
        array([0.1, 0.1])
        >>> cv_sampler.collapse_penalty((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1. , 1. , 1.), ['aa', 'b'])
        array([0.1, 1. ])
        >>> cv_sampler.collapse_penalty((0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1. , 2. , 1.), ['aa', 'b'])
        array([array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
               array([1., 2., 1.])], dtype=object)
        """
        if isinstance(features, basestring):
            df_bvhs = BasicVirtualHStack([self.dataframe[features].safe_dim()])
        else:
            df_bvhs = BasicVirtualHStack([self.dataframe[feat].safe_dim() for feat in features])
        block_penalty = df_bvhs.adjust_to_block_dimensions(penalty_value)
        out_penalty = list()
        for pen_array in block_penalty:
            if (pen_array == pen_array[0]).all():
                out_penalty.append(pen_array[0])
            else:
                out_penalty.append(pen_array)
        return np.array(out_penalty)


class GridGenerator(object):
    def __init__(self, initial_point):
        self.initial_point = initial_point

    def linear_grid(self, granularity=2, width=0.25):
        """Utility function to generate an hyper mesh.
        >>> GridGenerator(np.array([0.1, 1.])).linear_grid(width=0.5)
        array([[0.05, 0.5 ],
               [0.05, 1.5 ],
               [0.15, 0.5 ],
               [0.15, 1.5 ]])
        >>> GridGenerator(np.array([1., 1.])).linear_grid(granularity=3)
        array([[0.75, 0.75],
               [0.75, 1.  ],
               [0.75, 1.25],
               [1.  , 0.75],
               [1.  , 1.  ],
               [1.  , 1.25],
               [1.25, 0.75],
               [1.25, 1.  ],
               [1.25, 1.25]])
        """
        one_dim_grid = np.linspace(-1, 1, num=granularity)
        grid_correction = width * self.initial_point
        corrected_grid = cartesian([one_dim_grid for _ in xrange(len(self.initial_point))]).dot(np.diag(grid_correction))
        local_linear_grid = self.initial_point + corrected_grid
        local_linear_grid = np.clip(local_linear_grid, a_min=1e-6, a_max=None)  # bring grid back to positive orthant
        return local_linear_grid

    def scale_grid(self, downscale=-1, upscale=1):
        """Utility function to generate a logarithmic hyper mesh.
        >>> GridGenerator(np.array([0.1, 1.])).scale_grid(downscale=-1, upscale=0)
        array([[0.01, 0.1 ],
               [0.01, 1.  ],
               [0.1 , 0.1 ],
               [0.1 , 1.  ]])
        >>> GridGenerator(np.array([1., 1.])).scale_grid(-1, 1)
        array([[ 0.1,  0.1],
               [ 0.1,  1. ],
               [ 0.1, 10. ],
               [ 1. ,  0.1],
               [ 1. ,  1. ],
               [ 1. , 10. ],
               [10. ,  0.1],
               [10. ,  1. ],
               [10. , 10. ]])
        >>> GridGenerator(np.array([1.])).scale_grid(-1, 1)
        array([[ 0.1],
               [ 1. ],
               [10. ]])
        """
        one_dim_grid = np.logspace(downscale, upscale, num = np.abs(downscale) + np.abs(upscale) + 1)
        return self.initial_point * cartesian([one_dim_grid for _ in xrange(len(self.initial_point))])

    def gaussian_grid(self, n_samples=10, var_factor=1.):
        gaussian_grid = self.initial_point + var_factor * np.random.randn(n_samples, self.initial_point.shape[0])
        return np.clip(gaussian_grid, a_min=1e-6, a_max=None)


class GridEvaluator(object):
    def __init__(self):
        self.best_penalty = {}
        self.best_score = {}

    def best_mean_regul(self, df, metric='auc', order='max', robustness_threshold=1.):
        """Decide what is the best penalty already evaluated.
        >>> from karma.core.utils.utils import use_seed
        >>> with use_seed(42):
        ...     my_df = DataFrame({'a': np.random.randn(100, 5), 'b': np.random.randn(100, 7), 'y': np.random.binomial(1, 0.75, size=100)})
        >>> my_params = {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 1, 'verbose': False}
        >>> gs = GridSearch(my_df, my_params)
        >>> _ = gs.sequential_search([0.1, 1], ['a', 'b'], verbose=False)
        >>> evaluator = GridEvaluator()
        >>> evaluator.best_mean_regul(gs.evaluated_reguls)
        ({"['a', 'b']": (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)}, {"['a', 'b']": ('auc', -0.3867)})
        """
        test_penalty_df = self._compare_train_test(df, metric=metric)
        test_penalty_df_splitted = test_penalty_df.split_by('features')

        for feat in test_penalty_df_splitted:
            penalty_vec = test_penalty_df_splitted[feat][PENALTY_COL_NAME][:]
            metric_vec = test_penalty_df_splitted[feat]['{}'.format(metric)][:]
            delta_score = test_penalty_df_splitted[feat]['train_test_error_gap'][:]

            if order == 'min':
                metric_vec = -metric_vec

            metric_vec = np.where(np.abs(delta_score) <= robustness_threshold, metric_vec, -np.Inf)
            best_ind = np.argmax(metric_vec)
            self.best_penalty[feat] = penalty_vec[best_ind]
            self.best_score[feat] = (metric, metric_vec[best_ind])

        return self.best_penalty, self.best_score

    def _compare_train_test(self, df, metric):
        if df is None:
            raise ValueError("You need to generate a cross-validation before!")
        else:
            penalty_df = df.group_by(['fold_type', PENALTY_COL_NAME, 'features'],
                                      'mean({0}) as {0}'.format(metric))
        splitted_df = penalty_df.split_by('fold_type')
        train_score = splitted_df['train']['{}'.format(metric)][:]
        test_score = splitted_df['test']['{}'.format(metric)][:]
        delta_score = np.abs(np.array(train_score) - np.array(test_score))
        splitted_df['test']['train_test_error_gap'] = delta_score
        return splitted_df['test']


class GridSearch(CVSampler):
    """This class performs grid search, i.e. it is a CVSampler with a memory.

    Allows to sample the test error on a grid of penalty values.

    Parameters
    ----------
    dataframe : dataframe
    features : basestring or array-like of basestrings
    lib_parameters : dict, both training and cv parameters
    penalty_parameter_name : basestring default BAYES_PRIOR_COEF_VAR_NAME
    lib_symbol : default 'logistic_regression'
    metrics : basestring or array-like of basestrings, agregation symbols, default 'auc'
    metric_groups : basestring, default None, column to stratify metrics on

    >>> from karma.core.utils.utils import use_seed
    >>> with use_seed(42):
    ...     df = DataFrame({"aa": np.random.randn(100, 2), "b": np.random.randn(100, 1), "y": np.random.randint(2, size=100)})
    >>> gs = GridSearch(df, {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 2, 'verbose': False})
    >>> _ = gs.sequential_search([0.1, 1, 2], ['aa', 'b'], warm_start=False, verbose=False, n_jobs=2)
    >>> gs.evaluated_reguls.copy(exclude='full_betas').preview() # doctest: +NORMALIZE_WHITESPACE
    -----------------------------------------------------------------------------------
    penalty         | fold_type | Count | CountPos | auc     | penalty_hr | features
    -----------------------------------------------------------------------------------
    (0.1, 0.1, 0.1)   test        20      8          -0.0625   [0.1 0.1]    ['aa', 'b']
    (0.1, 0.1, 0.1)   test        20      8          -0.3333   [0.1 0.1]    ['aa', 'b']
    (0.1, 0.1, 0.1)   train       100     42         0.0837    [0.1 0.1]    ['aa', 'b']
    (1, 1, 1)         test        20      8          -0.0417   [1 1]        ['aa', 'b']
    (1, 1, 1)         test        20      8          -0.2708   [1 1]        ['aa', 'b']
    (1, 1, 1)         train       100     42         0.0878    [1 1]        ['aa', 'b']
    (2, 2, 2)         test        20      8          -0.0417   [2 2]        ['aa', 'b']
    (2, 2, 2)         test        20      8          -0.25     [2 2]        ['aa', 'b']
    (2, 2, 2)         train       100     42         0.0862    [2 2]        ['aa', 'b']
    >>> gs.summary().preview() # doctest: +NORMALIZE_WHITESPACE
    --------------------------------------------------------
    fold_type | penalty_hr | features    | mean_auc | sd_auc
    --------------------------------------------------------
    test        [0.1 0.1]    ['aa', 'b']   -0.1979    0.1354
    test        [1 1]        ['aa', 'b']   -0.1563    0.1145
    test        [2 2]        ['aa', 'b']   -0.1459    0.1041
    train       [0.1 0.1]    ['aa', 'b']   0.0837     0.0
    train       [1 1]        ['aa', 'b']   0.0878     0.0
    train       [2 2]        ['aa', 'b']   0.0862     0.0
    >>> gs = GridSearch(df, {'axis': 'y', 'cv': 0.2, 'verbose': False}, lib_symbol='logistic_regression')
    >>> _ = gs.sequential_search([0.01, 0.1], ['aa'], verbose=False, warm_start=True)
    >>> gs.evaluated_reguls.copy(exclude='full_betas').preview() # doctest: +NORMALIZE_WHITESPACE
    ----------------------------------------------------------------------------
    penalty      | fold_type | Count | CountPos | auc    | penalty_hr | features
    ----------------------------------------------------------------------------
    (0.01, 0.01)   test        20      8          0.0833   [0.01]       ['aa']
    (0.01, 0.01)   train       100     42         0.0681   [0.01]       ['aa']
    (0.1, 0.1)     test        20      8          0.0833   [0.1]        ['aa']
    (0.1, 0.1)     train       100     42         0.0755   [0.1]        ['aa']
    >>> _ = gs.sequential_search([0.01, 0.1], ['b'], verbose=False, warm_start=True)
    >>> gs.evaluated_reguls.copy(exclude='full_betas').preview() # doctest: +NORMALIZE_WHITESPACE
    -----------------------------------------------------------------------------
    penalty      | fold_type | Count | CountPos | auc     | penalty_hr | features
    -----------------------------------------------------------------------------
    (0.01, 0.01)   test        20      8          0.0833    [0.01]       ['aa']
    (0.01, 0.01)   train       100     42         0.0681    [0.01]       ['aa']
    (0.1, 0.1)     test        20      8          0.0833    [0.1]        ['aa']
    (0.1, 0.1)     train       100     42         0.0755    [0.1]        ['aa']
    (0.01,)        test        20      8          -0.1667   [0.01]       ['b']
    (0.01,)        train       100     42         0.0517    [0.01]       ['b']
    (0.1,)         test        20      8          -0.1667   [0.1]        ['b']
    (0.1,)         train       100     42         0.0517    [0.1]        ['b']
    >>> gs = GridSearch(df, {'axis': 'y', 'cv': 0.2, 'verbose': False}, lib_symbol='logistic_regression')
    >>> _ = gs.sequential_search([0.01, 0.1], ['aa'], verbose=False, warm_start=True, n_jobs=2)
    >>> _ = gs.sequential_search([0.01, 0.1], ['b'], verbose=False, warm_start=True)
    >>> gs.evaluated_reguls.copy(exclude='full_betas').preview() # doctest: +NORMALIZE_WHITESPACE
    -----------------------------------------------------------------------------
    penalty      | fold_type | Count | CountPos | auc     | penalty_hr | features
    -----------------------------------------------------------------------------
    (0.01, 0.01)   test        20      8          0.0833    [0.01]       ['aa']
    (0.01, 0.01)   train       100     42         0.0681    [0.01]       ['aa']
    (0.1, 0.1)     test        20      8          0.0833    [0.1]        ['aa']
    (0.1, 0.1)     train       100     42         0.0755    [0.1]        ['aa']
    (0.01,)        test        20      8          -0.1667   [0.01]       ['b']
    (0.01,)        train       100     42         0.0517    [0.01]       ['b']
    (0.1,)         test        20      8          -0.1667   [0.1]        ['b']
    (0.1,)         train       100     42         0.0517    [0.1]        ['b']
    >>> gs.summary().preview() # doctest: +NORMALIZE_WHITESPACE
    -----------------------------------------------------
    fold_type | penalty_hr | features | mean_auc | sd_auc
    -----------------------------------------------------
    test        [0.01]       ['aa']     0.0833     0.0
    test        [0.01]       ['b']      -0.1667    0.0
    test        [0.1]        ['aa']     0.0833     0.0
    test        [0.1]        ['b']      -0.1667    0.0
    train       [0.01]       ['aa']     0.0681     0.0
    train       [0.01]       ['b']      0.0517     0.0
    train       [0.1]        ['aa']     0.0755     0.0
    train       [0.1]        ['b']      0.0517     0.0
    """
    def __init__(self, dataframe, lib_parameters, penalty_parameter_name=BAYES_PRIOR_COEF_VAR_NAME,
                 lib_symbol='logistic_regression', metrics='auc', metric_groups=None):
        super(GridSearch, self).__init__(dataframe=dataframe, lib_parameters=lib_parameters,
                                         penalty_parameter_name=penalty_parameter_name, lib_symbol=lib_symbol,
                                         metrics=metrics)

        self.evaluated_reguls = DataFrame({PENALTY_COL_NAME: []})
        self.warm_dict = {}
        self.metric_groups = metric_groups

    def sequential_search(self, penalty_grid, features, warm_start=True, stopping_condition=lambda x: False,
                          verbose=True, n_jobs=1):
        """Sequential grid search.

        Parameters
        ----------
        penalty_grid : iterable of penalty values
        features : -
        warm_start : boolean, default True
        stopping_condition : any function taking already evaluated reguls as input and returns a boolean
        verbose : boolean, default True
        """
        def partial_search(partial_penalty_grid):
            summary_df_list = []
            for penalty_value in partial_penalty_grid:
                if verbose:
                    print("Evaluating penalty {}".format(penalty_value))
                if warm_start and tuple(features) in self.warm_dict:
                    self.training_params['w_warm'] = self.warm_dict[tuple(features)]
                else:
                    self.training_params['w_warm'] = None
                penalty_extended = self.extend_and_format_penalty(penalty_value, features)
                #if tuple(penalty_extended) in self.evaluated_reguls[PENALTY_COL_NAME]: # to remove
                #    continue

                penalty_summary_df = self.evaluate_and_summarize_cv(penalty_value, features, pred_col_name=PRED_COL_NAME,
                                                                    metric_groups=self.metric_groups)
                summary_df_list.append(penalty_summary_df)

                if stopping_condition(summary_df_list):
                    break

                if warm_start:
                    self.warm_dict[tuple(features)] = np.hstack([np.hstack(self.meta[BAYES_POST_COEF_MEAN_NAME]),
                                                                 self.meta[BAYES_POST_INTERCEPT_MEAN_NAME]])
            return squash(summary_df_list)

        if n_jobs > 1:
            n = int(len(penalty_grid) / float(n_jobs))
            penalty_grid_chunks = [penalty_grid[i:i+n] for i in xrange(0, len(penalty_grid), n)]
            par_summary_df = squash(Parallel(n_jobs, backend='multiprocessing').map(partial_search,
                                                                                    penalty_grid_chunks))  # broken with threading backend
            self.evaluated_reguls = squash(self.evaluated_reguls, par_summary_df)
        else:
            self.evaluated_reguls = squash(self.evaluated_reguls, partial_search(penalty_grid))
        return self.evaluated_reguls

    def summary(self):
        if self.evaluated_reguls is None:
            raise ValueError("You need to generate a cross-validation before!")
        else:
            metric_tuple = tuple(
                ('mean({0}) as mean_{0}'.format(metric), 'standard_deviation({0}) as sd_{0}'.format(metric)) for metric
                in self.metrics)
            metric_tuple = tuple(chain(*metric_tuple))
            penalty_df = self.evaluated_reguls.group_by(['fold_type', 'penalty_hr', 'features'],
                                                        metric_tuple)
        return penalty_df # do someting in summary with betas .copy(exclude=['full_betas'])


class LogisticPenaltySelector(object):
    """Automatic regularization selection.
    >>> from karma.core.utils.utils import use_seed
    >>> with use_seed(62):
    ...     df = DataFrame({"aa": np.random.randn(100, 2), "b": np.random.randn(100, 1), "y": np.random.randint(2, size=100)})
    >>> ps = LogisticPenaltySelector(df, {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 1, 'nb_threads': 1,
    ...                                   'verbose': False, 'nb_inner_threads': 1})
    >>> ps.by_features_search(['b', 'aa'], verbose=False) # doctest: +NORMALIZE_WHITESPACE
    ({'aa': (0.1, 0.1), 'b': (0.1,)}, {'aa': ('auc', -0.3), 'b': ('auc', -0.04)})
    >>> ps.naive_diagonal_search(['aa', 'b'], verbose=False)
    ({'aa': (0.1, 0.1), 'b': (0.1,), "['aa', 'b']": (0.05, 0.05, 0.05)}, {'aa': ('auc', -0.3), 'b': ('auc', -0.04), "['aa', 'b']": ('auc', -0.32)})
    """
    def __init__(self, dataframe, lib_parameters, penalty_parameter_name=BAYES_PRIOR_COEF_VAR_NAME,
                 lib_symbol='logistic_regression', metrics='auc', metric_groups=None):
        self.gs = GridSearch(dataframe, lib_parameters, penalty_parameter_name=penalty_parameter_name,
                             lib_symbol=lib_symbol, metrics=metrics, metric_groups=metric_groups)
        self.evaluator = GridEvaluator()

    def by_features_search(self, features, initial_penalty=None, warm_start=True, stopping_condition=lambda x: False,
                           verbose=True, n_jobs=1, metric='auc', order='max', robustness_threshold=1.,
                           grid='scale_grid', **grid_kwargs):
        tupled_features = coerce_to_tuple_and_check_all_strings(features)
        if initial_penalty is None:
            initial_penalty = np.ones(len(tupled_features)).reshape(-1, 1)
        for feat, initial_pen in zip(tupled_features, initial_penalty):
            if verbose:
                print(feat)
            feat_grid_constructor = GridGenerator(initial_point=np.array(initial_pen))
            grid_constructor = getattr(feat_grid_constructor, grid)
            feat_grid = grid_constructor(**grid_kwargs)
            self.gs.sequential_search(feat_grid, feat, warm_start=warm_start, stopping_condition=stopping_condition,
                                      verbose=verbose, n_jobs=n_jobs)
        return self.evaluator.best_mean_regul(self.gs.evaluated_reguls, metric=metric, order=order,
                                              robustness_threshold=robustness_threshold)

    def naive_diagonal_search(self, features, initial_penalty=None, warm_start=True,
                              stopping_condition=lambda x: False, dyadic_resolution=2,
                              verbose=True, n_jobs=1, metric='auc', order='max', robustness_threshold=1.,
                              grid='scale_grid', **grid_kwargs):


        by_feature_penalty, by_feature_score = self.by_features_search(features,
                                                                       initial_penalty=initial_penalty,
                                                                       warm_start=warm_start,
                                                                       stopping_condition=stopping_condition,
                                                                       verbose=verbose,
                                                                       n_jobs=n_jobs,
                                                                       metric=metric,
                                                                       order=order,
                                                                       robustness_threshold=robustness_threshold,
                                                                       grid=grid,
                                                                       **grid_kwargs)

        shrinkage_parameters = (np.arange(2 ** dyadic_resolution) + 1) / (2. ** dyadic_resolution)
        by_feature_penalty = np.hstack([by_feature_penalty[feat] for feat in features])
        feature_penalty_grid = by_feature_penalty * shrinkage_parameters.reshape(-1, 1)

        self.gs.sequential_search(feature_penalty_grid, features, warm_start=warm_start, stopping_condition=stopping_condition,
                                  verbose=verbose)
        return self.evaluator.best_mean_regul(self.gs.evaluated_reguls, metric=metric, order=order,
                                              robustness_threshold=robustness_threshold)

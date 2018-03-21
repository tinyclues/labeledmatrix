from collections import OrderedDict
from copy import deepcopy
import numpy as np
from karma.core.dataframe import DataFrame
from sklearn.utils.extmath import cartesian
from karma.macros import squash
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
    """
    def __init__(self, dataframe, features, lib_parameters, penalty_parameter_name=BAYES_PRIOR_COEF_VAR_NAME,
                 lib_symbol='logistic_regression', metrics='auc'):
        self.features = features

        self.lib_symbol = lib_symbol
        self.penalty_parameter_name = penalty_parameter_name

        self.training_params = deepcopy(lib_parameters)
        for cv_key in CV_KEYS:
            self.training_params.pop(cv_key, None)

        self.dataframe, self.cv_wrapper = CrossValidationWrapper.create_cv_from_data_and_params(dataframe,
                                                                                                lib_parameters)
        self.training_params['cv'] = self.cv_wrapper

        self.metrics = coerce_to_tuple_and_check_all_strings(metrics)

        self.kc = None
        self.meta = None

    def evaluate_cv(self, penalty_value, pred_col_name=PRED_COL_NAME):
        df_copy = self.dataframe.copy()

        self.training_params[self.penalty_parameter_name] = penalty_value

        self.kc = df_copy.build_karmacode(method=self.lib_symbol,
                                          inputs=self.features,
                                          outputs=pred_col_name,
                                          parameters=self.training_params)

        self.meta = self.kc[-1].meta
        df_copy += self.kc
        return df_copy

    def evaluate_and_summarize_cv(self, penalty_value, pred_col_name=PRED_COL_NAME, metric_groups=None): # add a check in case the lib does not generate a cv
        """
        >>> from karma.core.utils import use_seed
        >>> with use_seed(42):
        ...     random_df = DataFrame({'aa': np.random.randn(20, 9),
        ...                            'b': np.random.randn(20, 3),
        ...                            'group_col': np.random.randint(3, size=20),
        ...                            'strat_col': np.random.randint(3, size=20),
        ...                            'y': np.random.randint(2, size=20)})
        >>> cvs = CVSampler(random_df, ['aa', 'b'], {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 2})
        >>> cvs.evaluate_and_summarize_cv(0.1).copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
        -------------------------------------
        fold_type | Count | CountPos | auc
        -------------------------------------
        test        4       3          1.0
        test        4       3          -1.0
        train       20      13         0.8022
        >>> cvs = CVSampler(random_df, ['aa', 'b'], {'axis': 'y', 'cv': 0.2}, metrics='normalized_log_loss')
        >>> cvs.evaluate_and_summarize_cv(0.1).copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
        --------------------------------------------------
        fold_type | Count | CountPos | normalized_log_loss
        --------------------------------------------------
        test        4       3          0.8127
        train       20      13         0.7824
        >>> cvs = CVSampler(random_df, ['aa', 'b'], {'axis': 'y', 'cv': 0.2}, metrics=['auc', 'normalized_log_loss'])
        >>> cvs.evaluate_and_summarize_cv(0.1).copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
        -----------------------------------------------------------
        fold_type | Count | CountPos | auc    | normalized_log_loss
        -----------------------------------------------------------
        test        4       3          1.0      0.8127
        train       20      13         0.8022   0.7824
        >>> cvs.evaluate_and_summarize_cv(0.1, metric_groups='group_col').copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
        --------------------------------------------------------------------
        fold_type | group_col | Count | CountPos | auc | normalized_log_loss
        --------------------------------------------------------------------
        test        0           1       1          0.0   nan
        test        1           2       1          1.0   0.9949
        test        2           1       1          0.0   nan
        train       0           7       4          1.0   0.6888
        train       1           9       5          0.5   0.8749
        train       2           4       4          0.0   nan
        >>> cvs.evaluate_and_summarize_cv(0.1, metric_groups=['group_col', 'strat_col']).copy(exclude=['penalty', 'full_betas']).preview() # doctest: +NORMALIZE_WHITESPACE
        ---------------------------------------------------------------------------------
        fold_type | group_col | strat_col | Count | CountPos | auc  | normalized_log_loss
        ---------------------------------------------------------------------------------
        test        0           2           1       1          0.0    nan
        test        1           0           1       1          0.0    nan
        test        1           2           1       0          -1.0   nan
        test        2           0           1       1          0.0    nan
        train       0           0           1       1          0.0    nan
        train       0           1           2       0          -1.0   nan
        train       0           2           4       3          1.0    0.6848
        train       1           0           6       4          1.0    0.7435
        train       1           1           1       0          -1.0   nan
        train       1           2           2       1          -1.0   1.0295
        train       2           0           3       3          0.0    nan
        train       2           1           1       1          0.0    nan
        """
        df_copy = self.evaluate_cv(penalty_value)
        train_metric_results = self._metric_aggregation(df_copy, penalty_value, metric_groups)

        train_betas_and_intercept = np.hstack([np.hstack(self.meta[BAYES_POST_COEF_MEAN_NAME]),
                                               self.meta[BAYES_POST_INTERCEPT_MEAN_NAME]])
        train_metric_results['full_betas'] = np.repeat(train_betas_and_intercept.reshape(1, -1),
                                                       len(train_metric_results), axis=0)

        test_metric_results = DataFrame()
        test_betas_and_intercept = map(np.hstack, zip(map(np.hstack, self.meta['cv'].feat_coefs),
                                                      self.meta['cv'].intercepts))
        for i, (fold_test_indices, fold_test_y_hat) in enumerate(self.cv_wrapper.fold_indices_iter):
            predictions_df = self.dataframe.copy()[fold_test_indices][:]
            predictions_df[pred_col_name] = fold_test_y_hat

            test_fold_metric_results = self._metric_aggregation(predictions_df, penalty_value, metric_groups)
            test_fold_metric_results['full_betas'] = np.repeat(test_betas_and_intercept[i].reshape(1, -1),
                                                               len(test_fold_metric_results), axis=0)
            test_metric_results = squash(test_metric_results, test_fold_metric_results)

        final_metric_results = squash({'train': train_metric_results.copy(*test_metric_results.column_names),
                                       'test': test_metric_results}, label='fold_type') # if lazy=True it breaks

        fixed_columns = [PENALTY_COL_NAME, 'full_betas', 'fold_type']
        proper_col_order = fixed_columns + [c for c in final_metric_results.column_names if c not in fixed_columns]

        return final_metric_results.copy(*proper_col_order)

    def _metric_aggregation(self, df, penalty_value, metric_groups=None):
        penalty_extended = self.extend_and_format_penalty(penalty_value)
        metric_agg_tuple = tuple('{0}({1}, {2}) as {0}'.format(metric, PRED_COL_NAME, self.training_params['axis'])
                                 for metric in self.metrics)
        agg_tuple = ('# as Count', 'sum({}) as CountPos'.format(self.training_params['axis']),) + metric_agg_tuple

        df_grouped = df.group_by(metric_groups, agg_tuple)
        df_grouped = df_grouped.with_column(PENALTY_COL_NAME,
                                            'constant(constant={})'.format(tuple(penalty_extended)))
        return df_grouped

    def extend_and_format_penalty(self, penalty_value):
        """Utility function to provide common formatting for penalties.
        >>> df = DataFrame({"aa": np.random.randn(20, 9), "b": np.random.randn(20, 3), "y": np.random.randint(2, size=20)})
        >>> cv_sampler = CVSampler(df, ['aa', 'b'], {'axis': 'y', 'cv': 0.2})
        >>> cv_sampler.extend_and_format_penalty(0.1)
        array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        >>> cv_sampler.extend_and_format_penalty([0.1, 1])
        array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1. , 1. , 1. ])
        """
        if isinstance(self.features, basestring):
            df_bvhs = BasicVirtualHStack([self.dataframe[self.features].safe_dim()])
        else:
            df_bvhs = BasicVirtualHStack([self.dataframe[feat].safe_dim() for feat in self.features])
        return df_bvhs.adjust_array_to_total_dimension(penalty_value)

    def collapse_penalty(self, penalty_value):
        """Utility function to provide minimal description for penalties.
        >>> df = DataFrame({"aa": np.random.randn(20, 9), "b": np.random.randn(20, 3), "y": np.random.randint(2, size=20)})
        >>> cv_sampler = CVSampler(df, ['aa', 'b'], {'axis': 'y', 'cv': 0.2})
        >>> cv_sampler.collapse_penalty((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
        array([0.1, 0.1])
        >>> cv_sampler.collapse_penalty((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1. , 1. , 1.))
        array([0.1, 1. ])
        >>> cv_sampler.collapse_penalty((0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1. , 2. , 1.))
        array([array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
               array([1., 2., 1.])], dtype=object)
        """
        if isinstance(self.features, basestring):
            df_bvhs = BasicVirtualHStack([self.dataframe[self.features].safe_dim()])
        else:
            df_bvhs = BasicVirtualHStack([self.dataframe[feat].safe_dim() for feat in self.features])
        block_penalty = df_bvhs.adjust_to_block_dimensions(penalty_value)
        out_penalty = list()
        for pen_array in block_penalty:
            if (pen_array == pen_array[0]).all():
                out_penalty.append(pen_array[0])
            else:
                out_penalty.append(pen_array)
        return np.array(out_penalty)


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
    >>> gs = GridSearch(df, ['aa', 'b'], {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 2})
    >>> _ = gs.sequential_search([0.1, 1, 2], warm_start=False, verbose=False)
    >>> gs.evaluated_reguls.preview() # doctest: +NORMALIZE_WHITESPACE
    --------------------------------------------------------------------------------------------------------------------
    penalty         | full_betas:0 | full_betas:1 | full_betas:2 | full_betas:3 | fold_type | Count | CountPos | auc
    --------------------------------------------------------------------------------------------------------------------
    (0.1, 0.1, 0.1)   0.0418         -0.106         -0.105         -0.2722        test        20      8          -0.0625
    (0.1, 0.1, 0.1)   0.0974         -0.0489        -0.1625        -0.2701        test        20      8          -0.3333
    (0.1, 0.1, 0.1)   0.1118         -0.0719        -0.0604        -0.3047        train       100     42         0.0837
    (1, 1, 1)         0.0805         -0.1618        -0.1601        -0.2547        test        20      8          -0.0417
    (1, 1, 1)         0.1626         -0.0794        -0.2346        -0.2556        test        20      8          -0.2708
    (1, 1, 1)         0.1728         -0.1044        -0.0906        -0.2961        train       100     42         0.0878
    (2, 2, 2)         0.0846         -0.1667        -0.1651        -0.2531        test        20      8          -0.0417
    (2, 2, 2)         0.1689         -0.0823        -0.241         -0.2543        test        20      8          -0.25
    (2, 2, 2)         0.1783         -0.1071        -0.0932        -0.2953        train       100     42         0.0862
    """
    def __init__(self, dataframe, features, lib_parameters, penalty_parameter_name=BAYES_PRIOR_COEF_VAR_NAME,
                 lib_symbol='logistic_regression', metrics='auc', metric_groups=None):
        super(GridSearch, self).__init__(dataframe=dataframe, features=features, lib_parameters=lib_parameters,
                                         penalty_parameter_name=penalty_parameter_name, lib_symbol=lib_symbol,
                                         metrics=metrics)

        self.evaluated_reguls = DataFrame({PENALTY_COL_NAME: []})
        self.metric_groups = metric_groups

    def sequential_search(self, penalty_grid, warm_start=True, stopping_condition=lambda x: False, verbose=True):
        """Sequential grid search.

        Parameters
        ----------
        penalty_grid : iterable of penalty values
        warm_start : boolean, default True
        stopping_condition : any function taking already evaluated reguls as input and returns a boolean
        verbose : boolean, default True
        """
        for penalty_value in penalty_grid:
            if verbose:
                print("Evaluating penalty {}".format(penalty_value))
            penalty_extended = self.extend_and_format_penalty(penalty_value)
            if tuple(penalty_extended) in self.evaluated_reguls[PENALTY_COL_NAME]:
                continue

            penalty_summary_df = self.evaluate_and_summarize_cv(penalty_value, pred_col_name=PRED_COL_NAME,
                                                                metric_groups=self.metric_groups)
            self.evaluated_reguls = squash(self.evaluated_reguls, penalty_summary_df)

            if stopping_condition(penalty_summary_df):
                break

            if warm_start:
                self.training_params['w_warm'] = np.hstack([np.hstack(self.meta[BAYES_POST_COEF_MEAN_NAME]),
                                                           self.meta[BAYES_POST_INTERCEPT_MEAN_NAME]])
        return self.evaluated_reguls

    def parallel_search(self, penalty_grid, n_jobs=1):
        """Sequential grid search.

        Parameters
        ----------
        penalty_grid : iterable of penalty values
        warm_start : boolean, default True
        stopping_condition : any function taking already evaluated reguls as input and returns a boolean
        >>> from karma.core.utils.utils import use_seed
        >>> with use_seed(42):
        ...     df = DataFrame({"aa": np.random.randn(100, 2), "b": np.random.randn(100, 1), "y": np.random.randint(2, size=100)})
        >>> gs = GridSearch(df, ['aa', 'b'], {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 2})
        >>> _ = gs.parallel_search([0.1, 1, 2], n_jobs=3)
        >>> gs.evaluated_reguls.preview() # doctest: +NORMALIZE_WHITESPACE
        --------------------------------------------------------------------------------------------------------------------
        penalty         | full_betas:0 | full_betas:1 | full_betas:2 | full_betas:3 | fold_type | Count | CountPos | auc
        --------------------------------------------------------------------------------------------------------------------
        (0.1, 0.1, 0.1)   0.0418         -0.106         -0.105         -0.2722        test        20      8          -0.0625
        (0.1, 0.1, 0.1)   0.0974         -0.0489        -0.1625        -0.2701        test        20      8          -0.3333
        (0.1, 0.1, 0.1)   0.1118         -0.0719        -0.0604        -0.3047        train       100     42         0.0837
        (1, 1, 1)         0.0805         -0.1618        -0.1601        -0.2547        test        20      8          -0.0417
        (1, 1, 1)         0.1626         -0.0794        -0.2346        -0.2556        test        20      8          -0.2708
        (1, 1, 1)         0.1728         -0.1044        -0.0906        -0.2961        train       100     42         0.0878
        (2, 2, 2)         0.0846         -0.1667        -0.1651        -0.2531        test        20      8          -0.0417
        (2, 2, 2)         0.1689         -0.0823        -0.241         -0.2543        test        20      8          -0.25
        (2, 2, 2)         0.1783         -0.1071        -0.0932        -0.2953        train       100     42         0.0862
        """
        n_jobs = min(min(n_jobs, len(penalty_grid)), 6) # Clipping

        def mono_search(penalty_value):
            penalty_extended = self.extend_and_format_penalty(penalty_value)

            if tuple(penalty_extended) in self.evaluated_reguls[PENALTY_COL_NAME]:
                return DataFrame({PENALTY_COL_NAME: []})
            else:
                return self.evaluate_and_summarize_cv(penalty_value, pred_col_name=PRED_COL_NAME,
                                                      metric_groups=self.metric_groups)
        par_summary_df = squash(Parallel(n_jobs, backend='multiprocessing').map(mono_search, penalty_grid)) # broken with threading backend
        self.evaluated_reguls = squash(self.evaluated_reguls, par_summary_df)
        return self.evaluated_reguls


class LogisticPenaltySelector(GridSearch):
    """Automatic selection of penalty parameters for logistic regressions.

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
    ...     df = DataFrame({"aa": np.random.randn(20, 9), "b": np.random.randn(20, 3), "y": np.random.randint(2, size=20)})
    >>> lps = LogisticPenaltySelector(df, ['aa', 'b'], {'axis': 'y', 'cv': 0.2})
    >>> lps.naive_diagonal_search(verbose=False)
    >>> lps.best_mean_score_param(robustness_factor=0.1)
    ((0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025), (('auc',), 0.476))
    """
    def __init__(self, dataframe, features, lib_parameters, penalty_parameter_name=BAYES_PRIOR_COEF_VAR_NAME,
                 lib_symbol='logistic_regression', metrics='auc', metric_groups=None):
        super(LogisticPenaltySelector, self).__init__(dataframe=dataframe, features=features, lib_parameters=lib_parameters,
                                                      penalty_parameter_name=penalty_parameter_name, lib_symbol=lib_symbol,
                                                      metrics=metrics, metric_groups=metric_groups)

        self.best_single_feature_penalty = OrderedDict()
        self.single_feature_evaluated_reguls = OrderedDict()
        self.best_penalty = None
        self.best_score = None

    @staticmethod
    def generate_linear_grid(initial_point, granularity=2, width=0.25):
        """Utility function to generate an hyper mesh.
        >>> my_df = DataFrame({'a': np.random.randn(100, 5), 'b': np.random.randn(100, 7), 'y': np.random.binomial(1, 0.75, size=100)})
        >>> my_params = {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 3}
        >>> lps = LogisticPenaltySelector(my_df, ['a', 'b'], my_params)
        >>> lps.generate_linear_grid(np.array([0.1, 1.]), width=0.5)
        array([[0.05, 0.5 ],
               [0.05, 1.5 ],
               [0.15, 0.5 ],
               [0.15, 1.5 ]])
        >>> lps.generate_linear_grid(np.array([1., 1.]), granularity=3)
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
        grid_correction = width * initial_point
        corrected_grid = cartesian([one_dim_grid for _ in xrange(len(initial_point))]).dot(np.diag(grid_correction))
        local_linear_grid = initial_point + corrected_grid
        local_linear_grid = np.clip(local_linear_grid, a_min=1e-6, a_max=None)  # bring grid back to positive orthant
        return local_linear_grid

    @staticmethod
    def generate_scale_grid(initial_point, downscale=-1, upscale=1):
        """Utility function to generate a logarithmic hyper mesh.
        >>> my_df = DataFrame({'a': np.random.randn(100, 5), 'b': np.random.randn(100, 7), 'y': np.random.binomial(1, 0.75, size=100)})
        >>> my_params = {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 3}
        >>> lps = LogisticPenaltySelector(my_df, ['a', 'b'], my_params)
        >>> lps.generate_scale_grid(np.array([0.1, 1.]), downscale=-1, upscale=0)
        array([[0.01, 0.1 ],
               [0.01, 1.  ],
               [0.1 , 0.1 ],
               [0.1 , 1.  ]])
        >>> lps.generate_scale_grid(np.array([1., 1.]), -1, 1)
        array([[ 0.1,  0.1],
               [ 0.1,  1. ],
               [ 0.1, 10. ],
               [ 1. ,  0.1],
               [ 1. ,  1. ],
               [ 1. , 10. ],
               [10. ,  0.1],
               [10. ,  1. ],
               [10. , 10. ]])
        """
        one_dim_grid = 10. ** np.arange(downscale, upscale + 1)
        return initial_point * cartesian([one_dim_grid for _ in xrange(len(initial_point))])

    @staticmethod
    def generate_gaussian_grid(initial_point, n_samples=10, var_factor=1.):
        gaussian_grid = initial_point + var_factor * np.random.randn(n_samples, initial_point.shape[0])
        return np.clip(gaussian_grid, a_min=1e-6, a_max=None)

    def generate_grid_and_search(self, warm_start=True, stopping_condition=lambda x: False,
                                 verbose=True, grid_generation='generate_linear_grid', **grid_kwargs):
        initial_point = grid_kwargs.get('initial_point')
        if initial_point is None:
            features_tuple = coerce_to_tuple_and_check_all_strings(self.features)
            initial_point = np.ones(len(features_tuple))

        penalty_array = np.array(initial_point, dtype=np.float)
        grid_kwargs['initial_point'] = penalty_array

        grid_generation = getattr(self, grid_generation)
        grid_to_evaluate = grid_generation(**grid_kwargs)
        return self.sequential_search(grid_to_evaluate, warm_start=warm_start, stopping_condition=stopping_condition,
                                      verbose=verbose)

    def generate_grid_and_search_by_features(self, warm_start=True, stopping_condition=lambda x: False,
                                             verbose=True, metric='auc', order='max', robustness_factor=0.,
                                             grid_generation='generate_linear_grid', **grid_kwargs):

        sfselectors = [LogisticPenaltySelector(dataframe=self.dataframe,
                                               features=feat,
                                               lib_parameters=self.training_params,
                                               penalty_parameter_name=self.penalty_parameter_name,
                                               lib_symbol=self.lib_symbol,
                                               metrics=self.metrics) for feat in self.features]
        for selector in sfselectors:
            if verbose:
                print("Evaluating feature : {}".format(selector.features))
            selector_evaluated_reguls = selector.generate_grid_and_search(warm_start=warm_start,
                                                                          stopping_condition=stopping_condition,
                                                                          verbose=verbose,
                                                                          grid_generation=grid_generation,
                                                                          **grid_kwargs)
            self.single_feature_evaluated_reguls[selector.features] = selector_evaluated_reguls
            selector.best_mean_score_param(metric=metric, order=order, robustness_factor=robustness_factor)
            self.best_single_feature_penalty[selector.features] = selector.best_penalty

        mono_penalties = map(np.array, self.best_single_feature_penalty.values())
        global_penalty = self.collapse_penalty(np.hstack(mono_penalties))
        return global_penalty

    def naive_diagonal_search(self, warm_start=True, stopping_condition=lambda x: False, dyadic_resolution=2,
                              verbose=True, metric='auc', order='max', robustness_factor=0.,
                              grid_generation='generate_scale_grid', **grid_kwargs):


        by_feature_penalty = self.generate_grid_and_search_by_features(warm_start=warm_start,
                                                                       stopping_condition=stopping_condition,
                                                                       verbose=verbose, metric=metric, order=order,
                                                                       robustness_factor=robustness_factor,
                                                                       grid_generation=grid_generation,
                                                                       **grid_kwargs)

        shrinkage_parameters = (np.arange(2 ** dyadic_resolution) + 1) / (2. ** dyadic_resolution)
        feature_penalty_grid = by_feature_penalty * shrinkage_parameters.reshape(-1, 1)

        self.sequential_search(feature_penalty_grid, warm_start=warm_start, stopping_condition=stopping_condition,
                               verbose=verbose)

    def best_mean_score_param(self, metric='auc', order='max', robustness_factor=0.):
        """Decide what is the best penalty already evaluated.
        >>> from karma.core.utils.utils import use_seed
        >>> with use_seed(42):
        ...     my_df = DataFrame({'a': np.random.randn(100, 5), 'b': np.random.randn(100, 7), 'y': np.random.binomial(1, 0.75, size=100)})
        >>> my_params = {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 1}
        >>> lps = LogisticPenaltySelector(my_df, ['a', 'b'], my_params)
        >>> _ = lps.sequential_search([0.1, 1], verbose=False)
        >>> lps.best_mean_score_param()
        ((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), (('auc',), -0.3867))
        """
        test_penalty_df = self._compare_train_test(metric=metric)

        penalty_vec = test_penalty_df[PENALTY_COL_NAME][:]
        metric_vec = test_penalty_df['{}'.format(metric)][:]
        delta_score = test_penalty_df['train_test_error_gap'][:]
        metric_vec = metric_vec - robustness_factor * np.abs(delta_score)

        if order == 'max':
            best_ind = np.argmax(metric_vec)
        elif order == 'min':
            metric_vec = -metric_vec
            best_ind = np.argmax(metric_vec)
        else:
            raise ValueError("order argument should be max or min")
        self.best_penalty = penalty_vec[best_ind]
        self.best_score = (self.metrics, metric_vec[best_ind])
        return self.best_penalty, self.best_score  # fix printing of best penalty atm tuple with one value by row ...

    def _compare_train_test(self, metric):
        if self.evaluated_reguls is None:
            raise ValueError("You need to generate a cross-validation before!")
        else:
            penalty_df = self.evaluated_reguls.group_by(['fold_type', PENALTY_COL_NAME],
                                                         'mean({0}) as {0}'.format(metric))
        splitted_df = penalty_df.split_by('fold_type')
        train_score = splitted_df['train']['{}'.format(metric)][:]
        test_score = splitted_df['test']['{}'.format(metric)][:]
        delta_score = np.abs(np.array(train_score) - np.array(test_score))
        splitted_df['test']['train_test_error_gap'] = delta_score
        return splitted_df['test']

    # DO NOT USE FROM HERE
    def free_search(self, dict_of_search):
        """We suppose that we take as input dicts which specify the full search. We need three keys:
        search_method : dict with a name key and params
        best_method : dict with a name key and params (supposed to let the search how to select a "best" model)
        grids : list of dicts, each dict specifies the type of grid to generate and its params
        """
        search_dict = dict_of_search.get('search_method')
        best_dict = dict_of_search.get('best_method')
        search_sequence = dict_of_search.get('search_sequence')

        search_method = getattr(self, search_dict['name'])
        search_dict.pop('name')

        best_method = getattr(self, best_dict['name'])
        best_dict.pop('name')

        for i, grid in enumerate(search_sequence):
            grid_generation = getattr(self, grid['name'])
            grid.pop('name')

            if i > 0:
                optimal_penalty, _ = best_method(**best_dict)
                grid['initial_point'] = np.array(optimal_penalty)
            else:
                features_tuple = coerce_to_tuple_and_check_all_strings(self.features)
                initial_point = np.ones(len(features_tuple))
                grid['initial_point'] = initial_point

            grid_to_evaluate = grid_generation(**grid)
            search_method(grid_to_evaluate, **search_dict)

    def geometric_linear_search(self, warm_start=True, stopping_condition=lambda x: False, initial_penalty=None,
                                metric='auc', order='max', robustness_factor=0., downscale=-2, upscale=0,
                                granularity=2, width=0.25):

        scale_grid_kwargs = {'initial_point': initial_penalty,
                             'downscale': downscale,
                             'upscale': upscale}
        self.generate_grid_and_search(warm_start=warm_start, stopping_condition=stopping_condition,
                                      grid_generation='generate_scale_grid', **scale_grid_kwargs)

        optimal_penalty, _ = self.best_mean_score_param(metric=metric, order=order, robustness_factor=robustness_factor)
        optimal_penalty = self.collapse_penalty(optimal_penalty)
        print("Finished geometric search")

        linear_grid_kwargs = {'initial_point': optimal_penalty,
                              'granularity': granularity,
                              'width': width}
        self.generate_grid_and_search(warm_start=warm_start, stopping_condition=stopping_condition,
                                      grid_generation='generate_linear_grid', **linear_grid_kwargs)

    def by_features_search(self, sf_search_method='geometric_linear_search', **search_kwargs):

        sfselectors = [LogisticPenaltySelector(dataframe=self.dataframe,
                                               features=feat,
                                               lib_parameters=self.training_params,
                                               penalty_parameter_name=self.penalty_parameter_name,
                                               lib_symbol=self.lib_symbol,
                                               metrics=self.metrics) for feat in self.features]
        for selector in sfselectors:
            search_method = getattr(selector, sf_search_method)
            search_method(**search_kwargs)
            self.best_single_feature_penalty[selector.features] = selector.best_penalty

        mono_penalties = map(np.array, self.best_single_feature_penalty.values())
        global_penalty = self.collapse_penalty(np.hstack(mono_penalties))
        return global_penalty

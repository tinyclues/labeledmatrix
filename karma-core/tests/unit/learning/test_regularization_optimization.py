import unittest

import numpy as np
from cyperf.matrix.karma_sparse import KarmaSparse
from scipy.sparse import rand

from karma.core.dataframe import DataFrame
from karma.core.utils.utils import use_seed
from karma.learning.bayesian_constants import BAYES_PRIOR_COEF_VAR_NAME, BAYES_POST_COEF_MEAN_NAME
from karma.learning.regularization_optimization import CVSampler, PRED_COL_NAME, PENALTY_COL_NAME, GridSearch
from karma.learning.utils import CrossValidationWrapper


class CVSamplerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N = 100
        with use_seed(42):
            cls.random_df = DataFrame({'aa': np.random.randn(N, 9),
                                       'b': np.random.randn(N, 3),
                                       'c': KarmaSparse(rand(N, 10, 0.1)),
                                       'strat_col': np.random.randint(4, size=N),
                                       'group_col': np.random.randint(6, size=N),
                                       'y': np.random.binomial(1, p=0.3, size=N)})

    def test_evaluate_cv(self):
        df = self.random_df.copy()
        my_features = ['aa', 'b', 'c']
        my_params = {'axis': 'y', 'cv': 0.2, 'n_splits': 1, 'cv_groups': 'group_col'}
        cv_sampler = CVSampler(df, my_params)
        df_post_eval = cv_sampler.evaluate_cv(0.1, my_features)
        self.assertTrue(PRED_COL_NAME in df_post_eval)
        self.assertTrue(BAYES_POST_COEF_MEAN_NAME in cv_sampler.meta)

    def test_metric_aggregation(self):
        df = self.random_df.copy()
        my_features = ['aa', 'b', 'c']
        my_params = {'axis': 'y', 'cv': 0.2, 'n_splits': 1, 'cv_groups': 'group_col'}
        cv_sampler = CVSampler(df, my_params)
        df_copy = cv_sampler.evaluate_cv(0.1, my_features)
        metric_results = cv_sampler._metric_aggregation(df_copy, 0.1, my_features, metric_groups=['group_col', 'strat_col'])
        self.assertTrue('group_col' in metric_results)
        self.assertTrue('strat_col' in metric_results)
        self.assertEqual(metric_results.column_names,
                         ['group_col', 'strat_col', 'Count', 'CountPos', 'auc', 'penalty', 'penalty_hr'])

    def test_evaluate_and_summarize_cv(self):
        df = self.random_df.copy()
        my_features = ['aa', 'b']
        my_params = {'axis': 'y', 'cv': 0.2, 'n_splits': 1}
        cv_sampler = CVSampler(df, my_params)
        metric_results = cv_sampler.evaluate_and_summarize_cv(penalty_value=0.1, features=my_features)
        self.assertEqual(len(metric_results), cv_sampler.cv_wrapper.n_splits + 1)
        self.assertEqual(metric_results.column_names,
                         ['penalty', 'full_betas', 'fold_type', 'Count', 'CountPos', 'auc', 'penalty_hr', 'features'])


class GridSearchTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N = 100
        with use_seed(42):
            cls.random_df = DataFrame({'aa': np.random.randn(N, 9),
                                       'b': np.random.randn(N, 3),
                                       'c': KarmaSparse(rand(N, 10, 0.1)),
                                       'strat_col': np.random.randint(4, size=N),
                                       'group_col': np.random.randint(6, size=N),
                                       'y': np.random.randint(2, size=N)})

    def test_init(self):
        df = self.random_df.copy()
        my_params = {'axis': 'y', 'cv': 0.2, 'cv_n_splits': 2}
        grid_search = GridSearch(df, my_params)

        self.assertEqual(grid_search.dataframe, df)
        self.assertEqual(grid_search.penalty_parameter_name, BAYES_PRIOR_COEF_VAR_NAME)
        self.assertEqual(grid_search.lib_symbol, 'logistic_regression')
        self.assertEqual(grid_search.metrics, ('auc',))
        self.assertTrue(isinstance(grid_search.cv_wrapper, CrossValidationWrapper))
        self.assertTrue(isinstance(grid_search.training_params['cv'], CrossValidationWrapper))
        self.assertFalse('cv_n_splits' in grid_search.training_params)
        self.assertTrue(grid_search.metric_groups is None)
        self.assertEqual(grid_search.evaluated_reguls, DataFrame({PENALTY_COL_NAME: []}))

    def test_sequential_search(self):
        df = self.random_df.copy()
        my_features = ['aa', 'b', 'c']
        my_params = {'axis': 'y', 'cv': 0.2, 'n_splits': 1}
        grid_search = GridSearch(df, my_params)
        initial_penalty_values = [0.1, 1]
        evaluated_reguls_df = grid_search.sequential_search(initial_penalty_values, my_features)
        self.assertEqual(evaluated_reguls_df.column_names,
                         [PENALTY_COL_NAME, 'full_betas', 'fold_type', 'Count', 'CountPos', 'auc', 'penalty_hr', 'features'])
        self.assertEqual(len(evaluated_reguls_df),
                         (grid_search.cv_wrapper.n_splits + 1) * len(initial_penalty_values))
        self.assertEqual(len(grid_search.evaluated_reguls),
                         (grid_search.cv_wrapper.n_splits + 1) * len(initial_penalty_values))

        new_penalty_values = [0.01, 10]
        evaluated_reguls_df = grid_search.sequential_search(new_penalty_values, my_features, n_jobs=2)
        self.assertEqual(len(evaluated_reguls_df),
                         (grid_search.cv_wrapper.n_splits + 1) * (len(initial_penalty_values) + len(new_penalty_values)))
        self.assertEqual(len(grid_search.evaluated_reguls),
                         (grid_search.cv_wrapper.n_splits + 1) * (len(initial_penalty_values) + len(new_penalty_values)))

import unittest

import numpy as np

from karma import create_column_from_data
from karma.core.dataframe import DataFrame
from karma.core.utils.utils import use_seed
from karma.learning.utils import CrossValidationWrapper
from karma.lib.logistic_regression import logistic_regression
from numpy.testing import assert_equal

class CrossValidationWrapperTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 10 ** 4
        with use_seed(1234567):
            df = DataFrame(
                {'i': np.arange(n), 'x': np.random.rand(n), 'y': np.random.randint(2, size=n)})
        df['group'] = df['str({i} % 10)']
        df['y_group'] = df['str({y}) + "_" + {group}']
        cls.df = df

    def test_basic(self):
        df = self.df.copy()
        cv = CrossValidationWrapper(0.2, df['y'][:], n_splits=6, seed=123)
        _ = logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv})

        self.assertEquals(cv.test_size, 2000)
        self.assertEquals(len(cv.test_indices), 12000)
        self.assertEquals(len(cv.test_y_hat), 12000)

        self.assertAlmostEqual(cv.meta['train_MSE'], 0.25, places=2)

    def test_stratified_shuffle_split(self):
        df = self.df.copy()
        cv = CrossValidationWrapper(0.2, df['y'][:], n_splits=1, seed=123)
        _ = logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv})

        train_test = np.full(len(df), 'train')
        train_test[cv.test_indices] = 'test'
        df['tt'] = train_test
        res = df.group_by(('y', 'tt'), '#').pivot('y', 'tt', aggregate_by='sum(#)')
        count_ratios = res['divide(sum(#):test, add(sum(#):test, sum(#):train))'][:]

        self.assertGreater(min(count_ratios), cv.test_fraction - 0.0010)
        self.assertLess(max(count_ratios), cv.test_fraction + 0.0010)

    def test_stratified_shuffle_split_with_groups(self):
        df = self.df.copy()
        cv = CrossValidationWrapper(0.2, df['y'][:], groups=df['group'][:], seed=140191)
        _ = logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv})

        train_test = np.full(len(df), 'train')
        train_test[cv.test_indices] = 'test'
        df['tt'] = train_test
        res = df.group_by(('y_group', 'tt'), '#').pivot('y_group', 'tt', aggregate_by='sum(#)')
        count_ratios = res['divide(sum(#):test, add(sum(#):test, sum(#):train))'][:]

        self.assertGreater(min(count_ratios), cv.test_fraction - 0.0010)
        self.assertLess(max(count_ratios), cv.test_fraction + 0.0010)

    def test_metrics(self):
        df = self.df.copy()
        df['dummy' ] = create_column_from_data( np.random.rand(len(df), 2))
        cv = CrossValidationWrapper(0.2, df['y'][:], n_splits=2, seed=123)
        df += logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv})

        metrics = cv.calculate_train_test_metrics(df, ['group'], 'pred_y', 'y')
        self.assertEquals(metrics.keys(), ['group'])
        self.assertEquals(metrics['group'].column_names, ['group', 'AUC train', 'AUC test', 'NLL train', 'NLL test',
                                                          'Calibration train', 'Calibration test'])

    def test_classes(self):
        cv = CrossValidationWrapper(0.2,
                                    [1, 1, 0, 1, 1, 0],
                                    [2, 2, 1, 1, 1, 1], n_splits=1, seed=None)
        assert_equal(cv.classes, ['1_2', '1_2', '0_1', '1_1', '1_1', '0_1'])

        # class of size 1
        cv = CrossValidationWrapper(0.2,
                                    [1, 0, 0, 1, 1, 0],
                                    [2, 2, 1, 1, 1, 1], n_splits=1, seed=None)
        assert_equal(cv.classes, ['2', '2', '0_1', '1_1', '1_1', '0_1'])

        cv = CrossValidationWrapper(0.2,
                                    [1, 0, 0, 1, 1, 0, 0],
                                    [2, 2, 1, 1, 1, 1, 2], n_splits=1, seed=None)
        assert_equal(cv.classes, ['2', '2', '0_1', '1_1', '1_1', '0_1', '2'])

        with self.assertRaises(ValueError) as e:
            _ = CrossValidationWrapper(0.2,
                                        [1, 0, 0, 1, 1, 0],
                                        [3, 2, 1, 1, 1, 1], n_splits=1, seed=None)
        self.assertEquals(e.exception.message, "StratifiedShuffleSplit doesn't support classes of size 1")

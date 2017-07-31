import unittest

import numpy as np
from numpy.testing import assert_equal
from sklearn.model_selection import StratifiedShuffleSplit

from karma.core.dataframe import DataFrame
from karma.core.utils.utils import use_seed
from karma.learning.utils import get_indices_from_cv


class LearningUtilsGetTrainTestIdxTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 10 ** 4
        with use_seed(1234567):
            df = DataFrame(
                {'x': np.arange(n), 'y': np.random.randint(2, size=n)})
        df['group'] = df['str({x} % 10)']
        df['y_group'] = df['str({y}) + "_" + {group}']
        cls.df = df
        cls.cv_fraction = 0.2
        cls.ref_split = get_indices_from_cv(cls.cv_fraction, df['y'][:], groups=df['group'][:], seed=140191)

    def test_unicity(self):
        for indices in self.ref_split:
            assert_equal(np.sort(indices), np.unique(indices))

    def test_fallback_logic(self):
        assert_equal(self.ref_split, get_indices_from_cv(self.ref_split, np.random.rand(len(self.df))))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=140191)
        assert_equal(self.ref_split, next(sss.split(np.arange(len(self.df)),
                                                    self.df['str({y}) + "_" + {group}'][:])))

    def test_stratified_shuffle_split(self):
        df = self.df.copy()
        split = get_indices_from_cv(self.cv_fraction, df['y'][:], seed=140191)
        train_test = np.full(len(df), 'train')
        train_test[split[1]] = 'test'
        df['tt'] = train_test
        res = df.group_by(('y', 'tt'), '#').pivot('y', 'tt', aggregate_by='sum(#)')
        count_ratios = res['divide(sum(#):test, add(sum(#):test, sum(#):train))'][:]

        self.assertGreater(min(count_ratios), self.cv_fraction - 0.0010)
        self.assertLess(max(count_ratios), self.cv_fraction + 0.0010)

    def test_stratified_shuffle_split_with_groups(self):
        df = self.df.copy()
        train_test = np.full(len(df), 'train')
        train_test[self.ref_split[1]] = 'test'
        df['tt'] = train_test
        res = df.group_by(('y_group', 'tt'), '#').pivot('y_group', 'tt', aggregate_by='sum(#)')
        count_ratios = res['divide(sum(#):test, add(sum(#):test, sum(#):train))'][:]

        self.assertGreater(min(count_ratios), self.cv_fraction - 0.0010)
        self.assertLess(max(count_ratios), self.cv_fraction + 0.0010)

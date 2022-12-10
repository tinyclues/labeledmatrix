import unittest

import numpy as np
from cyperf.indexing.indexed_list import is_increasing
from cyperf.matrix.karma_sparse import KarmaSparse, DTYPE
from mock import patch
from numpy.testing import assert_equal, assert_array_equal, assert_almost_equal
from sklearn.model_selection import StratifiedShuffleSplit

from karma import create_column_from_data
from karma.core.dataframe import DataFrame
from karma.core.utils.utils import use_seed
from karma.learning.logistic import logistic_coefficients
from karma.learning.matrix_utils import as_vector_batch
from karma.learning.utils import (CrossValidationWrapper, validate_regression_model, VirtualDirectProduct,
                                  BasicVirtualHStack, VirtualHStack, NB_THREADS_MAX, _prepare_and_check_classes)
from karma.lib.bayesian_logistic_regression import bayesian_logistic_regression
from karma.lib.logistic_regression import logistic_regression
from six.moves import range


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
        _ = df.build_lib_column('logistic_regression', 'x', parameters={'axis': 'y', 'cv': cv}, output='pred_y')

        self.assertEquals(cv.test_size, 2000)
        self.assertEquals(len(cv.test_indices), 12000)
        self.assertEquals(len(cv.test_y_hat), 12000)
        self.assertIsNotNone(cv.method_output)

        self.assertAlmostEqual(cv.meta['train_MSE'], 0.25, places=2)

        self.assertIsInstance(cv.cv, StratifiedShuffleSplit)
        self.assertEqual(cv.cv.test_size, 0.2)

    def test_stratified_shuffle_split(self):
        df = self.df.copy()
        cv = CrossValidationWrapper(0.2, df['y'][:], n_splits=1, seed=123)
        _ = logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv})

        self.assertTrue(is_increasing(cv.test_indices))
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
        df['dummy'] = create_column_from_data(np.random.rand(len(df), 2))
        cv = CrossValidationWrapper(0.2, df['y'][:], n_splits=2, seed=123)
        df += logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv})

        metrics = cv.calculate_train_test_metrics(df, ['group'], 'pred_y', 'y')
        self.assertEquals(metrics.keys(), ['group'])
        self.assertEquals(metrics['group'].column_names, ['group', '#', '# positive', 'AUC train', 'AUC test',
                                                          'RIG train', 'RIG test', 'Calib train', 'Calib test'])

    def test_classes(self):
        cv = CrossValidationWrapper(0.2,
                                    [1, 1, 0, 1, 1, 0],
                                    [2, 2, 1, 1, 1, 1], n_splits=1, seed=None)
        assert_equal(cv._classes, ['1_2', '1_2', '0_1', '1_1', '1_1', '0_1'])

        # class of size 1
        cv = CrossValidationWrapper(0.2,
                                    [1, 0, 0, 1, 1, 0],
                                    [2, 2, 1, 1, 1, 1], n_splits=1, seed=None)
        assert_equal(cv._classes, ['2', '2', '0_1', '1_1', '1_1', '0_1'])

        cv = CrossValidationWrapper(0.2,
                                    [1, 0, 0, 1, 1, 0, 0],
                                    [2, 2, 1, 1, 1, 1, 2], n_splits=1, seed=None)
        assert_equal(cv._classes, ['2', '2', '0_1', '1_1', '1_1', '0_1', '2'])

        with self.assertRaises(ValueError) as e:
            _ = CrossValidationWrapper(0.2,
                                       [1, 0, 0, 1, 1, 0],
                                       [3, 2, 1, 1, 1, 1], n_splits=1, seed=None)
        self.assertEquals(e.exception.message, "StratifiedShuffleSplit doesn't support classes of size 1")

    def test_sample_weight(self):
        df = self.df.copy()
        with use_seed(1516):
            kwargs = {'max_iter': 150, 'solver': 'lbfgs', 'C': 1e10,
                      'sample_weight': np.random.rand(len(df)), 'w_warm': np.zeros(2)}
        cv = validate_regression_model([as_vector_batch(df['x'][:])], df['y'][:], 0.2, logistic_coefficients,
                                       warmup_key='w_warm', **kwargs)
        self.assertAlmostEqual(cv.meta['train_MSE'], 0.25, places=2)
        self.assertEqual(cv.meta['curves'].auc, 0.0062)

    def test_np_array_kwargs(self):
        df = self.df.copy()
        with use_seed(1516):
            kwargs = {'max_iter': 150, 'solver': 'lbfgs', 'C': np.array(1e10),
                      'sample_weight': np.random.rand(len(df)), 'w_warm': np.zeros(2)}
        cv = validate_regression_model([as_vector_batch(df['x'][:])], df['y'][:], 0.2, logistic_coefficients,
                                       warmup_key='w_warm', **kwargs)
        self.assertAlmostEqual(cv.meta['train_MSE'], 0.25, places=2)
        self.assertEqual(cv.meta['curves'].auc, 0.0062)

    def test_weights_storage(self):
        df = self.df.copy()
        cv = CrossValidationWrapper(0.2, df['y'][:], n_splits=6, seed=123)
        _ = df.build_lib_column('logistic_regression', 'x', parameters={'axis': 'y', 'cv': cv}, output='pred_y')
        self.assertEqual(len(cv.feat_coefs), cv.n_splits)
        self.assertEqual(len(cv.intercepts), cv.n_splits)

    def test_fold_iter(self):
        df = self.df.copy()
        cv = CrossValidationWrapper(0.2, df['y'][:], n_splits=6, seed=123)
        test_indices_list = []
        test_y_hat_list = []
        for test_indices, test_y_hat in cv.fold_indices_iter:
            test_indices_list.append(test_indices)
            test_y_hat_list.append(test_y_hat)
        self.assertEqual(len(test_indices_list), cv.n_splits)
        self.assertEqual(len(test_y_hat_list), cv.n_splits)

    def test_cv_static_creation(self):
        dataframe = DataFrame({'y': np.random.randint(0, 2, 55),
                               'gr': ['a'] * 20 + ['b'] * 30 + ['c', 'd', 'e', 'f', 'g']})
        cv = CrossValidationWrapper.create_cv_from_data_and_params(dataframe, cv=0.2, cv_groups='gr', axis='y')
        self.assertEqual(len(cv._kept_indices), 50)
        self.assertEqual(len(np.unique(cv._kept_indices)), 50)
        self.assertEqual(cv._classes.shape[0], 50)
        expected_test_size = cv.split().next()[1].shape[0]
        self.assertEqual(expected_test_size, 10)
        self.assertEqual(expected_test_size, cv.test_size)

        with self.assertRaises(AssertionError):
            _ = CrossValidationWrapper(0.2, [1.1, 0.5], ['a', 'b', 'c'])

        with self.assertRaises(AssertionError):
            _ = CrossValidationWrapper(0.2, [0, 1], ['a', 'b', 'c'])

    def test_summary_in_validate(self):
        df = self.df.copy()

        cv = CrossValidationWrapper(0.2, df['y'][:], groups=df['group'][:], seed=140191)
        _ = logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv, 'compute_summary': True})
        self.assertEqual(len(cv.summary), 1)
        cv = CrossValidationWrapper(0.2, df['y'][:], groups=df['group'][:], seed=140191)
        _ = bayesian_logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv, 'compute_summary': True})
        self.assertEqual(len(cv.summary), 1)

    def test_nogaincurves_in_validate(self):
        df = self.df.copy()

        cv = CrossValidationWrapper(0.2, df['y'][:], groups=df['group'][:], seed=140191)
        _ = logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv, 'compute_gaincurves': False})
        self.assertTrue(cv.meta is None)
        cv = CrossValidationWrapper(0.2, df['y'][:], groups=df['group'][:], seed=140191)
        _ = bayesian_logistic_regression(df, ['x'], 'pred_y', {'axis': 'y', 'cv': cv, 'compute_gaincurves': False})
        self.assertTrue(cv.meta is None)

    def test__prepare_and_check_classes(self):
        with use_seed(1234567):
            y_bin = np.random.randint(2, size=1000)
            y = 10 * np.exp(np.random.normal(size=1000))
            groups = np.random.randint(10, size=1000)

        np.testing.assert_array_equal(_prepare_and_check_classes(y_bin, None)[:4], [1, 1, 0, 0])
        np.testing.assert_array_equal(_prepare_and_check_classes(y_bin, groups)[:4], ['1_2', '1_4', '0_3', '0_5'])

        np.testing.assert_array_equal(_prepare_and_check_classes(y, None)[:4], [6, 4, 6, 8])
        np.testing.assert_array_equal(_prepare_and_check_classes(y, groups)[:4], ['6_2', '4_4', '6_3', '8_5'])

        np.testing.assert_array_equal(_prepare_and_check_classes(y[:50], None)[:4], [0, 0, 0, 0])
        np.testing.assert_array_equal(_prepare_and_check_classes(y * 0, groups)[:4], ['0_2', '0_4', '0_3', '0_5'])

    def test_get_cv_groups_from_columns(self):
        df = self.df.copy()
        seed = 123

        self.assertTrue(CrossValidationWrapper.get_cv_groups_from_columns(df, cv_groups_col=None) is None)

        np.testing.assert_array_equal(CrossValidationWrapper.get_cv_groups_from_columns(df,
                                                                                        cv_groups_col='i'), df['i'][:])

        self.assertTrue(CrossValidationWrapper.get_cv_groups_from_columns(df, cv_groups_col='z') is None)

        self.assertTrue(CrossValidationWrapper.get_cv_groups_from_columns(df, cv_groups_col=['z', 'a']) is None)

        np.testing.assert_array_equal(CrossValidationWrapper.get_cv_groups_from_columns(df,
                                                                                        cv_groups_col=['i', 'x', 'z'],
                                                                                        seed=seed),
                                      CrossValidationWrapper.get_cv_groups_from_columns(df, cv_groups_col=['i', 'x'],
                                                                                        seed=seed))


class VirtualHStackTestCase(unittest.TestCase):
    def test_init(self):
        hstack = BasicVirtualHStack([np.ones((5, 2)), np.ones((5, 3)), 2, np.int32(3), np.asarray(2, dtype=np.int8)])
        self.assertTrue(hstack.is_block)
        np.testing.assert_array_equal(hstack.X[0], np.ones((5, 2)))
        np.testing.assert_array_equal(hstack.X[1], np.ones((5, 3)))
        np.testing.assert_array_equal(hstack.X[2], np.zeros((5, 2)))
        np.testing.assert_array_equal(hstack.X[3], np.zeros((5, 3)))
        np.testing.assert_array_equal(hstack.X[4], np.zeros((5, 2)))
        np.testing.assert_array_equal(hstack.dims, [0, 2, 5, 7, 10, 12])
        self.assertEqual(hstack.shape, (5, 12))

        hstack_copy = BasicVirtualHStack(hstack)
        self.assertEqual(id(hstack_copy.X), id(hstack.X))
        self.assertEqual(hstack_copy.is_block, hstack.is_block)
        self.assertEqual(hstack_copy.shape, hstack.shape)

        hstack = BasicVirtualHStack(np.zeros((15, 8)))
        self.assertEqual(hstack.shape, (15, 8))

        hstack_copy = BasicVirtualHStack(hstack)
        self.assertEqual(id(hstack_copy.X), id(hstack.X))
        self.assertEqual(hstack_copy.is_block, hstack.is_block)
        self.assertEqual(hstack_copy.shape, hstack.shape)

        with self.assertRaises(ValueError) as e:
            BasicVirtualHStack([])
        self.assertEqual('Cannot create a VirtualHStack of an empty list',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            BasicVirtualHStack([np.ones((5, 2)), 'a'])
        self.assertEqual('Cannot create a VirtualHStack with <type \'str\'>',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            BasicVirtualHStack([np.ones((5, 2)), np.ones((6, 3))])
        self.assertEqual('Cannot create a VirtualHStack for an array of length 6 while other array has length 5',
                         str(e.exception))

    def test_split_by_dims(self):
        hstack = BasicVirtualHStack([np.ones((5, 2)), np.ones((5, 3)), 1])
        self.assertEqual(map(len, hstack.split_by_dims(np.ones(6))), [2, 3, 1])
        with self.assertRaises(AssertionError):
            hstack.split_by_dims(np.ones(7))

    def test_adjust_array_to_total_dimension(self):
        hstack = BasicVirtualHStack([np.ones((5, 2)), np.ones((5, 3)), 1])
        np.testing.assert_array_equal(hstack.adjust_array_to_total_dimension(5), [5] * 6)
        np.testing.assert_array_equal(hstack.adjust_array_to_total_dimension([5] * 3 + [6] * 3), [5] * 3 + [6] * 3)
        np.testing.assert_array_equal(hstack.adjust_array_to_total_dimension([[1, 2], 5, 6]), [1, 2, 5, 5, 5, 6])

        with self.assertRaises(ValueError) as e:
            hstack.adjust_array_to_total_dimension([5] * 5)
        self.assertEqual('parameter is invalid: expected float or an array-like of length 6 or 3, '
                         'got array-like of length 5',
                         str(e.exception))

        hstack = BasicVirtualHStack(np.zeros((15, 8)))
        with self.assertRaises(ValueError) as e:
            hstack.adjust_array_to_total_dimension([5] * 5, 'tt')
        self.assertEqual('parameter \'tt\' is invalid: expected float or an array-like of length 8, '
                         'got array-like of length 5',
                         str(e.exception))

    def test_adjust_to_block_dimensions(self):
        hstack = BasicVirtualHStack([np.ones((5, 2)), np.ones((5, 3)), 1])
        self.assertEqual(map(len, hstack.adjust_to_block_dimensions([[1, 2], 5, 6])), [2, 3, 1])
        with self.assertRaises(ValueError):
            hstack.adjust_to_block_dimensions(np.ones(7))

    def test_promote_types(self):
        hstack = BasicVirtualHStack([np.ones((5, 2), dtype=np.float16),
                                     np.full((5, 1), 3, dtype=np.float32),
                                     np.ones((5, 3), dtype=np.float64),
                                     1])
        self.assertEqual(hstack.X[0].dtype, np.float32)  # dtype is promoted
        self.assertEqual(hstack.X[1].dtype, np.float32)
        self.assertEqual(hstack.X[2].dtype, np.float64)
        self.assertEqual(hstack.X[3].dtype, np.float64)
        self.assertEqual(hstack.materialize().dtype, np.float64)
        np.testing.assert_array_equal(hstack.materialize(), [[1, 1, 3, 1, 1, 1, 0]] * 5)

        hstack = BasicVirtualHStack(np.ones((5, 2), dtype=np.float16))
        self.assertEqual(hstack.X.dtype, np.float32)  # dtype is promoted
        self.assertEqual(hstack.materialize().dtype, np.float32)

    def test_set_parallelism(self):
        hstack = VirtualHStack([np.ones((5, 2), dtype=np.float16),
                                np.full((5, 1), 3, dtype=np.float32),
                                np.ones((5, 3), dtype=np.float64),
                                1], nb_threads=0, nb_inner_threads=4)

        self.assertEqual(hstack.nb_threads, 0)
        self.assertEqual(hstack.nb_inner_threads, 4)
        self.assertTrue(hstack.is_block)
        self.assertTrue(hstack.pool is None)

        hstack = VirtualHStack([np.ones((5, 2), dtype=np.float16)],
                               nb_threads=4, nb_inner_threads=4)

        self.assertEqual(hstack.nb_threads, 1)
        self.assertEqual(hstack.nb_inner_threads, 4)
        self.assertTrue(hstack.is_block)
        self.assertTrue(hstack.pool is None)

        hstack = VirtualHStack([np.ones((5, 2), dtype=np.float16),
                                np.full((5, 1), 3, dtype=np.float32)],
                               nb_threads=4)
        self.assertTrue(hstack.is_block)
        self.assertTrue(hstack.pool is not None)
        self.assertEqual(hstack.nb_threads, 2)
        self.assertEqual(hstack.nb_inner_threads, NB_THREADS_MAX)

    def test_row_nnz(self):
        rand = lambda x: np.random.randint(25, size=(10, x))
        for i in range(20):
            blocks = [
                rand(3),
                KarmaSparse(rand(5)),
                VirtualDirectProduct(KarmaSparse(rand(2)), rand(7)),
                2  # zeros
            ]
            X = VirtualHStack(blocks).materialize()
            row_nnz = np.count_nonzero(X) / 10.
            # we need the hack below to take into account the fact that VirtualDirectProduct is an approximation
            row_nnz_adjusted = row_nnz + (blocks[2].nnz - blocks[2].materialize().nnz) / 10.
            with patch('karma.learning.utils.VirtualDirectProduct.materialize') as mock_materialize:
                np.testing.assert_almost_equal(row_nnz_adjusted, VirtualHStack(blocks).row_nnz)
                np.testing.assert_almost_equal(row_nnz_adjusted,
                                               VirtualHStack(blocks, nb_threads=2, nb_inner_threads=4).row_nnz)
                np.testing.assert_almost_equal(row_nnz, VirtualHStack(X).row_nnz)
            self.assertEqual(0, mock_materialize.call_count)


class VirtualDirectProductTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with use_seed(124567):
            cls.vdp = VirtualDirectProduct(KarmaSparse(np.random.randint(-1, 2, size=(10, 2)),
                                                       shape=(10, 2)),
                                           (2 * np.random.rand(10, 3) - 1).astype(DTYPE))
            cls.vdpt = cls.vdp.T
            cls.dp = cls.vdp.materialize()
            cls.dpt = cls.vdp.materialize().T

    def test_virtual_direct_product_method(self):
        self.assertIsInstance(self.dp, KarmaSparse)
        self.assertFalse(self.vdp.is_transposed)

        self.assertEqual(self.vdp.shape, self.dp.shape)
        self.assertEqual(self.vdp.dtype, self.dp.dtype)
        self.assertEqual(self.vdp.nnz, self.dp.nnz)
        self.assertEqual(self.vdp.ndim, self.dp.ndim)
        self.assertEqual(self.vdp.min(), self.dp.min())
        self.assertEqual(self.vdp.max(), self.dp.max())

        ind = [4, 3, 4, -1]
        assert_array_equal(self.vdp[ind], self.dp[ind])

        assert_almost_equal(self.vdp.sum(), self.dp.sum(), 7)
        assert_almost_equal(self.vdp.sum(axis=0), self.dp.sum(axis=0), 6)
        assert_almost_equal(self.vdp.sum(axis=1), self.dp.sum(axis=1), 6)

        w = np.random.rand(self.vdp.shape[1])
        assert_almost_equal(self.vdp.dot(w), self.dp.dot(w), 6)

        w = np.random.rand(self.vdp.shape[0])
        assert_almost_equal(self.vdp.transpose_dot(w), self.dp.dense_vector_dot_left(w), 6)

        assert_almost_equal(self.vdp.second_moment(), self.dp.T.dot(self.dp), 6)

    def test_virtual_direct_product_transpose_method(self):
        self.assertTrue(self.vdpt.is_transposed)

        self.assertEqual(self.vdpt.shape, self.dpt.shape)
        self.assertEqual(self.vdpt.dtype, self.dpt.dtype)
        self.assertEqual(self.vdpt.nnz, self.dpt.nnz)
        self.assertEqual(self.vdpt.ndim, self.dpt.ndim)

        assert_almost_equal(self.vdp.sum(), self.vdpt.sum(), 5)
        assert_almost_equal(self.vdpt.sum(), self.dpt.sum(), 5)
        assert_almost_equal(self.vdpt.sum(axis=0), self.dpt.sum(axis=0), 6)
        assert_almost_equal(self.vdpt.sum(axis=1), self.dpt.sum(axis=1), 6)

        w = np.random.rand(self.vdpt.shape[1])
        assert_almost_equal(self.vdpt.dot(w), self.dpt.dot(w), 6)

        w = np.random.rand(self.vdpt.shape[0])
        assert_almost_equal(self.vdpt.transpose_dot(w), self.dpt.dense_vector_dot_left(w), 6)

import unittest

import numpy as np
import pandas as pd
from cyperf.matrix.karma_sparse import DTYPE

from labeledmatrix.core.labeledmatrix import LabeledMatrix
from labeledmatrix.core.utils import lm_compute_volume_at_cutoff


class LabeledMatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.mat = np.array([[4, 5, 0, 1, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, 4, 5, 0],
                             [1, 0, 5, 4, 0],
                             [5, 1, 9, 4, 2],
                             [7, 6, 0, 3, 2],
                             [-1, -2, 0, 8, -3]])
        self.lm = LabeledMatrix([np.arange(7).astype(str).tolist(), np.arange(5).astype(str).tolist()], self.mat)
        deco = {str(i): 'deco_{}'.format(i) for i in np.arange(7)}
        self.decorated_lm = self.lm.copy()
        self.decorated_lm.set_deco(row_deco=deco)

    def test_to_flat_dataframe(self):
        dataframe = self.lm.to_flat_dataframe()
        row_index = self.lm.without_zeros().row
        col_index = self.lm.without_zeros().column

        np.testing.assert_equal(len(dataframe), 22)
        np.testing.assert_equal(sorted(dataframe.columns), ['col0', 'col1', 'similarity'])

        np.testing.assert_equal(sorted(set(dataframe['col0'])), sorted(row_index))
        np.testing.assert_equal(sorted(set(dataframe['col1'])), sorted(col_index))
        np.testing.assert_equal(dataframe['similarity'].values[11], 9)

        # check if decoration columns are correctly exported
        dataframe = self.decorated_lm.to_flat_dataframe(deco_row='foo', deco_col='bar')
        np.testing.assert_equal(sorted(dataframe.columns), ['bar', 'col0', 'col1', 'foo', 'similarity'])
        np.testing.assert_equal(set(dataframe['foo']),
                                set(['deco_0', 'deco_2', 'deco_3', 'deco_4', 'deco_5', 'deco_6']))
        np.testing.assert_equal(set(dataframe['bar']), {''})
        np.testing.assert_equal(dataframe['col0'][4], dataframe['foo'][4].split('_')[1])

        dataframe = self.decorated_lm.to_flat_dataframe(deco_row='toto')
        np.testing.assert_equal(sorted(dataframe.columns), ['col0', 'col1', 'similarity', 'toto'])

    def test_rank(self):
        np.testing.assert_array_equal(self.lm.rank(axis=1, reverse=False).matrix, [[3, 4, 0, 2, 1],
                                                                                   [0, 1, 2, 3, 4],
                                                                                   [0, 2, 3, 4, 1],
                                                                                   [2, 0, 4, 3, 1],
                                                                                   [3, 0, 4, 2, 1],
                                                                                   [4, 3, 0, 2, 1],
                                                                                   [2, 1, 3, 4, 0]])

        np.testing.assert_array_equal(self.lm.rank(axis=1, reverse=True).matrix, [[1, 0, 4, 2, 3],
                                                                                  [4, 3, 2, 1, 0],
                                                                                  [4, 2, 1, 0, 3],
                                                                                  [2, 4, 0, 1, 3],
                                                                                  [1, 4, 0, 2, 3],
                                                                                  [0, 1, 4, 2, 3],
                                                                                  [2, 3, 1, 0, 4]])

        np.testing.assert_array_equal(self.lm.rank(axis=0, reverse=False).matrix, [[4, 5, 0, 1, 1],
                                                                                   [1, 1, 1, 0, 2],
                                                                                   [2, 3, 4, 5, 3],
                                                                                   [3, 2, 5, 3, 4],
                                                                                   [5, 4, 6, 4, 5],
                                                                                   [6, 6, 2, 2, 6],
                                                                                   [0, 0, 3, 6, 0]])

        np.testing.assert_array_equal(self.lm.rank(axis=0, reverse=True).matrix, [[2, 1, 6, 5, 5],
                                                                                  [5, 5, 5, 6, 4],
                                                                                  [4, 3, 2, 1, 3],
                                                                                  [3, 4, 1, 3, 2],
                                                                                  [1, 2, 0, 2, 1],
                                                                                  [0, 0, 4, 4, 0],
                                                                                  [6, 6, 3, 0, 6]])

    def test_truncate_by_count(self):
        with self.assertRaises(ValueError) as e:
            _ = self.lm.truncate_by_count(-1, axis=1)
        self.assertEqual('max_rank must be non-negative number or array', str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = self.lm.truncate_by_count(2, axis=5)
        self.assertEqual('axis must be 0, 1 or None', str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = self.lm.truncate_by_count([1, 2, 3, 4, 5], axis=0)
        self.assertEqual('max_rank must be integer or dict', str(e.exception))

        np.testing.assert_array_equal(self.lm.truncate_by_count(0, axis=0).matrix, np.zeros((7, 5)))

        np.testing.assert_array_equal(self.lm.truncate_by_count(2, axis=1).matrix, [[4, 5, 0, 0, 0],
                                                                                    [0, 0, 0, 0, 0],
                                                                                    [0, 0, 4, 5, 0],
                                                                                    [0, 0, 5, 4, 0],
                                                                                    [5, 0, 9, 0, 0],
                                                                                    [7, 6, 0, 0, 0],
                                                                                    [0, 0, 0, 8, 0]])

        np.testing.assert_array_equal(self.lm.truncate_by_count(2, axis=0).matrix, [[0, 5, 0, 0, 0],
                                                                                    [0, 0, 0, 0, 0],
                                                                                    [0, 0, 0, 5, 0],
                                                                                    [0, 0, 5, 0, 0],
                                                                                    [5, 0, 9, 0, 2],
                                                                                    [7, 6, 0, 0, 2],
                                                                                    [0, 0, 0, 8, 0]])

        np.testing.assert_array_equal(self.lm.truncate_by_count(6, axis=None).matrix, [[0, 5, 0, 0, 0],
                                                                                       [0, 0, 0, 0, 0],
                                                                                       [0, 0, 0, 0, 0],
                                                                                       [0, 0, 5, 0, 0],
                                                                                       [0, 0, 9, 0, 0],
                                                                                       [7, 6, 0, 0, 0],
                                                                                       [0, 0, 0, 8, 0]])

        np.testing.assert_array_equal(self.lm.truncate_by_count({'0': 3, '1': 4, '2': 1, '3': 1, '4': 7},
                                                                axis=0).matrix, [[4, 5, 0, 0, 0],
                                                                                 [0, 0, 0, 0, 0],
                                                                                 [0, 1, 0, 0, 0],
                                                                                 [0, 0, 0, 0, 0],
                                                                                 [5, 1, 9, 0, 2],
                                                                                 [7, 6, 0, 0, 2],
                                                                                 [0, 0, 0, 8, -3]])

        np.testing.assert_array_equal(self.lm.truncate_by_count({'0': 2, '1': 1, '2': 2, '3': 1, '4': 3, '5': 6,
                                                                 '6': 2}, axis=1).matrix, [[4, 5, 0, 0, 0],
                                                                                           [0, 0, 0, 0, 0],
                                                                                           [0, 0, 4, 5, 0],
                                                                                           [0, 0, 5, 0, 0],
                                                                                           [5, 0, 9, 4, 0],
                                                                                           [7, 6, 0, 3, 2],
                                                                                           [0, 0, 0, 8, 0]])

    def test_truncate(self):
        lm = LabeledMatrix((['b', 'c'], ['x', 'z', 'y']), np.array([[4, 6, 5], [7, 9, 8]])).sort()
        np.testing.assert_array_equal(lm.to_sparse().truncate(cutoff=6).matrix, np.array([[0., 0., 6.], [7., 8., 9.]]))
        np.testing.assert_array_equal(lm.truncate(cutoff=6).matrix, np.array([[0., 0., 6.], [7., 8., 9.]]))
        np.testing.assert_array_equal(lm.to_sparse().truncate(nb_h=2).matrix, np.array([[0, 5, 6], [0, 8, 9]]))
        np.testing.assert_array_equal(lm.to_sparse().truncate(nb_v=1).matrix, np.array([[0, 0, 0], [7, 8, 9]]))
        np.testing.assert_array_equal(lm.truncate(nb=2).matrix, np.array([[0, 0, 0], [0, 8, 9]]))
        np.testing.assert_array_equal(lm.truncate(nb_h=1).matrix, np.array([[0, 0, 6], [0, 0, 9]]))
        np.testing.assert_array_equal(lm.truncate(nb_v=1).matrix, np.array([[0, 0, 0], [7, 8, 9]]))
        np.testing.assert_array_equal(lm.transpose().truncate(nb_v=1).matrix, np.array([[0, 0], [0, 0], [6, 9]]))

        lm = LabeledMatrix((['b', 'c'], ['x', 'z', 'y']), np.random.rand(2, 3).astype(DTYPE))
        np.testing.assert_array_equal(lm.truncate(nb_v=1).matrix, lm.to_sparse().truncate(nb_v=1).matrix)
        np.testing.assert_array_equal(lm.truncate(nb_h=2).matrix, lm.to_sparse().truncate(nb_h=2).matrix)
        np.testing.assert_array_equal(lm.truncate(nb_h=0).matrix, lm.zeros().matrix)

    def test_rank_dispatch(self):
        matrix = np.array([[10, 1, 3], [2, 5, 3], [5, 6, 6], [1, 3, 5]])
        lm = LabeledMatrix((list(range(4)), ['a', 'b', 'c']), matrix)

        np.testing.assert_array_equal(np.asarray(lm.round_robin_allocation(1).matrix), [[10, 0, 0],
                                                                                        [0, 5, 0],
                                                                                        [0, 6, 0],
                                                                                        [0, 0, 5]])

        np.testing.assert_array_equal(np.asarray(lm.round_robin_allocation(1, 1, 1).matrix), [[10, 0, 0],
                                                                                              [0, 0, 0],
                                                                                              [0, 6, 0],
                                                                                              [0, 0, 0]])

        np.testing.assert_array_equal(np.asarray(
            lm.round_robin_allocation(1, {'a': 3, 'b': 1, 'c': 2}, {'a': 3, 'b': 1, 'c': 2}).matrix),
            [[10, 0, 0],
             [2, 0, 0],
             [0, 6, 0],
             [0, 0, 5]])

        with self.assertRaises(ValueError) as e:
            _ = lm.round_robin_allocation(1, [1, 1, 1], 2)
        self.assertEqual('max_ranks must be integer or dict', str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = lm.round_robin_allocation(1, 2, [2, 2, 2])
        self.assertEqual('max_volumes must be integer or dict', str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = lm.round_robin_allocation(1, 2, {'a': -3, 'b': 1, 'c': 2})
        self.assertEqual('max_volumes must be positive or 0', str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = lm.round_robin_allocation(1, {'a': -3, 'b': 1, 'c': 2}, 2)
        self.assertEqual('max_ranks must be positive or 0', str(e.exception))

    def test_argmax_dispatch(self):
        """Testing argmax dispatch."""
        matrix = np.array([[0.2, 0.5, 0.7], [0.1, 0.4, 0.3], [0.8, 0.3, 0.75], [0.2, 0.7, 0.9]])
        lm = LabeledMatrix((list(range(matrix.shape[0])), ['a', 'b', 'c']), matrix)
        lm_sparse = lm.to_sparse()

        np.testing.assert_array_equal(lm_sparse.argmax_allocation(maximum_pressure=1).matrix,
                                      np.array([[0., 0., 0.7],
                                                [0., 0.4, 0.],
                                                [0.8, 0., 0.],
                                                [0., 0., 0.9]], dtype=DTYPE))

        np.testing.assert_array_equal(lm_sparse.argmax_allocation(maximum_pressure=1, max_volumes=1).matrix,
                                      np.array([[0., 0.5, 0.],
                                                [0., 0., 0.],
                                                [0.8, 0., 0.],
                                                [0., 0., 0.9]], dtype=DTYPE))

        np.testing.assert_array_equal(lm_sparse.argmax_allocation(maximum_pressure=1,
                                                                  max_volumes={'a': 1, 'b': 2, 'c': 1}).matrix,
                                      np.array([[0., 0.5, 0.],
                                                [0., 0.4, 0.],
                                                [0.8, 0., 0.],
                                                [0., 0., 0.9]], dtype=DTYPE))

        np.testing.assert_array_equal(lm_sparse.argmax_allocation(maximum_pressure=4).matrix,
                                      np.array([[0.2, 0.5, 0.7],
                                                [0.1, 0.4, 0.3],
                                                [0.8, 0.3, 0.75],
                                                [0.2, 0.7, 0.9]], dtype=DTYPE))

        self.assertEqual(np.count_nonzero(lm_sparse.argmax_allocation(maximum_pressure=4, max_volumes=0).matrix), 0)
        self.assertEqual(np.count_nonzero(lm_sparse.argmax_allocation(maximum_pressure=4, max_ranks=0).matrix), 0)

        matrix = np.random.rand(1000, 10)
        lm_sparse = LabeledMatrix((list(range(matrix.shape[0])), list(range(matrix.shape[1]))), matrix).to_sparse()
        max_volume, maximum_pressure, max_rank = (300, 4, 100)
        dispatched_lm = lm_sparse.argmax_allocation(maximum_pressure=maximum_pressure,
                                                    max_volumes=max_volume,
                                                    max_ranks=max_rank).nonzero_mask()

        self.assertEqual(dispatched_lm.sum(axis=1).max(axis=None), 4)
        self.assertTrue(dispatched_lm.sum(axis=0).max(axis=None) <= min(max_volume, max_rank))

        with self.assertRaises(ValueError) as e:
            _ = lm_sparse.argmax_allocation(maximum_pressure=1, max_volumes=[2, 2, 2])
        self.assertEqual('max_volumes must be integer or dict', str(e.exception))

    def test_lm_from_pivot(self):
        d = pd.DataFrame()
        d['gender'] = ['1', '1', '2', '2', '1', '2', '1', '3']
        d['revenue'] = [100, 42, 60, 30, 80, 35, 33, 20]
        d['csp'] = ['+', '-', '+', '-', '+', '-', '-', '+']
        lm = LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'mean')
        np.testing.assert_array_almost_equal(lm.matrix.toarray(), [[90, 37.5],
                                                                   [60, 32.5],
                                                                   [20, 0]])

        lm = LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'std')
        np.testing.assert_array_almost_equal(lm.matrix.toarray(), [[10, 4.5],
                                                                   [0, 2.5],
                                                                   [0, 0]])

        lm = LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'std', sparse=False)
        np.testing.assert_array_almost_equal(lm.matrix, [[10, 4.5],
                                                         [0, 2.5],
                                                         [0, 0]])

        with self.assertRaises(ValueError) as e:
            LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'dummyAggregator', sparse=False)
        self.assertEqual('Unknown aggregator `dummyAggregator`', str(e.exception))

    def test_lm_pivot_missing(self):
        d = pd.DataFrame()
        d['gender'] = ['1', '1', '2', '2', '1', '2', '1', '3', '3']
        d['revenue'] = [100, 42, 60, 30, 80, 35, 33, 20, None]
        d['csp'] = ['+', '-', '+', '-', '+', '-', '-', '+', '+']
        lm = LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'mean')
        np.testing.assert_array_almost_equal(lm.matrix.toarray(), [[90, 37.5],
                                                                   [60, 32.5],
                                                                   [20, 0]])

    def test_lm_pivot_dtypes_strategy(self):
        d = pd.DataFrame()
        d['gender'] = ['1', '1', '2', '2', '1', '2', '1', '3', '3']
        d['revenue'] = np.asarray([100, 42, 60, 30, 80, 35, 33, 20, np.nan], np.float32)
        d['csp'] = ['+', '-', '+', '-', '+', '-', '-', '+', '+']
        lm = LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'sum', sparse=False)
        np.testing.assert_array_almost_equal(lm.matrix, [[180., 75.],
                                                         [60., 65.],
                                                         [20., 0.]])
        self.assertEqual(lm.matrix.dtype, np.float32)

        d = pd.DataFrame()
        d['gender'] = ['1', '1', '2', '2', '1', '2', '1', '3', '3']
        d['revenue'] = np.asarray([100, 42, 60, 30, 80, 35, 33, 20, 10], np.int32)
        d['csp'] = ['+', '-', '+', '-', '+', '-', '-', '+', '+']
        lm = LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'sum', sparse=False)
        np.testing.assert_array_almost_equal(lm.matrix, [[180., 75.],
                                                         [60., 65.],
                                                         [30., 0.]])
        self.assertEqual(lm.matrix.dtype, np.int32)

        d = pd.DataFrame()
        d['gender'] = ['1', '1', '2', '2', '1', '2', '1', '3', '3']
        d['revenue'] = np.asarray([100, 42, 60, 30, 80, 35, 33, 20, 10], np.int32)
        d['csp'] = ['+', '-', '+', '-', '+', '-', '-', '+', '+']
        lm = LabeledMatrix.from_pivot(d, 'gender', 'csp', 'revenue', 'mean', sparse=False)
        self.assertEqual(lm.matrix.dtype, np.float64)

    def test_lm_compute_vol_at_cutoff(self):
        # Test parameters initialization
        matrix = np.array([[0.2, 0.1], [0.02, 0.5], [0.1, 0.57], [0.05, 0.4], [0.4, 0.2]])
        row, column = ['user_a', 'user_b', 'user_c', 'user_d', 'user_e'], ['topic_1', 'topic_2']
        lm = LabeledMatrix((row, column), matrix.copy())
        potential_cutoff = 0.8
        expected = {'topic_1': np.float32(0.4), 'topic_2': np.float32(0.4)}

        actual = lm_compute_volume_at_cutoff(lm, potential_cutoff)

        # Assertion
        self.assertDictEqual(actual, expected)

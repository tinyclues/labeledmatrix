#
# Copyright tinyclues, All rights reserved
#

import unittest
import numpy as np

from cyperf.indexing.truncated_join import sorted_unique, SortedDateIndex, PastTruncatedIndex
from cyperf.indexing import ColumnIndex


class TestTruncatedJoin(unittest.TestCase):
    def test_sorted_unique(self):
        a = np.sort(np.arange(90) % 11)
        np.testing.assert_equal(sorted_unique(a), np.unique(a, return_index=True))


class TestSortedDateIndex(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(AssertionError):
            _ = SortedDateIndex(['2017-03-12', '2017-03-11'])

        ii = SortedDateIndex(['2017-03-12', '2017-03-22', '2017-03-22', '2017-03-31', '2017-04-01', '2017-04-01'])
        self.assertEqual(ii.max_date, np.datetime64('2017-04-01'))
        self.assertEqual(ii.min_date, np.datetime64('2017-03-12'))
        self.assertEqual(len(ii.unique_date), 21)
        np.testing.assert_array_equal(ii.interval_indices,
                                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 6])
        self.assertTrue(ii.is_without_holes)

    def test_get_window_indices(self):
        ii = SortedDateIndex(['2017-03-12', '2017-03-22', '2017-03-22', '2017-03-31', '2017-04-01', '2017-04-01'])
        with self.assertRaises(AssertionError):
            _ = ii.get_window_indices('2017-03-28', lower=-1, upper=-2)
        self.assertEqual(ii.get_window_indices('2017-03-28', lower=-6, upper=3), (1, 3))
        self.assertEqual(ii.get_window_indices('2017-03-28', lower=-5, upper=3), (3, 3))
        self.assertEqual(ii.get_window_indices('2017-03-28', lower=-6, upper=4), (1, 4))
        self.assertEqual(ii.get_window_indices('2017-03-28', lower=-100, upper=100), (0, 6))

        self.assertEqual(zip(*ii.get_window_indices(['2017-03-28', '2017-03-29'], lower=-6, upper=3)),
                         [(1, 3), (3, 4)])
        self.assertEqual(zip(*ii.get_window_indices(['2017-03-28', '2017-03-29'], lower=-7, upper=4)),
                         [(1, 4), (1, 6)])

    def test_decay(self):
        ii = SortedDateIndex(['2017-03-12', '2017-03-22', '2017-03-22', '2017-03-31', '2017-04-01', '2017-04-01'])
        row_decay, column_decay = ii.decay(['2017-03-21', '2017-04-11'], 10)
        self.assertEqual(row_decay[0] * column_decay[0],
                         2 ** (-(np.datetime64('2017-03-21') - np.datetime64('2017-03-12')).astype(float) / 10))
        self.assertEqual(row_decay[1] * column_decay[3],
                         2 ** (-(np.datetime64('2017-04-11') - np.datetime64('2017-03-31')).astype(float) / 10))

        row_decay, column_decay = ii.decay(['2017-03-22', '2017-04-11'], 10)
        # decay for 2017-03-12 from 2017-03-22
        self.assertEqual(row_decay[0] * column_decay[0], 0.5)
        # decay for 2017-04-01 from 2017-04-11
        self.assertEqual(row_decay[1] * column_decay[-1], 0.5)


class PastTruncatedIndexTestCase(unittest.TestCase):
    def setUp(self):
        self.uu = ColumnIndex(['u1', 'u2', 'u1', 'u1', 'u2', 'u2'])
        self.dd = SortedDateIndex(['2017-03-12', '2017-03-22', '2017-03-22', '2017-03-31', '2017-04-01', '2017-04-01'])
        self.ii = PastTruncatedIndex(self.uu, self.dd)

    def test_get_batch_window_indices(self):
        indices, indptr = self.ii.get_batch_window_indices(['u1', 'u1', 'u2'],
                                                           ['2017-03-13', '2017-03-23', '2017-04-02'])
        np.testing.assert_array_equal(indices, [0, 2, 4, 5])
        np.testing.assert_array_equal(indptr, [0, 1, 2, 4])

        indices, indptr = self.ii.get_batch_window_indices(['u1', 'u1', 'u2'],
                                                           ['2017-03-13', '2017-03-23', '2017-04-02'],
                                                           lower=0, upper=9)
        np.testing.assert_array_equal(indices, [3])
        np.testing.assert_array_equal(indptr, [0, 0, 1, 1])

        indices, indptr = self.ii.get_batch_window_indices(['u1', 'u1', 'u2'],
                                                           ['2017-03-13', '2017-03-23', '2017-04-02'],
                                                           lower=0, upper=8)
        np.testing.assert_array_equal(indices, [])
        np.testing.assert_array_equal(indptr, [0, 0, 0, 0])

    def test_get_batch_window_indices_with_intensity(self):
        indices, intensities = self.ii.get_batch_window_indices_with_intensity(['u1', 'u1', 'u2'],
                                                                               ['2017-03-13', '2017-03-23',
                                                                                '2017-04-02'], half_life=1)
        np.testing.assert_array_equal(indices, [0, 2, 4, 5])
        np.testing.assert_array_equal(intensities[[0, 1, 2, 2], np.arange(4)], np.full(4, 0.5))

        indices, intensities = self.ii.get_batch_window_indices_with_intensity(['u1', 'u1', 'u2'],
                                                                               ['2017-03-13', '2017-03-23',
                                                                                '2017-04-02'], half_life=10,
                                                                               lower=-14, upper=14)
        self.assertEqual(len(indices), 6)
        self.assertEqual(intensities.nnz, 8)
        np.testing.assert_array_almost_equal(intensities.toarray()[2], [0., 0.4665165, 0., 0., 0.93303299, 0.93303299])

    def test_get_first_batch_window_indices_with_intensities(self):
        indices, intensities = self.ii.get_first_batch_window_indices_with_intensities(['u1', 'u1', 'u2'],
                                                                                       ['2017-03-13', '2017-03-23',
                                                                                        '2017-04-02'], half_life=10,
                                                                                       lower=-14, upper=14)
        self.assertEqual(len(indices), 2)
        self.assertEqual(intensities.nnz, 3)
        np.testing.assert_array_almost_equal(intensities.toarray()[2], [0., 0.4665165])

    def test_get_last_batch_window_indices_with_intensities(self):
        indices, intensities = self.ii.get_last_batch_window_indices_with_intensities(['u1', 'u1', 'u2'],
                                                                                      ['2017-03-13', '2017-03-23',
                                                                                       '2017-04-02'], half_life=10,
                                                                                      lower=-14, upper=14)
        self.assertEqual(len(indices), 3)
        self.assertEqual(intensities.nnz, 3)
        np.testing.assert_array_almost_equal(intensities.toarray()[2], [0., 0., 0.93303299])

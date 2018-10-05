#
# Copyright tinyclues, All rights reserved
#

import unittest
import numpy as np

from cyperf.indexing.truncated_join import (sorted_unique, SortedDateIndex, SortedTruncatedIndex,
                                            LookUpDateIndex, LookUpTruncatedIndex, create_truncated_index,
                                            two_integer_array_deduplication, _merge_ks_struct)
from cyperf.indexing import ColumnIndex


class TestDeduplication(unittest.TestCase):
    def test_two_integer_array_deduplication(self):
        for _ in xrange(100):
            a1 = np.random.randint(-100, np.random.randint(100) + 1, size=10**4)
            a2 = np.random.randint(-200, np.random.randint(100) + 1, size=10**4)
            ind, (u1, u2) = two_integer_array_deduplication(a1, a2)
            self.assertEqual(sorted(set(zip(a1, a2))), sorted(zip(u1, u2)))
            np.testing.assert_equal(u1[ind], a1)
            np.testing.assert_equal(u2[ind], a2)

    def test_merge_ks_struct(self):
        indices, indptr = _merge_ks_struct([([1, 2, 2, 3], [0, 2, 4]), ([1, 0, 0], [0, 1, 2, 3])])
        self.assertEqual(indices.dtype.kind, 'i')
        np.testing.assert_equal(indices, [1, 2, 2, 3, 1, 0, 0])
        np.testing.assert_equal(indptr, [0, 2, 4, 5, 6, 7])

        indices, indptr = _merge_ks_struct([([1, 2, 2, 3], [0, 2, 4]), ([], [0, 0, 0])])
        np.testing.assert_equal(indices, [1, 2, 2, 3])
        np.testing.assert_equal(indptr, [0, 2, 4, 4, 4])
        self.assertEqual(indptr.dtype.kind, 'i')


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
        np.testing.assert_almost_equal(row_decay[0] * column_decay[0],
                                       2 ** (-(np.datetime64('2017-03-21') -
                                               np.datetime64('2017-03-12')).astype(float) / 10))
        np.testing.assert_almost_equal(row_decay[1] * column_decay[3],
                                       2 ** (-(np.datetime64('2017-04-11') -
                                               np.datetime64('2017-03-31')).astype(float) / 10))

        row_decay, column_decay = ii.decay(['2017-03-22', '2017-04-11'], 10)
        # decay for 2017-03-12 from 2017-03-22
        self.assertEqual(row_decay[0] * column_decay[0], 0.5)
        # decay for 2017-04-01 from 2017-04-11
        self.assertEqual(row_decay[1] * column_decay[-1], 0.5)


class TruncatedIndexSortedTestCase(unittest.TestCase):
    def setUp(self):
        self.uu = ColumnIndex(['u1', 'u2', 'u1', 'u1', 'u2', 'u2'])
        dd = ['2017-03-12', '2017-03-22', '2017-03-22', '2017-03-31', '2017-04-01', '2017-04-01']
        self.dd = SortedDateIndex(dd)
        self.ii = SortedTruncatedIndex(self.uu, self.dd)
        self.ii_lookup = LookUpTruncatedIndex(self.uu, LookUpDateIndex(dd))

    def test_get_batch_window_indices_with_nb(self):
        u, d = ['u1', 'u2', 'u1', 'u2'], ['2017-03-23', '2017-04-02', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii\
            .get_batch_window_indices_with_intensity(u, d, lower=-1000, upper=1000)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=-1000, upper=1000)

        np.testing.assert_array_equal(target_indices, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(repeated_indices, [0, 1, 0, 1])
        np.testing.assert_array_equal(ks.indices, [0, 2, 3, 1, 4, 5])
        np.testing.assert_array_equal(ks.indptr, [0, 3, 6])

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)

        target_indices, ks, repeated_indices = self.ii\
            .get_batch_window_indices_with_intensity(u, d, lower=-1000, upper=1000, nb=2)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=-1000, upper=1000, nb=2)

        np.testing.assert_array_equal(target_indices, [2, 3, 4, 5])
        np.testing.assert_array_equal(repeated_indices, [0, 1, 0, 1])
        np.testing.assert_array_equal(ks.indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(ks.indptr, [0, 2, 4])

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)

    def test_get_batch_window_indices0(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii.get_batch_window_indices_with_intensity(u, d)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup.get_batch_window_indices_with_intensity(u, d)

        np.testing.assert_array_equal(target_indices, [0, 2, 4, 5])
        self.assertIsNone(repeated_indices)
        np.testing.assert_array_equal(ks.indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(ks.indptr, [0, 1, 2, 4])

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)

    def test_get_batch_window_indices1(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']

        target_indices, ks, repeated_indices = self.ii.get_batch_window_indices_with_intensity(u, d, lower=0, upper=9)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=0, upper=9)

        np.testing.assert_array_equal(target_indices, [3])
        self.assertIsNone(repeated_indices)
        np.testing.assert_array_equal(ks.indices, [0])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 1, 1])

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)

    def test_get_batch_window_indices2(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']

        target_indices, ks, repeated_indices = self.ii.get_batch_window_indices_with_intensity(u, d, lower=0, upper=8)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=0, upper=8)

        np.testing.assert_array_equal(target_indices, [0])
        self.assertIsNone(repeated_indices)

        np.testing.assert_array_equal(ks.nnz, 0)
        np.testing.assert_array_equal(ks.shape, (3, 1))

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)

    def test_get_batch_window_indices_with_intensity0(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii.get_batch_window_indices_with_intensity(u, d, half_life=1)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, half_life=1)

        np.testing.assert_array_equal(target_indices, [0, 2, 4, 5])
        np.testing.assert_array_equal(ks.indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(ks[[0, 1, 2, 2], np.arange(4)], np.full(4, 0.5))

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)

    def test_get_batch_window_indices_with_intensity1(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii\
            .get_batch_window_indices_with_intensity(u, d, half_life=10,
                                                     lower=-14, upper=14)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, half_life=10,
                                                     lower=-14, upper=14)
        self.assertEqual(len(target_indices_lu), 6)
        self.assertIsNone(repeated_indices)
        self.assertEqual(ks.nnz, 8)
        np.testing.assert_array_almost_equal(ks.toarray()[2], [0., 0.4665165, 0., 0., 0.93303299, 0.93303299])

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)

    def test_get_first_batch_window_indices_with_intensities(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii\
            .get_first_batch_window_indices_with_intensities(u, d, half_life=10, lower=-14, upper=14)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup\
            .get_first_batch_window_indices_with_intensities(u, d, half_life=10, lower=-14, upper=14)

        self.assertEqual(len(target_indices), 2)
        self.assertEqual(ks.nnz, 3)
        np.testing.assert_array_almost_equal(ks.toarray()[2], [0., 0.4665165])

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)

    def test_get_last_batch_window_indices_with_intensities(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']

        target_indices, ks, repeated_indices = self.ii\
            .get_last_batch_window_indices_with_intensities(u, d, half_life=10, lower=-14, upper=14)
        target_indices_lu, ks_lu, repeated_indices_lu = self.ii_lookup\
            .get_last_batch_window_indices_with_intensities(u, d, half_life=10, lower=-14, upper=14)

        np.testing.assert_array_equal(target_indices, [2, 3, 5])
        self.assertEqual(len(target_indices), 3)
        self.assertEqual(ks.nnz, 3)
        np.testing.assert_array_almost_equal(ks.toarray()[2], [0., 0., 0.93303299])

        np.testing.assert_array_equal(ks_lu, ks)
        np.testing.assert_array_equal(target_indices, target_indices_lu)
        np.testing.assert_array_equal(repeated_indices, repeated_indices_lu)


class TruncatedIndexUnsortedTestCase(unittest.TestCase):
    def setUp(self):
        self.uu = ColumnIndex(['u1', 'u2', 'u1', 'u2', 'u1', 'u2'])
        dd = ['2017-03-31', '2017-04-01', '2017-03-12', '2017-03-22', '2017-03-22', '2017-04-01']
        self.ii_lookup = LookUpTruncatedIndex(self.uu, LookUpDateIndex(dd))

    def test_get_batch_window_indices0(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']

        target_indices, ks, repeated_indices = self.ii_lookup.get_batch_window_indices_with_intensity(u, d)
        np.testing.assert_array_equal(target_indices, [1, 2, 4, 5])
        np.testing.assert_array_equal(ks.indices, [1, 2, 0, 3])
        np.testing.assert_array_equal(ks.indptr, [0, 1, 2, 4])
        self.assertIsNone(repeated_indices)

    def test_get_batch_window_indices1(self):
        u, d = ['u1', 'u2', 'u1'], ['2017-03-13', '2017-04-02', '2017-03-23']
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=0, upper=9)

        np.testing.assert_array_equal(ks.indices, [0])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 0, 1])

    def test_get_batch_window_indices2(self):
        u, d = ['u1', 'u2', 'u1'], ['2017-03-13', '2017-04-02', '2017-03-23']
        target_indices, ks, repeated_indices = self.ii_lookup.get_batch_window_indices_with_intensity(u, d, lower=0, upper=8)

        np.testing.assert_array_equal(ks.indices, [])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 0, 0])

    def test_get_batch_window_indices_with_intensity0(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii_lookup.get_batch_window_indices_with_intensity(u, d, half_life=1)

        np.testing.assert_array_equal(target_indices, [1, 2, 4, 5])
        np.testing.assert_array_equal(ks[[0, 1, 2, 2], [1, 2, 3, 3]], 0.5)

    def test_get_batch_window_indices_with_intensity1(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=-14, upper=14, half_life=10)
        self.assertEqual(len(target_indices), 6)
        self.assertEqual(ks.nnz, 8)
        np.testing.assert_array_almost_equal(ks.toarray()[2],
                                             [0.,  0.933033,  0.,  0.466516,  0.,  0.933033])

    def test_get_first_batch_window_indices_with_intensities(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_first_batch_window_indices_with_intensities(u, d, half_life=10, lower=-14, upper=14)

        self.assertEqual(len(target_indices), 2)
        self.assertEqual(ks.nnz, 3)
        np.testing.assert_array_almost_equal(ks.toarray()[2], [0., 0.4665165])

    def test_get_last_batch_window_indices_with_intensities(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']

        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_last_batch_window_indices_with_intensities(u, d, half_life=10, lower=-14, upper=14)

        self.assertEqual(len(target_indices), 3)
        self.assertEqual(ks.nnz, 3)
        np.testing.assert_array_almost_equal(ks.toarray()[2], [0., 0., 0.93303299])


class TruncatedIndexUnsortedDirtyTestCase(unittest.TestCase):
    def setUp(self):
        uu = ColumnIndex(['u1', 'u2', 'u1', 'u1', 'u2', 'u2', 'u1', 'u2'])
        dd = ['2017-03-31', '2017-04-01', '',  '2017-03-12', '2017-03-22', 'NN', '2017-03-22', '2017-04-01']
        self.ii_lookup = create_truncated_index(uu, dd)

    def test_get_batch_window_indices0(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']

        target_indices, ks, repeated_indices = self.ii_lookup.get_batch_window_indices_with_intensity(u, d)

        np.testing.assert_array_equal(target_indices, [1, 3, 6, 7])
        np.testing.assert_array_equal(ks.indptr, [0, 1, 2, 4])
        np.testing.assert_array_equal(ks.data, 1)
        np.testing.assert_array_equal(ks.toarray(), [[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 1.]])
        self.assertIsNone(repeated_indices)

    def test_get_batch_window_indices_dirty(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', None, '2017-04-02']

        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d)

        np.testing.assert_array_equal(target_indices, [1, 3, 7])
        np.testing.assert_array_equal(ks.indices, [1, 0, 2])
        np.testing.assert_array_equal(ks.indptr, [0, 1, 1, 3])
        np.testing.assert_array_equal(ks.data, 1)
        self.assertIsNone(repeated_indices)

    def test_get_batch_window_indices1(self):
        u, d = ['u1', 'u2', 'u1'], ['2017-03-13', '2017-04-02', '2017-03-23']
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=0, upper=9)

        np.testing.assert_array_equal(ks.indices, [0])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 0, 1])
        np.testing.assert_array_equal(ks.data, 1)

    def test_get_batch_window_indices2(self):
        u, d = ['u1', 'u2', 'u1'], ['2017-03-13', '2017-04-02', '2017-03-23']
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=0, upper=8)

        np.testing.assert_array_equal(ks.indices, [])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 0, 0])
        np.testing.assert_array_equal(ks.data, 1)

    def test_get_batch_window_indices_with_intensity0(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii_lookup.get_batch_window_indices_with_intensity(u, d, half_life=1)

        np.testing.assert_array_equal(target_indices, [1, 3, 6, 7])
        np.testing.assert_array_equal(ks[[0, 1, 2, 2], [1, 2, 3, 3]], 0.5)
        np.testing.assert_array_equal(ks.shape, (3, 4))
        self.assertIsNone(repeated_indices)

    def test_get_batch_window_indices_with_dirty_intensity0(self):
        u, d = ['u1', 'u1', 'u2', 'u2'], ['2017-03-13', '2017-03-23', 'Dirty', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii_lookup.get_batch_window_indices_with_intensity(u, d, half_life=1)

        np.testing.assert_array_equal(target_indices, [1, 3, 6, 7])
        np.testing.assert_array_equal(ks.indices, [1, 2, 0, 3])
        np.testing.assert_array_equal(ks[[0, 1, 3, 3], [1, 2, 3, 3]], 0.5)
        self.assertIsNone(repeated_indices)

    def test_get_batch_window_indices_with_intensity1(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, half_life=10,
                                                     lower=-14, upper=14)
        self.assertEqual(len(target_indices), 6)
        self.assertEqual(ks.nnz, 8)
        np.testing.assert_array_almost_equal(ks.toarray()[2],
                                             [0.,  0.933033,  0.,  0.466516,  0.,  0.933033])
        self.assertIsNone(repeated_indices)

    def test_get_first_batch_window_indices_with_intensities(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_first_batch_window_indices_with_intensities(u, d, half_life=10,
                                                             lower=-14, upper=14)

        self.assertEqual(len(target_indices), 2)
        self.assertEqual(ks.nnz, 3)
        np.testing.assert_array_almost_equal(ks.toarray()[2], [0., 0.4665165])
        self.assertIsNone(repeated_indices)

    def test_get_last_batch_window_indices_with_intensities(self):
        u, d = ['u1', 'u1', 'u2'], ['2017-03-13', '2017-03-23', '2017-04-02']

        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_last_batch_window_indices_with_intensities(u, d, half_life=10, lower=-14, upper=14)

        self.assertEqual(len(target_indices), 3)
        self.assertEqual(ks.nnz, 3)
        np.testing.assert_array_almost_equal(ks.toarray()[2], [0., 0., 0.93303299])
        self.assertIsNone(repeated_indices)


class TruncatedIndexSortedDirtyTestCase2(unittest.TestCase):

    def setUp(self):
        uu = ColumnIndex(['u1', 'u2', 'u3', 'u1', 'u2'])
        dd = ['2017-03-31', '2017-04-01', '2017-04-01', '2017-04-12', '2017-04-22']
        self.ii_sorted = create_truncated_index(uu, dd)

    def test_dirty_get_case1(self):
        target_indices, ks, repeated_indices = self.ii_sorted\
            .get_batch_window_indices_with_intensity(['u1', 'u2', 'u3'],
                                                     [None, '2017-04-22', None], lower=0, upper=1)

        np.testing.assert_array_equal(target_indices, [4])
        self.assertIsNone(repeated_indices)
        np.testing.assert_array_equal(ks.indices, [0])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 1, 1])
        np.testing.assert_array_equal(ks.data, 1)

        target_indices, ks, repeated_indices = self.ii_sorted\
            .get_batch_window_indices_with_intensity(['u1', 'u2', 'u3'],
                                                     [None, '2017-04-22', None],
                                                     lower=-100, upper=0)
        np.testing.assert_array_equal(target_indices, [1])
        self.assertIsNone(repeated_indices)
        np.testing.assert_array_equal(ks.indices, [0])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 1, 1])
        np.testing.assert_array_equal(ks.data, 1)

    def test_dirty_get_case_all_dirty(self):
        target_indices, ks, repeated_indices = self.ii_sorted\
            .get_batch_window_indices_with_intensity(['u1', 'u2', 'u3'],
                                                     [None, '', None],
                                                     lower=0, upper=1)
        np.testing.assert_array_equal(ks.data, [])
        np.testing.assert_array_equal(ks.indices, [])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 0, 0])

    def test_dirty_repeated_values1(self):
        u, d = [''] * 4, [None] * 4
        target_indices, ks, repeated_indices = self.ii_sorted\
            .get_batch_window_indices_with_intensity(u, d, lower=-100, upper=1000)

        np.testing.assert_array_equal(ks.nnz, 0)
        np.testing.assert_array_equal(target_indices, [0])
        np.testing.assert_array_equal(repeated_indices, [0] * 4)

        # composition test
        out = np.arange(10).reshape(5, 2)
        np.testing.assert_array_equal(ks.dot(out[target_indices])[repeated_indices], 0)

    def test_dirty_repeated_values2(self):
        u, d = ['u1', 'u2', 'u1', 'u2'], ['2017-04-01', '', '2017-04-01', '']
        target_indices, ks, repeated_indices = self.ii_sorted\
            .get_batch_window_indices_with_intensity(u, d, lower=-100, upper=1000)

        np.testing.assert_array_equal(target_indices, [0, 3])
        np.testing.assert_array_equal(repeated_indices, [0, 1, 0, 1])
        np.testing.assert_array_equal(ks.indices, [0, 1])
        np.testing.assert_array_equal(ks.indptr, [0, 2, 2])
        np.testing.assert_array_equal(ks.shape, (2, 2))
        np.testing.assert_array_equal(ks.toarray(), [[1, 1], [0, 0]])

        # composition test
        out = np.arange(10).reshape(5, 2)
        np.testing.assert_array_equal(ks.dot(out[target_indices])[repeated_indices],
                                      [[6., 8.], [0., 0.], [6., 8.], [0., 0.]])


class TruncatedIndexUnsortedDirtyTestCase2(unittest.TestCase):

    def setUp(self):
        uu = ColumnIndex(['u1', 'u2', 'u3', 'u1', 'u2', 'u1'])
        dd = ['2017-03-31', '2017-04-22', '', '2017-04-12', '2017-04-01', '']
        self.ii_lookup = create_truncated_index(uu, dd)

    def test_dirty_get_case1_all_decay(self):
        target_indices, ks, repeated_indices = self.ii_lookup\
            .ks_get_batch_window_indices(['u1', 'u2'], ['', 'Nn'], lower=-1000, upper=1000,
                                         truncation='all', half_life=2)

        np.testing.assert_array_equal(target_indices, [0])
        np.testing.assert_array_equal(ks.shape, (2, 1))
        np.testing.assert_array_equal(ks.nnz, 0)
        self.assertIsNone(repeated_indices)

    def test_dirty_get_case1_all(self):
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(['u1'], [None], lower=-1000, upper=1000)

        np.testing.assert_array_equal(ks.indices, [])
        np.testing.assert_array_equal(ks.indptr, [0, 0])

        # no modification on data
        np.testing.assert_array_equal(self.ii_lookup.date_index.date_values,
                                      np.array(['2017-03-31', '2017-04-22', 'NaT',
                                                '2017-04-12', '2017-04-01', 'NaT'], dtype='M'))

    def test_dirty_get_case1_first(self):
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_first_batch_window_indices_with_intensities(['u1'], [None], lower=-1000, upper=1000)
        np.testing.assert_array_equal(target_indices, [0])
        np.testing.assert_array_equal(ks.indices, [])
        np.testing.assert_array_equal(ks.indptr, [0, 0])
        self.assertIsNone(repeated_indices)

    def test_dirty_get_case1_last(self):
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_last_batch_window_indices_with_intensities(['u1'], [None], lower=-1000, upper=1000)
        np.testing.assert_array_equal(target_indices, [0])
        np.testing.assert_array_equal(ks.indices, [])
        np.testing.assert_array_equal(ks.indptr, [0, 0])

    def test_dirty_get_case2_all(self):
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(['u3', 'u1'], ['2017-04-01', '2017-04-01'],
                                                     lower=10, upper=1000,)

        np.testing.assert_array_equal(target_indices, [3])
        np.testing.assert_array_equal(ks.indices, [0])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 1])
        self.assertIsNone(repeated_indices)

    def test_dirty_get_case2_first(self):
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_first_batch_window_indices_with_intensities(['u2', 'u1'], [None, '2017-04-01'],
                                                             lower=-100, upper=1000)

        np.testing.assert_array_equal(target_indices, [0])
        np.testing.assert_array_equal(ks.indices, [0])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 1])
        self.assertIsNone(repeated_indices)

    def test_dirty_get_case2_last(self):
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_last_batch_window_indices_with_intensities(['u2', 'u1'], [None, '2017-04-01'],
                                                            lower=-100, upper=1000)

        np.testing.assert_array_equal(target_indices, [3])
        np.testing.assert_array_equal(ks.indices, [0])
        np.testing.assert_array_equal(ks.indptr, [0, 0, 1])
        self.assertIsNone(repeated_indices)

    def test_dirty_repeated_values(self):
        u, d = [''] * 4, [None] * 4
        target_indices, ks, repeated_indices = self.ii_lookup\
            .get_batch_window_indices_with_intensity(u, d, lower=-100, upper=1000)

        np.testing.assert_array_equal(target_indices, [0])
        np.testing.assert_array_equal(ks.nnz, 0)
        np.testing.assert_array_equal(repeated_indices, [0, 0, 0, 0])

        u, d = ['u1', 'u2', 'u1', 'u2'], ['2017-04-01', '', '2017-04-01', '']
        target_indices, ks, repeated_indices = self\
            .ii_lookup.get_batch_window_indices_with_intensity(u, d, lower=-100, upper=1000)
        np.testing.assert_array_equal(target_indices, [0, 3])
        np.testing.assert_array_equal(ks.indices, [0, 1])
        np.testing.assert_array_equal(ks.indptr, [0, 2, 2])
        np.testing.assert_array_equal(repeated_indices, [0, 1, 0, 1])

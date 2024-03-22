
import numpy as np
from numpy.testing import assert_equal
import unittest

from cyperf.indexing.column_index import groupsort_indexer_as_parallel_argsort, groupsort_indexer, merge_sort
from cyperf.indexing import unique_indices, factorize


class ColumnIndexCase(unittest.TestCase):

    def check_groupsort_indexer_one_array(self, a):
        aa = a.copy()

        indptr1 = np.zeros(a.max() + 2, dtype=np.int64)
        nb_unique1, indices1 = groupsort_indexer(indptr1, a)
        assert_equal(a, aa)

        indptr2 = np.zeros(a.max() + 2, dtype=np.int64)
        nb_unique2, indices2 = groupsort_indexer_as_parallel_argsort(indptr2, a)
        assert_equal(a, aa)

        assert_equal(indptr1, indptr2)
        assert_equal(indices1, indices2)
        assert_equal(nb_unique1, nb_unique2)

        u, rev = np.unique(a, return_inverse=True)
        assert_equal(len(u), nb_unique2)
        assert_equal(np.argsort(rev, kind="merge"), indices1)

    def test_groupsort_indexer(self):
        for _ in range(20):
            a = np.random.randint(0, np.random.randint(1, 1000), np.random.randint(1, 1000))
            self.check_groupsort_indexer_one_array(a)

        # test same values
        self.check_groupsort_indexer_one_array(np.zeros(10, dtype=np.int64))

    def test_merge_sort(self):
        for _ in range(10):
            a = np.unique(np.random.randint(-90, 50, np.random.randint(10)))
            b = np.unique(np.random.randint(-90, 50, np.random.randint(10)))

            ab = merge_sort(a, b)
            ba = merge_sort(b, a)

            expected = np.unique(np.concatenate([a, b]))

            assert_equal(ab, expected)
            assert_equal(ba, expected)

    def test_unique_indices(self):
        assert_equal([0, 1, 5], unique_indices(['a', 'b', 'a', 'a', 'b', 'c'], True))
        assert_equal([3, 4, 5], unique_indices(['a', 'b', 'a', 'a', 'b', 'c'], False))
        assert_equal([3, 4, 5], unique_indices(['a', 'b', 'a', 'a', 'b', 'c'], False))
        assert_equal([1, 4, 5, 6], unique_indices(np.array([3, 3, 2, 0, 2, 1, 0]), False))
        assert_equal([0, 2, 3, 5], unique_indices(tuple([3, 3, 2, 0, 2, 1, 0]), True))

    def test_factorize(self):
        position, reversed_indices, n_keys = factorize(['b', 'a', 'a', 'b', 'c', 'c', 'c', 'b'])
        assert_equal([0, 1, 1, 0, 2, 2, 2, 0], reversed_indices)
        assert_equal(3, n_keys)
        self.assertEqual(position, {'a': 1, 'b': 0, 'c': 2})

        position, reversed_indices, n_keys = factorize([1, 0, -1, 4, 4, 1, 0])
        assert_equal([0, 1, 2, 3, 3, 0, 1], reversed_indices)
        assert_equal(4, n_keys)
        self.assertEqual(position, {1: 0, 0: 1, -1: 2, 4: 3})

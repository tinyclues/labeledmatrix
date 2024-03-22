#
# Copyright tinyclues, All rights reserved
#

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from cyperf.indexing.indexed_list import inplace_reversed_index, reversed_index


class TestReversedIndex(unittest.TestCase):
    def test_reversed_index(self):
        unique_values_1, indices_1 = reversed_index([4, 'a'])

        assert_array_equal(unique_values_1, [4, 'a'])
        assert_array_equal(indices_1, [0, 1])

        unique_values_2, indices_2 = reversed_index(['b', 'a', 'b'])

        assert_array_equal(unique_values_2, ['b', 'a'])
        assert_array_equal(indices_2, [0, 1, 0])

        unique_values_3, indices_3 = reversed_index([])

        assert_array_equal(unique_values_3, [])
        assert_array_equal(indices_3, [])

        unique_values_4, indices_4 = reversed_index(np.arange(10))

        assert_array_equal(unique_values_4, np.arange(10))
        assert_array_equal(indices_4, np.arange(10))

    def test_inplace_reversed_index(self):
        first_values = [4, 4, 'a', 'b', 'a', 2]
        second_values = [5, 2, 'C', 'b', 3.3, 2]
        indices = np.zeros(len(first_values) + len(second_values), dtype=np.int64)
        position = {}
        unique_values = []

        unique_values_1, indices_1 = reversed_index(first_values + second_values)
        inplace_reversed_index(first_values, indices, position, unique_values)
        inplace_reversed_index(second_values, indices[len(first_values):], position, unique_values)

        expected_indices = [0, 0, 1, 2, 1, 3, 4, 3, 5, 2, 6, 3]
        expected_unique_values = [4, 'a', 'b', 2, 5, 'C', 3.3]

        assert_array_equal(unique_values, unique_values_1)
        assert_array_equal(indices, indices_1)

        assert_array_equal(indices, expected_indices)
        assert_array_equal(unique_values, expected_unique_values)

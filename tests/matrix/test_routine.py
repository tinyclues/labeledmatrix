import unittest
from bisect import bisect_left as bisect_left_old

import numpy as np
from cyperf.matrix.routine import kronii, bisect_left
from cyperf.matrix.karma_sparse import KarmaSparse, sp


class RoutineTestCase(unittest.TestCase):

    def test_kronii(self):
        x, y = np.array([[1, 10, 3]]), np.array([[5, 6], [0, 1]])

        with self.assertRaises(ValueError) as e:
            _ = kronii(x, y)
        self.assertEqual('operands could not be broadcast together with shape'
                         '{} and {}.'.format(x.shape, y.shape),
                         e.exception.message)

        x, y = np.array([[1, 10, 3], [2, -2, 5]]), np.array([[5, 6], [0, 1]])
        xx = KarmaSparse(x)
        yy = KarmaSparse(y)

        result1 = kronii(x, y)
        result2 = xx.kronii(y)
        result3 = xx.kronii(yy)

        np.testing.assert_array_almost_equal(result1,
                                             np.array([[5, 6, 50, 60, 15, 18],
                                                       [0, 2, 0, -2, 0, 5]]))
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def test_kronii_random(self):
        for _ in xrange(30):
            nrows = np.random.randint(1, 100)
            xx = KarmaSparse(sp.rand(nrows, np.random.randint(1, 100), np.random.rand()))
            yy = KarmaSparse(sp.rand(nrows, np.random.randint(1, 100), np.random.rand()))

            result1 = kronii(xx.toarray(), yy.toarray())
            result2 = xx.kronii(yy.toarray())
            result3 = xx.kronii(yy)

            np.testing.assert_array_almost_equal(result1, result2)
            np.testing.assert_array_almost_equal(result2, result3)
            self.assertAlmostEqual(result1[-1, -1], xx[-1, -1] * yy[-1, -1])

    def test_bisect_left(self):
        for _ in xrange(5):
            a_unique = np.unique(np.random.rand(1000))
            a_int = np.random.randint(0, 100, 1000)
            a_int_sorted = np.sort(np.random.randint(0, 100, 1000))

            for _ in xrange(5):
                x_unique = np.random.rand(1)
                x_int = np.random.randint(0, 100, 1)
                x_array = np.random.randint(0, 100, 100)
                for a, x in [(a_unique, x_unique), (a_int, x_int), (a_int_sorted, x_int)]:
                    self.assertEqual(bisect_left(a, x), bisect_left_old(a, x))
                    self.assertEqual(bisect_left(a, x), a.searchsorted(x))
                np.testing.assert_array_equal(bisect_left(a_int_sorted, x_array), a_int_sorted.searchsorted(x_array))

        self.assertEqual(bisect_left([1, 2, 7, 12], [0]).tolist(), [0])
        self.assertEqual(bisect_left([1, 2, 7, 12], [2]).tolist(), [1])
        self.assertEqual(bisect_left([1, 2, 7, 12], [7]).tolist(), [2])
        self.assertEqual(bisect_left([1, 2, 7, 12], [10]).tolist(), [3])
        self.assertEqual(bisect_left([1, 2, 7, 12], [1, 3, 12, 14]).tolist(), [0, 2, 3, 4])

import unittest
import numpy as np
from cyperf.matrix.routine import kronii


class RoutineTestCase(unittest.TestCase):

    def test_kronii(self):
        x, y = np.array([[1, 10, 3]]), np.array([[5, 6], [0, 1]])
        with self.assertRaises(ValueError) as e:
            result = kronii(x, y)
        self.assertEqual('operands could not be broadcast together with shape'
                         '{} and {}.'.format(x.shape, y.shape),
                         e.exception.message)

        x, y = np.array([[1, 10, 3], [2, -2, 5]]), np.array([[5, 6], [0, 1]])
        result = kronii(x, y)
        np.testing.assert_array_almost_equal(result,
                                             np.array([[5, 6, 50, 60, 15, 18],
                                                       [0, 2, 0, -2, 0, 5]]))

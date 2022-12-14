
from itertools import product
from bisect import bisect_left as bisect_left_old
import numpy as np
import unittest

from cyperf.tools import slice_length, compose_slices, take_indices
from cyperf.tools.getter import apply_python_dict
from cyperf.tools.sort_tools import (bisect_left, cython_argpartition, _inplace_permutation, cython_argsort)
from cyperf.tools.vector import int32Vector, float32Vector, int64Vector, float64Vector


class GetterTestCase(unittest.TestCase):

    def test_vector_int32(self):
        for v in [int32Vector(), int64Vector(), float32Vector(), float64Vector()]:

            self.assertFalse(v.exported)
            v.push_back(-4.)
            self.assertFalse(v.exported)

            v.extend(np.arange(4))
            arr = np.asarray(v)
            self.assertTrue(v.exported)
            self.assertEquals(arr.dtype, getattr(np, v.__class__.__name__.split('Vector')[0]))
            np.testing.assert_equal(arr, [-4, 0, 1, 2, 3])

            with self.assertRaises(RuntimeError):
                np.asarray(v)

            with self.assertRaises(RuntimeError):
                v.push_back(5)

            with self.assertRaises(RuntimeError):
                v.extend(np.arange(4))

            arr[2] = -12
            self.assertEquals(arr[0], -4)
            self.assertEquals(arr[-1], 3)
            self.assertEquals(arr[2], -12)

    def test_argpartition(self):
        for _ in range(100):
            x = np.random.rand(np.random.randint(2, 1000))
            size = np.random.randint(1, len(x))
            res = cython_argpartition(x, size, False)
            np_res = np.argpartition(x, size)
            self.assertEqual(len(np.intersect1d(res[:size], np_res[:size], assume_unique=True)),
                             size)

            res = cython_argpartition(x, size, True)
            np_res = np.argpartition(-x, size)
            self.assertEqual(len(np.intersect1d(res[:size], np_res[:size], assume_unique=True)),
                             size)

    def test_inplace_permutation(self):
        for _ in range(1000):
            a = np.random.rand(np.random.randint(1, 1000))
            aa = a.copy()
            b = np.arange(len(a))
            np.random.shuffle(b)
            bb = b.copy()
            _inplace_permutation(a, b)
            self.assertTrue(np.all(a[bb] == aa))

    def test_apply_python_dict(self):
        mapping = {'X': 1, 'y': 4}
        iterable = ['X', 3, [], 'y', 'y']

        self.assertEqual(apply_python_dict(mapping, iterable, -1, False), [1, -1, -1, 4, 4])
        self.assertEqual(apply_python_dict(mapping, tuple(iterable), -1, False), [1, -1, -1, 4, 4])
        self.assertEqual(apply_python_dict(mapping, [], -1, False), [])
        with self.assertRaises(AssertionError):
            apply_python_dict(mapping, 5, -1, False)

        self.assertEqual(apply_python_dict(mapping, iterable, -1, True), [1, 3, [], 4, 4])
        self.assertEqual(apply_python_dict(mapping, tuple(iterable), -1, True), [1, 3, [], 4, 4])
        self.assertEqual(apply_python_dict(mapping, [], -1, True), [])
        with self.assertRaises(AssertionError):
            apply_python_dict(mapping, int, -1, True)

    def test_argsort(self):
        for _ in range(10):
            x = np.random.randn(50)
            y = np.argsort(x)
            np.testing.assert_equal(y, cython_argsort(x, x.shape[0], False))
            np.testing.assert_equal(y[:5], cython_argsort(x, 5, False)[:5])
            np.testing.assert_equal(y[::-1], cython_argsort(x, x.shape[0], True))
            np.testing.assert_equal(y[::-1][:5], cython_argsort(x, 5, True)[:5])

        x = np.arange(20)
        np.random.shuffle(x)
        np.testing.assert_equal(x[cython_argsort(x, x.shape[0], False)], np.arange(20))

    def test_argsort_repeated_values(self):
        y = [np.array([1, 0, 1, 1]),
             np.array([2, 3, 0, 2, 3]),
             np.array([1, 0, 1, 3, 1]),
             np.array([0, 2, 0, 0, 3, 1, 2, 0, 1, 1]),
             np.array([3, 1, 3, 2, 1, 2, 2, 0, 1, 1, 3, 3, 2, 1, 3, 2, 2, 2, 1, 1])]
        for x in y:
            np.testing.assert_equal(x[cython_argsort(x, x.shape[0], False)], np.sort(x))
            np.testing.assert_equal(x[cython_argsort(x, x.shape[0], True)], np.sort(x)[::-1])

        for _ in range(1000):
            x = np.random.randint(0, 6, size=np.random.randint(4, 20))
            np.testing.assert_equal(x[cython_argsort(x, x.shape[0], False)], np.sort(x))
            np.testing.assert_equal(x[cython_argsort(x, x.shape[0], True)], np.sort(x)[::-1])

        x = np.ones(20)
        x[10:] = 2.
        y = x.copy()
        np.random.shuffle(x)
        np.testing.assert_equal(x[cython_argsort(x, x.shape[0], False)], y)

    def test_compose_slices(self):
        arr = np.arange(20, dtype=np.uint8)
        length = len(arr)

        indices = [None, -31, -20, -19, 0, 8, 19]
        steps = [None, -20, -10, -3, -1, 1, 7]

        for (a, b, c, x, y, z) in product(indices, indices, steps, indices, indices, steps):
            slice1, slice2 = slice(a, b, c), slice(x, y, z)

            sub_arr = arr[slice1]
            self.assertEqual(slice_length(slice1, length), len(sub_arr))

            combined = compose_slices(slice1, slice2, length)
            self.assertTrue(np.all(sub_arr[slice2] == arr[combined]))

    def test_take_on_slice(self):
        arr = np.arange(20, dtype=np.uint8)
        length = len(arr)

        indices = [None, -31, -20, -19, 0, 8, 19]
        steps = [None, -20, -10, -3, -1, 1, 7]
        for (a, b, c) in product(indices, indices, steps):
            slice1 = slice(a, b, c)
            inner_length = slice_length(slice1, length)
            if inner_length > 0:
                indices = np.random.randint(-inner_length + 1, inner_length, size=10)
                np.testing.assert_equal(arr[slice1][indices], arr[take_indices(slice1, indices, length)])

        selection_int64_res = take_indices(slice(2 ** 31, 2 ** 31 + 10000), [0, 1, 2], 2 ** 31 + 10000)
        self.assertEqual(np.dtype('int64'), selection_int64_res.dtype)
        np.testing.assert_array_equal([2 ** 31, 2 ** 31 + 1, 2 ** 31 + 2], selection_int64_res)

        np.testing.assert_array_equal(take_indices(slice(2150000000, 2160000000, None), [0], length=3782366988),
                                      [2150000000])

    def test_bisect_left(self):
        for _ in range(5):
            a_unique = np.unique(np.random.rand(1000))
            a_int = np.random.randint(0, 100, 1000)
            a_int_sorted = np.sort(np.random.randint(0, 100, 1000))

            for _ in range(5):
                x_unique = np.random.rand(1)
                x_int = np.random.randint(0, 100, 1)
                x_array = np.random.randint(0, 100, 100)
                for a, x in [(a_unique, x_unique), (a_int, x_int), (a_int_sorted, x_int)]:
                    self.assertEqual(bisect_left(a, x), bisect_left_old(a, x))
                    self.assertEqual(bisect_left(a, x), a.searchsorted(x))
                np.testing.assert_array_equal(bisect_left(a_int_sorted, x_array), a_int_sorted.searchsorted(x_array))

        np.testing.assert_array_equal(bisect_left([1, 2, 7, 12], [0]), [0])
        np.testing.assert_array_equal(bisect_left(np.array([1, 2, 7, 12]), [0]), [0])
        np.testing.assert_array_equal(bisect_left(np.array([1., 2., 7, 12]), [0]), [0])
        np.testing.assert_array_equal(bisect_left([1., 2., 7, 12], np.array([0])), [0])
        np.testing.assert_array_equal(bisect_left([1, 2, 7, 12], [2]), [1])
        np.testing.assert_array_equal(bisect_left([1, 2, 7, 12], [7]), [2])
        np.testing.assert_array_equal(bisect_left([1, 2, 7, 12], [10]), [3])
        np.testing.assert_array_equal(bisect_left([1, 2, 7, 12], [1, 3, 12, 14]), [0, 2, 3, 4])

        np.testing.assert_array_equal(bisect_left(np.array(['1', '2', '77']), ['1', '12', '3']), [0, 1, 2])
        np.testing.assert_array_equal(bisect_left([1, 2, 7, 12], [1, 3, 12, 14]), [0, 2, 3, 4])

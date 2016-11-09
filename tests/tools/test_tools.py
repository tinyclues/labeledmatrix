# import unittest
# import doctest
# import numpy as np
# from itertools import product
# from karma.types import Error
# from numpy import allclose as eq
# from karma.core.perf.common import (logistic, logit, argsort, slice_length, compose_slices,
#                                     take_indices, apply_python_dict, python_feature_hasher)
# from karma.core.perf.common import METRIC_LIST, fast_buddies
# from karma.core.perf.common import vector_distance, pairwise_square, pairwise_flat
# from scipy.spatial.distance import cdist, pdist
# from karma.core.perf.common.sort_tools import cython_argpartition, _inplace_permutation


# class GetterTestCase(unittest.TestCase):

#     def test_argpartition(self):
#         for _ in xrange(100):
#             x = np.random.rand(np.random.randint(2, 1000))
#             size = np.random.randint(1, len(x))
#             res = cython_argpartition(x, size, False)
#             np_res = np.argpartition(x, size)
#             self.assertEqual(len(np.intersect1d(res[:size], np_res[:size], assume_unique=True)),
#                              size)

#             res = cython_argpartition(x, size, True)
#             np_res = np.argpartition(-x, size)
#             self.assertEqual(len(np.intersect1d(res[:size], np_res[:size], assume_unique=True)),
#                              size)

#     def test_inplace_permutation(self):
#         for _ in xrange(1000):
#             a = np.random.rand(np.random.randint(1, 1000))
#             aa = a.copy()
#             b = np.arange(len(a))
#             np.random.shuffle(b)
#             bb = b.copy()
#             _inplace_permutation(a, b)
#             self.assertTrue(np.all(a[bb] == aa))

#     def test_apply_python_dict(self):
#         mapping = {'X': 1, 'y': 4}
#         iterable = ['X', 3, [], 'y', 'y']

#         self.assertEqual(apply_python_dict(mapping, iterable, -1, False), [1, -1, -1, 4, 4])
#         self.assertEqual(apply_python_dict(mapping, tuple(iterable), -1, False), [1, -1, -1, 4, 4])
#         self.assertEqual(apply_python_dict(mapping, [], -1, False), [])
#         with self.assertRaises(AssertionError):
#             apply_python_dict(mapping, 5, -1, False)

#         self.assertEqual(apply_python_dict(mapping, iterable, -1, True), [1, 3, [], 4, 4])
#         self.assertEqual(apply_python_dict(mapping, tuple(iterable), -1, True), [1, 3, [], 4, 4])
#         self.assertEqual(apply_python_dict(mapping, [], -1, True), [])
#         with self.assertRaises(AssertionError):
#             apply_python_dict(mapping, int, -1, True)

#     def test_python_feature_hasher(self):
#         features = [1, 'toto', (1, 4), Error]
#         result = python_feature_hasher(features, 2**10)
#         self.assertEqual(result.tolist(), [1, 684, 846, 687])
    # def test_argsort(self):
    #     for _ in xrange(10):
    #         x = np.random.randn(50)
    #         y = np.argsort(x)
    #         self.assertTrue(eq(y, argsort(x, x.shape[0], False)))
    #         self.assertTrue(eq(y[:5], argsort(x, 5, False)[:5]))
    #         self.assertTrue(eq(y[::-1], argsort(x, x.shape[0], True)))
    #         self.assertTrue(eq(y[::-1][:5], argsort(x, 5, True)[:5]))

    #     x = np.arange(20)
    #     np.random.shuffle(x)
    #     self.assertTrue(eq(x[argsort(x, x.shape[0], False)], np.arange(20)))

    # def test_argsort_repeated_values(self):
    #     y = [np.array([1, 0, 1, 1]),
    #          np.array([2, 3, 0, 2, 3]),
    #          np.array([1, 0, 1, 3, 1]),
    #          np.array([0, 2, 0, 0, 3, 1, 2, 0, 1, 1]),
    #          np.array([3, 1, 3, 2, 1, 2, 2, 0, 1, 1, 3, 3, 2, 1, 3, 2, 2, 2, 1, 1])]
    #     for x in y:
    #         self.assertTrue(eq(x[argsort(x, x.shape[0], False)], np.sort(x)))
    #         self.assertTrue(eq(x[argsort(x, x.shape[0], True)], np.sort(x)[::-1]))

    #     for _ in xrange(1000):
    #         x = np.random.randint(0, 6, size=np.random.randint(4, 20))
    #         self.assertTrue(eq(x[argsort(x, x.shape[0], False)], np.sort(x)))
    #         self.assertTrue(eq(x[argsort(x, x.shape[0], True)], np.sort(x)[::-1]))

    #     x = np.ones(20)
    #     x[10:] = 2.
    #     y = x.copy()
    #     np.random.shuffle(x)
    #     self.assertTrue(eq(x[argsort(x, x.shape[0], False)], y))

    # def test_logistic(self):
    #     self.assertTrue(eq(logit(3, 0, 1), logistic(3, 0, 1)))
    #     self.assertTrue(eq(logit(3, -5, 2), logistic(3, -5, 2)))

    # def test_compose_slices(self):
    #     arr = np.arange(20, dtype=np.uint8)
    #     length = len(arr)

    #     indices = [None, -31, -20, -19, 0, 8, 19]
    #     steps = [None, -20, -10, -3, -1, 1, 7]

    #     for (a, b, c, x, y, z) in product(indices, indices, steps, indices, indices, steps):
    #         slice1, slice2 = slice(a, b, c), slice(x, y, z)

    #         sub_arr = arr[slice1]
    #         self.assertEqual(slice_length(slice1, length), len(sub_arr))

    #         combined = compose_slices(slice1, slice2, length)
    #         self.assertTrue(np.all(sub_arr[slice2] == arr[combined]))

    # def test_take_on_slice(self):
    #     arr = np.arange(20, dtype=np.uint8)
    #     length = len(arr)

    #     indices = [None, -31, -20, -19, 0, 8, 19]
    #     steps = [None, -20, -10, -3, -1, 1, 7]
    #     for (a, b, c) in product(indices, indices, steps):
    #         slice1 = slice(a, b, c)
    #         inner_length = slice_length(slice1, length)
    #         if inner_length > 0:
    #             indices = np.random.randint(-inner_length + 1, inner_length, size=10)
    #             self.assertTrue(np.all(arr[slice1][indices] == arr[take_indices(slice1, indices, length)]))


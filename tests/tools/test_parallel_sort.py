import numpy as np
from numpy.testing import assert_equal
import unittest
from itertools import chain, product

from cyperf.tools import (parallel_unique, parallel_sort, cy_parallel_sort, parallel_argsort, cy_parallel_argsort,
                          argsort_fallback, sort_fallback, take_indices)
from cyperf.tools.parallel_sort_routine import (inplace_string_parallel_sort, inplace_numerical_parallel_sort,
                                                parallel_argsort_object_int, parallel_argsort_object_long,
                                                parallel_argsort_float64_int_nan, parallel_argsort_numpy_strings_int)

SupportedNumericalDtype = [np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64,
                           np.float32, np.float64]


class ParallelSortCase(unittest.TestCase):

    def get_string_data_generator(self):
        yield []
        yield ['c', 'd', 'f', 'a', 'c', 'b', 'b']
        yield ['a', 'a', 'aa', 'aaaa', 'a', 'a']
        yield ['a', 'b', 'b']
        yield ['a', 'b', 'b', 'a', 'a', 'a', 'b']
        yield ['a', 'b', 'a', 'a', 'b', 'b']
        yield ['R', 'T', 'B', 'TR', 'T']
        yield ['Rrrrrrrrrrrrrrrrr' * 100, 'Rrrrr' * 200] * 3

        for i in range(10):
            length = np.random.randint(0, 1000)
            yield list(map(str, np.random.rand(length)))
            yield list(map(str, np.random.randint(0, 3, length)))

    def get_numerical_data_generator(self):
        for _ in range(20):
            a = np.random.randn(np.random.randint(1000) + 1)
            a *= 1000
            for dtype in SupportedNumericalDtype:
                yield a.astype(dtype)

    def get_other_type_generator(self):
        return chain([np.random.randint(0, 2, np.random.randint(1, 1000)).astype(bool) for _ in range(10)] +
                     [np.random.choice(np.arange("2010-01-01", "2012-01-01", dtype='M'), np.random.randint(1, 100))
                      for _ in range(10)] +
                     [np.random.randint(0, 10**8, np.random.randint(1, 100)).astype('datetime64[s]') for _ in range(10)] +
                     [np.random.randint(1, 100, np.random.randint(1, 100)).astype('m') for _ in range(10)] +
                     [np.array(["2012-01-01", "2010-01-01", "", "2012-01-01", "2012-01-01"], dtype="M")])

    def test_inplace_string_parallel_sort(self):
        with self.assertRaises(TypeError):
            inplace_string_parallel_sort(np.random.rand(5))

        with self.assertRaises(TypeError):
            inplace_string_parallel_sort(np.random.rand(5).astype('S'))

        with self.assertRaises(TypeError):
            inplace_string_parallel_sort(np.random.rand(5).astype('U'))

        with self.assertRaises(TypeError):  # it does not allow to mix unicode and bytes
            inplace_string_parallel_sort(['a', u'b', u'a', b'a', b'b', 'b'])

        for a in self.get_string_data_generator():
            a_arr = np.array(a, dtype='O')  # copy
            inplace_string_parallel_sort(a)
            self.assertEqual(a, sorted(a_arr))

            inplace_string_parallel_sort(a, reverse=True)
            self.assertEqual(a, sorted(a_arr)[::-1])

            a = a_arr.tolist()  # copy
            inplace_string_parallel_sort(a_arr)
            assert_equal(a_arr, np.sort(a))
            self.assertEqual(a_arr.tolist(), sorted(a))

            inplace_string_parallel_sort(a_arr, reverse=True)
            assert_equal(a_arr, np.sort(a)[::-1])
            self.assertEqual(a_arr.tolist(), sorted(a)[::-1])

            if len(a) > 2:
                with self.assertRaises(ValueError):
                    inplace_string_parallel_sort(a_arr[::2])  # 'ndarray is not C-contiguous'

            if len(a) > 0:
                with self.assertRaises(TypeError):
                    a[len(a) // 2] = 123
                    inplace_string_parallel_sort(a)

                with self.assertRaises(TypeError):
                    a_arr[len(a) // 2] = object
                    inplace_string_parallel_sort(a_arr)

    def test_inplace_numerical_parallel_sort(self):
        for b in self.get_numerical_data_generator():
            bb = b.copy()

            inplace_numerical_parallel_sort(b)
            assert_equal(b, np.sort(b))

            b = bb.copy()
            inplace_numerical_parallel_sort(b, reverse=True)
            assert_equal(b, np.sort(bb)[::-1])

    def test_unique_parallel(self):
        for _ in range(20):
            x = np.random.randint(-np.random.randint(100), np.random.randint(1, 100), np.random.randint(1, 1000))
            x //= 7
            for dtype in SupportedNumericalDtype:
                y = x.astype(dtype)
                assert_equal(parallel_unique(y), np.unique(y))

        assert_equal(parallel_unique([]), np.unique([]))
        assert_equal(parallel_unique([1] * 100), np.unique([1]))
        assert_equal(parallel_unique([1.23] * 100), np.unique([1.23]))

        for x in self.get_string_data_generator():
            assert_equal(parallel_unique(x), np.unique(x))  # this is supported
            assert_equal(parallel_unique(np.asarray(x, dtype='O')), np.unique(x))  # this is supported
            assert_equal(parallel_unique(np.asarray(x, dtype='S')), np.unique(np.asarray(x, dtype='S')))
            assert_equal(parallel_unique(np.asarray(x, dtype='U')), np.unique(np.asarray(x, dtype='U')))

            if len(x) > 0:
                x[0] = 12
                assert_equal(parallel_unique(x), np.unique(x))

    def test_parallel_sort(self):
        for b in self.get_numerical_data_generator():
            bb = b.copy()
            assert_equal(parallel_sort(b), np.sort(b))
            assert_equal(parallel_sort(b[::-1]), np.sort(b))  # non contiguous case
            assert_equal(parallel_sort(b, reverse=True), np.sort(b)[::-1])
            assert_equal(bb, b)

        for a in self.get_string_data_generator():
            if len(a) > 0:
                for b in [a, np.asarray(a, dtype='O'), np.asarray(a, dtype='S'), np.asarray(a, dtype='U')]:
                    bb = np.asarray(b, dtype='O')
                    assert_equal(parallel_sort(b), sorted(b))
                    assert_equal(parallel_sort(b, reverse=True), sorted(b)[::-1])
                    assert_equal(bb, b)

    def test_sort_other_types(self):
        for b in self.get_other_type_generator():
            bb = b.copy()
            assert_equal(cy_parallel_sort(b), np.sort(b))
            assert_equal(bb, b)
            self.assertEqual(b.dtype, bb.dtype)

            np.random.shuffle(b)
            assert_equal(cy_parallel_sort(b[::-1]), np.sort(b))  # non contiguous case

            np.random.shuffle(b)
            assert_equal(cy_parallel_sort(b, reverse=True), np.sort(b)[::-1])

    def test_argsort_py_and_numpy_string(self):
        for a in self.get_string_data_generator():
            expected = {False: np.argsort(a, kind="merge"), True: argsort_fallback(a, reverse=True)}

            for reverse in [True, False]:
                assert_equal(parallel_argsort_object_int(a, reverse), expected[reverse])
                assert_equal(parallel_argsort_object_long(a, reverse), expected[reverse])

            for value, reverse in product([a, np.array(a, dtype='O'), np.array(a, dtype='S')], [True, False]):
                value_copy = list(value)
                assert_equal(cy_parallel_argsort(value, reverse), expected[reverse])
                assert_equal(parallel_argsort(value, reverse), expected[reverse])
                assert_equal(argsort_fallback(value, reverse), expected[reverse])
                assert_equal(take_indices(value, cy_parallel_argsort(value, reverse)), sort_fallback(value, reverse))
                assert_equal(take_indices(value, cy_parallel_argsort(value, reverse)), parallel_sort(value, reverse))
                self.assertEqual(list(value), value_copy)

            if len(a) > 0:
                with self.assertRaises(TypeError):
                    a[len(a) // 2] = 123
                    parallel_argsort_object_int(a, reverse=True)

                with self.assertRaises(TypeError):
                    a[len(a) // 2] = 123
                    cy_parallel_argsort(a)

    def test_sort_argsort_string_subclass(self):
        for a in self.get_string_data_generator():
            a = list(np.array(a).astype(str)) + a
            assert_equal(parallel_argsort_object_int(a, reverse=False), np.argsort(a, kind="merge"))
            assert_equal(parallel_argsort_object_long(a, reverse=True), argsort_fallback(a, reverse=True))

    def test_sort_argsort_long_numpy_string(self):
        for a in self.get_string_data_generator():
            a = np.array(a, dtype='S302')
            assert_equal(parallel_argsort(a, reverse=False), np.argsort(a, kind="merge"))
            assert_equal(parallel_argsort(a, reverse=True), argsort_fallback(a, reverse=True))

    def test_sort_argsort_long_numpy_string_explicit(self):
        a = np.array(['aaa' * 101, 'aaa' * 101], dtype='S302')
        for reverse in [True, False]:
            assert_equal(parallel_argsort_numpy_strings_int(a, reverse=reverse), [0, 1])

    def _check_float(self, a):
        assert_equal(parallel_sort(a), np.sort(a))
        assert_equal(cy_parallel_sort(a), np.sort(a))
        assert_equal(parallel_sort(a, reverse=True), np.sort(a)[::-1])
        assert_equal(cy_parallel_sort(a, reverse=True), np.sort(a)[::-1])

        assert_equal(a[parallel_argsort(a)], a[np.argsort(a, kind="merge")])
        assert_equal(parallel_argsort(a), np.argsort(a, kind="merge"))
        assert_equal(a[parallel_argsort(a, True)], np.sort(a)[::-1])

    def test_sort_argsort_numeric_nan(self):
        for _ in range(5):
            for dtype in [np.float32, np.float64]:
                a = np.random.rand(100).astype(dtype)
                a[::3] = np.nan
                self._check_float(a)
                assert_equal(parallel_argsort(a), parallel_argsort_float64_int_nan(a.astype(np.float)))
                assert_equal(parallel_argsort(a, True), parallel_argsort_float64_int_nan(a.astype(np.float), True))

    def test_sort_argsort_numeric_inf(self):
        for _ in range(5):
            for dtype in [np.float32, np.float64]:
                a = np.random.rand(100).astype(dtype)
                a[::3] = np.inf
                a[::2] = -np.inf
                self._check_float(a)
                assert_equal(parallel_argsort(a), parallel_argsort_float64_int_nan(a.astype(np.float)))
                assert_equal(parallel_argsort(a, True), parallel_argsort_float64_int_nan(a.astype(np.float), True))

    def test_sort_argsort_numeric_inf_nan(self):
        for _ in range(5):
            for dtype in [np.float32, np.float64]:
                a = np.random.rand(100).astype(dtype)
                a[::3] = np.inf
                a[::2] = -np.inf
                a[::7] = np.nan
                self._check_float(a)
                assert_equal(parallel_argsort(a), parallel_argsort_float64_int_nan(a.astype(np.float)))
                assert_equal(parallel_argsort(a, True), parallel_argsort_float64_int_nan(a.astype(np.float), True))

    def test_argsort_numeric(self):
        for b in chain(self.get_numerical_data_generator(), self.get_other_type_generator()):
            bb = b.copy()
            assert_equal(parallel_argsort(b), np.argsort(b, kind="merge"))
            assert_equal(cy_parallel_argsort(b), np.argsort(b, kind="merge"))
            assert_equal(parallel_argsort(b[::2]), np.argsort(b[::2], kind="merge"))  # non contiguous case

            assert_equal(b[parallel_argsort(b, reverse=True)], b[np.argsort(b)[::-1]])
            assert_equal(b[cy_parallel_argsort(b, reverse=True)], b[np.argsort(b)[::-1]])
            assert_equal(bb, b)

    def test_argsort_numeric_stability(self):
        a = np.arange(10) % 3
        assert_equal(cy_parallel_argsort(a), np.argsort(a, kind="merge"))
        assert_equal(cy_parallel_argsort(a, reverse=True), [2, 5, 8, 1, 4, 7, 0, 3, 6, 9])
        assert_equal(cy_parallel_argsort(a, reverse=True), argsort_fallback(a, reverse=True))

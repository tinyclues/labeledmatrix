
from itertools import product
import numpy as np
import unittest

from cyperf.tools import slice_length, compose_slices, take_indices
from cyperf.tools.getter import (apply_python_dict, cy_safe_intern,
                                 cast_to_float_array, cast_to_long_array, cast_to_ascii, cast_to_unicode,
                                 coalesce_is_not_none, coalesce_generic, Unifier)
from cyperf.tools.sort_tools import (cython_argpartition, _inplace_permutation, cython_argsort)
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

    def test_cy_safe_intern_ITER_type(self):
        s1_array = np.array(['1', '2', '4'], dtype='S1')
        self.assertEqual(cy_safe_intern(s1_array), s1_array.tolist())

        s2_array = np.array(['1', '2', '4'], dtype='S2')
        self.assertEqual(cy_safe_intern(s2_array), s2_array.tolist())

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

    def test_coerse_float(self):
        np.testing.assert_equal(cast_to_float_array([]), [])

        # making copy in trivial case
        aa = np.array([3., 4, 5], dtype=np.float)
        bb = cast_to_float_array(aa)
        self.assertNotEqual(id(aa), id(bb))
        np.testing.assert_equal(aa, bb)

        np.testing.assert_equal(cast_to_float_array(['3', '4', '5.4']), [3, 4, 5.4])
        np.testing.assert_equal(cast_to_float_array(('3.0', 4, '5???4')), [3, 4, np.nan])
        np.testing.assert_equal(cast_to_float_array(['3.0', str, 5]), [3, np.nan, 5])
        np.testing.assert_equal(cast_to_float_array(np.array(['3.0', str, 5])), [3, np.nan, 5])
        np.testing.assert_equal(cast_to_float_array(np.array([4, 3, '3'], dtype=np.int), 'safe'), [4., 3., 3.])
        with self.assertRaises(TypeError):
            cast_to_float_array([4, 're', 3.1], 'safe')
        with self.assertRaises(TypeError):
            cast_to_float_array(np.array([4, 're', 3.1], dtype=np.object), 'safe')
        np.testing.assert_equal(cast_to_float_array([4, 're', 3.1]), [4., np.nan, 3.1])
        np.testing.assert_equal(cast_to_float_array(np.array([4, 3]), 'safe'), [4., 3.])
        np.testing.assert_equal(cast_to_float_array(['3', np.nan, '5.4']), [3, np.nan, 5.4])

    def test_coerse_long(self):
        np.testing.assert_equal(cast_to_long_array([]), [])

        # making copy in trivial case
        aa = np.array([3, 4, 5], dtype=np.int64)
        bb = cast_to_long_array(aa)
        self.assertNotEqual(id(aa), id(bb))
        np.testing.assert_equal(aa, bb)

        np.testing.assert_equal(cast_to_long_array(['3', '4', '5.4']), [3, 4, 5])
        np.testing.assert_equal(cast_to_long_array(['3', '4', '5.9']), [3, 4, 5])
        np.testing.assert_equal(cast_to_long_array(('3.0', 4, '5???4'), default=-1), [3, 4, -1])
        np.testing.assert_equal(cast_to_long_array(['3.0', str, 5], default=-1), [3, -1, 5])
        np.testing.assert_equal(cast_to_long_array(np.array(['3.0', str, 5]), default=-1), [3, -1, 5])

        np.testing.assert_equal(cast_to_long_array([4, 3, '3'], 'same_kind'), [4, 3, -9223372036854775808])
        np.testing.assert_equal(cast_to_long_array([4, 3, '3'], 'unsafe'), [4, 3, 3])
        np.testing.assert_equal(cast_to_long_array(np.array([4, 3, '3'], dtype=np.int), 'safe'), [4, 3, 3])
        with self.assertRaises(TypeError):
            cast_to_long_array([4, 're', 3], 'safe')
        with self.assertRaises(TypeError):
            cast_to_long_array(np.array([4, 're', 3], dtype=np.object), 'safe')
        np.testing.assert_equal(cast_to_long_array([4, 're', 3], 'same_kind'), [4, -9223372036854775808, 3])
        np.testing.assert_equal(cast_to_long_array([4, 're', 3], 'unsafe'), [4, -9223372036854775808, 3])
        np.testing.assert_equal(cast_to_long_array(np.array([4.4, 3.1]), 'safe'), [4, 3])
        np.testing.assert_equal(cast_to_long_array(['3', np.nan, '5.4']), [3, -9223372036854775808, 5])
        np.testing.assert_equal(cast_to_long_array(np.array([[3, np.nan, 5.4]])), [[3, -9223372036854775808, 5]])

    def test_coerse_ascii(self):
        arr = [b'camelCase', b'\xe8cop\xc3ge', u'\xe8cop\xc3ge', 1, np.nan, ()]
        self.assertEqual(cast_to_ascii(arr), ['camelCase', 'copge', b'copge', '1', 'nan', '()'])

        def py_ascii(x):
            import six  # FIXME
            return x.encode('ascii', errors='ignore') \
                if isinstance(x, six.text_type) \
                else str(six.text_type(x, 'ascii', errors='ignore')) \
                if isinstance(x, (str, bytes)) else str(x)
        self.assertEqual(cast_to_ascii(arr), list(map(py_ascii, arr)))

    def test_coerse_unicode(self):
        arr = [b'camelCase', b'\xe8cO\xa8e\xc3\xa9', b'\xc3\xa90e', 1, np.nan, ()]
        self.assertEqual(cast_to_unicode(arr),
                         [b'camelCase', b'cOe\xc3\xa9', b'\xc3\xa90e', u'1', 'nan', '()'])
        self.assertEqual(type(cast_to_unicode([1])[0]), str)

        def py_uni(x):
            import six  # FIXME
            return (six.text_type(x, 'utf-8', errors='ignore') if isinstance(x, bytes) else x)\
                    .encode('utf-8', errors='ignore') if isinstance(x, bytes) else six.text_type(x)

        self.assertEqual(cast_to_unicode(arr), list(map(py_uni, arr)))

    def test_coalesce_is_not_none(self):
        self.assertEqual(coalesce_is_not_none(None, 0, None, 3, default=-1), 0)
        self.assertEqual(coalesce_is_not_none(None, None, None, default=-1), -1)
        self.assertEqual(coalesce_is_not_none(default=-1), -1)

    def test_coalesce_generic(self):
        self.assertEqual(coalesce_generic(0, -123, 2, predicate=lambda x: x > 0, default=-42), 2)
        self.assertEqual(coalesce_generic(0, -123, 0, predicate=lambda x: x > 0, default=-42), -42)
        self.assertEqual(coalesce_generic(predicate=lambda x: x > 0, default=-42), -42)


class UnifierTestCase(unittest.TestCase):

    def test_unifier_map(self):
        u = Unifier()
        self.assertIsInstance(u, dict)
        s1, s2 = 'foo!', 'foo!'
        seq = [s1, s2, 4, [4, 2], (2, 4), (2, 4)]

        unified_seq = u.map(seq)
        self.assertEquals(unified_seq, list(map(u.unify, seq)))
        self.assertEquals(seq, unified_seq)
        self.assertEquals(u, {s1: s1, 4: 4, (2, 4): (2, 4)})
        self.assertIs(unified_seq[0], unified_seq[1])
        self.assertIs(unified_seq[-1], unified_seq[-2])

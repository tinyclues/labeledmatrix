
import unittest
from numpy.testing import assert_equal
from cyperf.where import (indices_where_between, np, indices_where_same, indices_where_not_same, indices_where_ge,
                          indices_where_gt, indices_where, indices_where_not, indices_where_eq, indices_where_ne,
                          indices_where_lt, indices_where_le, fast_filter)
from six import PY3


class testWhere(unittest.TestCase):

    array1 = np.arange(4).astype('S')
    array2 = np.array(['x', '', 'xx', 'y', '', 'z'], dtype='S')
    array3 = np.array([]).astype('S')

    def test_fast_filter(self):
        assert_equal(indices_where_eq(self.array1, b'1'), fast_filter('eq', b'1')(self.array1))
        assert_equal(indices_where_le(self.array1, b'2'), fast_filter('le', b'2')(self.array1))
        assert_equal(indices_where_between(self.array2, (b'x', b'y')),
                     fast_filter('between', (b'x', b'y'))(self.array2))

    def test_indices_where(self):
        assert_equal(indices_where(self.array1), np.arange(4))
        assert_equal(indices_where(self.array1.tolist()), np.arange(4))
        assert_equal(indices_where(self.array1.astype(np.float)), np.arange(1, 4))
        assert_equal(indices_where(self.array1.astype('O')), np.arange(4))
        assert_equal(indices_where(self.array1.astype('U')), np.arange(4))

        assert_equal(indices_where(self.array2), [0, 2, 3, 5])
        assert_equal(indices_where(self.array2.astype('O')), [0, 2, 3, 5])
        assert_equal(indices_where(self.array2.tolist()), [0, 2, 3, 5])
        assert_equal(indices_where(self.array3), [])

    def test_indices_where_not(self):
        assert_equal(indices_where_not(self.array1), [])
        assert_equal(indices_where_not(self.array1.astype('U')), [])
        assert_equal(indices_where_not(self.array1.astype('O')), [])
        assert_equal(indices_where_not(self.array1.astype(np.int)), [0])
        assert_equal(indices_where_not(self.array1.astype(np.float)), [0])

        assert_equal(indices_where_not(self.array2.astype('O')), [1, 4])
        assert_equal(indices_where_not(self.array2), [1, 4])
        assert_equal(indices_where_not(self.array2.astype('U')), [1, 4])
        assert_equal(indices_where_not(self.array2.astype('O')), [1, 4])

        assert_equal(indices_where_not(self.array3), [])
        assert_equal(indices_where_not(self.array3.astype('U')), [])
        assert_equal(indices_where_not(self.array3.astype('O')), [])

    def test_indices_where_eq(self):
        assert_equal(indices_where_eq(self.array1, b'1'), [1])
        assert_equal(indices_where_eq(self.array1.tolist(), b'1'), [1])
        assert_equal(indices_where_eq(self.array1, 1), [])
        assert_equal(indices_where_eq(self.array1.astype('U'), u'1'), [1])

        assert_equal(indices_where_eq(self.array1, b'2'), [2])
        assert_equal(indices_where_eq(self.array1.astype('O'), b'2'), [2])

        assert_equal(indices_where_eq(self.array2, b''), [1, 4])
        assert_equal(indices_where_eq(self.array2, b'x'), [0])
        assert_equal(indices_where_eq(self.array2.astype('U'), u'x'), [0])
        assert_equal(indices_where_eq(self.array2.astype('O'), b'x'), [0])

        assert_equal(indices_where_eq(self.array3, b''), [])
        assert_equal(indices_where_eq(self.array1, {}), [])

    def test_indices_where_ne(self):
        assert_equal(indices_where_ne(self.array1, b'1'), [0, 2, 3])
        assert_equal(indices_where_ne(self.array1.astype('U'), u'1'), [0, 2, 3])
        assert_equal(indices_where_ne(self.array1, 1), [0, 1, 2, 3])
        assert_equal(indices_where_ne(self.array1.astype('O'), 1), [0, 1, 2, 3])
        assert_equal(indices_where_ne(self.array1, b'2'), [0, 1, 3])
        assert_equal(indices_where_ne(self.array1.astype('O'), b'2'), [0, 1, 3])
        assert_equal(indices_where_ne(self.array1.astype('U').astype('O'), u'2'), [0, 1, 3])
        assert_equal(indices_where_ne(self.array1, {}), [0, 1, 2, 3])

        assert_equal(indices_where_ne(self.array2, b''), [0, 2, 3, 5])
        assert_equal(indices_where_ne(self.array2, b'x'), [1, 2, 3, 4, 5])
        assert_equal(indices_where_ne(self.array2, b'x'), [1, 2, 3, 4, 5])

        assert_equal(indices_where_ne(self.array3, ''), [])

    def test_indices_where_lt(self):
        assert_equal(indices_where_lt(self.array1, min(self.array1)), [])
        assert_equal(indices_where_lt(self.array1, max(self.array1)), [0, 1, 2])
        # passing python list and together with np.string_
        assert_equal(indices_where_lt(self.array1.tolist(), max(self.array1)), [0, 1, 2])
        assert_equal(indices_where_lt(self.array1, b'1'), [0])
        assert_equal(indices_where_lt(self.array1, b'200'), [0, 1, 2])
        assert_equal(indices_where_lt(self.array2, b''), [])
        assert_equal(indices_where_lt(self.array2, b'x'), [1, 4])
        assert_equal(indices_where_lt(self.array3, b'r'), [])

        if PY3:
            assert_equal(indices_where_lt(self.array1.astype('U'), '200'), [0, 1, 2])
        else:
            # in PY3 it raises TypeError: '<' not supported between instances of 'numpy.ndarray' and 'str'
            # TypeError: '<' not supported between instances of 'numpy.ndarray' and 'int'
            assert_equal(indices_where_lt(self.array1, 1), [])
            assert_equal(indices_where_lt(self.array1, u'1'), [0])
            assert_equal(indices_where_lt(self.array1, {}), [])

    def test_indices_where_le(self):
        assert_equal(indices_where_le(self.array1, b'1'), [0, 1])
        assert_equal(indices_where_le(self.array1.astype('O'), b'1'), [0, 1])
        assert_equal(indices_where_le(self.array1, b'200'), [0, 1, 2])
        assert_equal(indices_where_le(self.array2, b''), [1, 4])
        assert_equal(indices_where_le(self.array2, b'x'), [0, 1, 4])
        assert_equal(indices_where_le(self.array3, b'r'), [])

        if PY3:
            assert_equal(indices_where_le(self.array1.astype('U'), u'1'), [0, 1])
        else:
            assert_equal(indices_where_le(self.array1, 1), [])
            assert_equal(indices_where_le(self.array1, {}), [])
            assert_equal(indices_where_le(self.array1, u'1'), [0, 1])

    def test_indices_where_gt(self):
        assert_equal(indices_where_gt(self.array1, b'1'), [2, 3])
        assert_equal(indices_where_gt(self.array1, b'200'), [3])
        assert_equal(indices_where_gt(self.array2, b''), [0, 2, 3, 5])
        assert_equal(indices_where_gt(self.array2, b'x'), [2, 3, 5])
        assert_equal(indices_where_gt(self.array3, b'r'), [])

        if PY3:
            assert_equal(indices_where_gt(self.array1.astype('U'), u'1'), [2, 3])
        else:
            assert_equal(indices_where_gt(self.array1, 1), [0, 1, 2, 3])
            assert_equal(indices_where_gt(self.array1, u'1'), [2, 3])
            assert_equal(indices_where_gt(self.array1, {}), [0, 1, 2, 3])

    def test_indices_where_ge(self):
        assert_equal(indices_where_ge(self.array1, b'1'), [1, 2, 3])
        assert_equal(indices_where_ge(self.array1, b'200'), [3])
        assert_equal(indices_where_ge(self.array2, b''), [0, 1, 2, 3, 4, 5])
        assert_equal(indices_where_ge(self.array2, b'x'), [0, 2, 3, 5])
        assert_equal(indices_where_ge(self.array3, b'r'), [])

        if PY3:
            assert_equal(indices_where_ge(self.array1.astype('U'), u'1'), [1, 2, 3])
        else:
            assert_equal(indices_where_ge(self.array1, u'1'), [1, 2, 3])
            assert_equal(indices_where_ge(self.array1, {}), [0, 1, 2, 3])
            assert_equal(indices_where_ge(self.array1, 1), [0, 1, 2, 3])

    def test_indices_where_between(self):
        assert_equal(indices_where_between(self.array1, (b'1', b'3')), [1, 2])
        assert_equal(indices_where_between(self.array1, (b'200', b'5')), [3])
        assert_equal(indices_where_between(self.array2, (b'', b'xx')), [0, 1, 4])
        assert_equal(indices_where_between(self.array2, (b'y', b'y')), [])
        assert_equal(indices_where_between(self.array3, (b'', b'x')), [])
        if PY3:
            assert_equal(indices_where_between(self.array1.astype('U'), (u'1', u'3')), [1, 2])
            assert_equal(indices_where_between(self.array2.astype('U'), ('', 'xx')), [0, 1, 4])
        else:
            assert_equal(indices_where_between(self.array1, (1, 3)), [])
            assert_equal(indices_where_between(self.array1, ({}, {})), [])

    def test_indices_same(self):
        assert_equal(indices_where_same(self.array1, self.array1), [0, 1, 2, 3])
        assert_equal(indices_where_same(self.array1, self.array1.astype(np.int)), [])
        if PY3:
            assert_equal(indices_where_same(self.array1, self.array1.astype('U')), [])
        else:
            assert_equal(indices_where_same(self.array1, self.array1.astype('U')), [0, 1, 2, 3])
        assert_equal(indices_where_same(self.array1, [x for x in self.array1]), [0, 1, 2, 3])
        a = self.array1.copy()
        a[::2] = 'r'
        assert_equal(indices_where_same(self.array1, a), [1, 3])

        assert_equal(indices_where_same(self.array2, self.array2[::-1]), [1, 4])

        assert_equal(indices_where_same(self.array3, self.array3), [])

    def test_indices_not_same(self):
        assert_equal(indices_where_not_same(self.array1, self.array1), [])
        assert_equal(indices_where_not_same(self.array1, self.array1.astype(np.int)), [0, 1, 2, 3])
        if PY3:
            assert_equal(indices_where_not_same(self.array1, self.array1.astype('U')), [0, 1, 2, 3])
        else:
            assert_equal(indices_where_not_same(self.array1, self.array1.astype('U')), [])

        assert_equal(indices_where_not_same(self.array1, [x for x in self.array1]), [])
        a = self.array1.copy()
        a[::2] = 'r'
        assert_equal(indices_where_not_same(self.array1, a), [0, 2])

        assert_equal(indices_where_not_same(self.array2, self.array2[::-1]), [0, 2, 3, 5])

        assert_equal(indices_where_not_same(self.array3, self.array3), [])

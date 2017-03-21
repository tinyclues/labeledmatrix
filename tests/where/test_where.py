import unittest
from cyperf.where import *


class testWhere(unittest.TestCase):

    array1 = np.arange(4).astype('S')
    array2 = np.array(['x', '', 'xx', 'y', '', 'z'])
    array3 = np.array([]).astype('S')

    def test_indices_where(self):
        np.testing.assert_equal(indices_where(self.array1), np.arange(4))
        np.testing.assert_equal(indices_where(self.array1.astype(np.float)), np.arange(1, 4))
        np.testing.assert_equal(indices_where(self.array1.astype('O')), np.arange(4))
        np.testing.assert_equal(indices_where(self.array2), [0, 2, 3, 5])
        np.testing.assert_equal(indices_where(self.array3), [])

    def test_indices_where_not(self):
        np.testing.assert_equal(indices_where_not(self.array1), [])
        np.testing.assert_equal(indices_where_not(self.array1.astype(np.int)), [0])

        np.testing.assert_equal(indices_where_not(self.array2), [1, 4])

        np.testing.assert_equal(indices_where_not(self.array3), [])

    def test_indices_where_eq(self):
        np.testing.assert_equal(indices_where_eq(self.array1, '1'), [1])
        np.testing.assert_equal(indices_where_eq(self.array1.tolist(), '1'), [1])
        np.testing.assert_equal(indices_where_eq(self.array1, 1), [])
        np.testing.assert_equal(indices_where_eq(self.array1, u'1'), [1])
        np.testing.assert_equal(indices_where_eq(self.array1, {}), [])
        np.testing.assert_equal(indices_where_eq(self.array1, '2'), [2])

        np.testing.assert_equal(indices_where_eq(self.array2, ''), [1, 4])
        np.testing.assert_equal(indices_where_eq(self.array2, 'x'), [0])

        np.testing.assert_equal(indices_where_eq(self.array3, ''), [])

    def test_indices_where_ne(self):
        np.testing.assert_equal(indices_where_ne(self.array1, '1'), [0, 2, 3])
        np.testing.assert_equal(indices_where_ne(self.array1, 1), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_ne(self.array1.astype('O'), 1), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_ne(self.array1, u'1'), [0, 2, 3])
        np.testing.assert_equal(indices_where_ne(self.array1, {}), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_ne(self.array1, '2'), [0, 1, 3])

        np.testing.assert_equal(indices_where_ne(self.array2, ''), [0, 2, 3, 5])
        np.testing.assert_equal(indices_where_ne(self.array2, 'x'), [1, 2, 3, 4, 5])

        np.testing.assert_equal(indices_where_ne(self.array3, ''), [])

    def test_indices_where_lt(self):
        np.testing.assert_equal(indices_where_lt(self.array1, min(self.array1)), [])
        np.testing.assert_equal(indices_where_lt(self.array1, max(self.array1)), [0, 1, 2])
        # passing python list and together with np.string_
        np.testing.assert_equal(indices_where_lt(self.array1.tolist(), max(self.array1)), [0, 1, 2])
        np.testing.assert_equal(indices_where_lt(self.array1, '1'), [0])
        np.testing.assert_equal(indices_where_lt(self.array1, 1), [])
        np.testing.assert_equal(indices_where_lt(self.array1, u'1'), [0])
        np.testing.assert_equal(indices_where_lt(self.array1, {}), [])
        np.testing.assert_equal(indices_where_lt(self.array1, '200'), [0, 1, 2])

        np.testing.assert_equal(indices_where_lt(self.array2, ''), [])
        np.testing.assert_equal(indices_where_lt(self.array2, 'x'), [1, 4])

        np.testing.assert_equal(indices_where_lt(self.array3, 'r'), [])

    def test_indices_where_le(self):
        np.testing.assert_equal(indices_where_le(self.array1, '1'), [0, 1])
        np.testing.assert_equal(indices_where_le(self.array1, 1), [])
        np.testing.assert_equal(indices_where_le(self.array1, u'1'), [0, 1])
        np.testing.assert_equal(indices_where_le(self.array1, {}), [])
        np.testing.assert_equal(indices_where_le(self.array1, '200'), [0, 1, 2])

        np.testing.assert_equal(indices_where_le(self.array2, ''), [1, 4])
        np.testing.assert_equal(indices_where_le(self.array2, 'x'), [0, 1, 4])

        np.testing.assert_equal(indices_where_le(self.array3, 'r'), [])

    def test_indices_where_gt(self):
        np.testing.assert_equal(indices_where_gt(self.array1, '1'), [2, 3])
        np.testing.assert_equal(indices_where_gt(self.array1, 1), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_gt(self.array1, u'1'), [2, 3])
        np.testing.assert_equal(indices_where_gt(self.array1, {}), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_gt(self.array1, '200'), [3])

        np.testing.assert_equal(indices_where_gt(self.array2, ''), [0, 2, 3, 5])
        np.testing.assert_equal(indices_where_gt(self.array2, 'x'), [2, 3, 5])

        np.testing.assert_equal(indices_where_gt(self.array3, 'r'), [])

    def test_indices_where_ge(self):
        np.testing.assert_equal(indices_where_ge(self.array1, '1'), [1, 2, 3])
        np.testing.assert_equal(indices_where_ge(self.array1, 1), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_ge(self.array1, u'1'), [1, 2, 3])
        np.testing.assert_equal(indices_where_ge(self.array1, {}), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_ge(self.array1, '200'), [3])

        np.testing.assert_equal(indices_where_ge(self.array2, ''), [0, 1, 2, 3, 4, 5])
        np.testing.assert_equal(indices_where_ge(self.array2, 'x'), [0, 2, 3, 5])

        np.testing.assert_equal(indices_where_ge(self.array3, 'r'), [])

    def test_indices_where_between(self):
        np.testing.assert_equal(indices_where_between(self.array1, ('1', '3')), [1, 2])
        np.testing.assert_equal(indices_where_between(self.array1, (1, 3)), [])
        # strange but it DOES WORK AS expected
        #np.testing.assert_equal(indices_where_between(self.array1, (u'1', u'3')), [1, 2])
        np.testing.assert_equal(indices_where_between(self.array1, ({}, {})), [])
        np.testing.assert_equal(indices_where_between(self.array1, ('200', '5')), [3])

        np.testing.assert_equal(indices_where_between(self.array2, ('', 'xx')), [0, 1, 4])
        np.testing.assert_equal(indices_where_between(self.array2, ('y', 'y')), [])

        np.testing.assert_equal(indices_where_between(self.array3, ('', 'x')), [])

    def test_indices_same(self):
        np.testing.assert_equal(indices_where_same(self.array1, self.array1), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_same(self.array1, self.array1.astype(np.int)), [])
        np.testing.assert_equal(indices_where_same(self.array1, self.array1.astype('U')), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_same(self.array1, [x for x in self.array1]), [0, 1, 2, 3])
        a = self.array1.copy()
        a[::2] = 'r'
        np.testing.assert_equal(indices_where_same(self.array1, a), [1, 3])

        np.testing.assert_equal(indices_where_same(self.array2, self.array2[::-1]), [1, 4])

        np.testing.assert_equal(indices_where_same(self.array3, self.array3), [])

    def test_indices_not_same(self):
        np.testing.assert_equal(indices_where_not_same(self.array1, self.array1), [])
        np.testing.assert_equal(indices_where_not_same(self.array1, self.array1.astype(np.int)), [0, 1, 2, 3])
        np.testing.assert_equal(indices_where_not_same(self.array1, self.array1.astype('U')), [])
        np.testing.assert_equal(indices_where_not_same(self.array1, [x for x in self.array1]), [])
        a = self.array1.copy()
        a[::2] = 'r'
        np.testing.assert_equal(indices_where_not_same(self.array1, a), [0, 2])

        np.testing.assert_equal(indices_where_not_same(self.array2, self.array2[::-1]), [0, 2, 3, 5])

        np.testing.assert_equal(indices_where_not_same(self.array3, self.array3), [])

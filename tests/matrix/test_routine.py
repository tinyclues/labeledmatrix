
import unittest
from bisect import bisect_left as bisect_left_old

import numpy as np
from cyperf.matrix.routine import (bisect_left, batch_contains_mask, batch_is_exceptional_mask,
                                   cy_safe_slug, cy_domain_from_email_lambda, indices_lookup)
from cyperf.matrix.karma_sparse import KarmaSparse, sp
import six
from six.moves import range
from six import PY2


def kronii(left, right):
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    return np.einsum('ij, ik -> ijk', left, right, optimize='optimal').reshape(left.shape[0], -1)


class RoutineTestCase(unittest.TestCase):

    def test_safe_slug(self):
        self.assertEqual(cy_safe_slug(' "Foo bar q"u"x " '), b'foo_bar_q_u_x')
        self.assertEqual(cy_safe_slug(b'\xc3\xa9'), b'e')
        self.assertEqual(cy_safe_slug(b'\xe9'), b'?')
        self.assertEqual(cy_safe_slug(b'\xc3foo!'), b'?foo!')
        self.assertEqual(cy_safe_slug(b't\xb0st'), b't?st'),
        self.assertEquals(cy_safe_slug(u'foo\xe8@_-'), b'fooe@_-')

        if PY2:
            self.assertTrue(cy_safe_slug(b'foo!intern') is cy_safe_slug(b'foo!intern'))
        self.assertEquals(cy_safe_slug(b'`Error!'), b'`error!')
        self.assertEquals(cy_safe_slug(b'5454'), b'5454')

        self.assertEquals(cy_safe_slug(u'`Error!'), b'`error!')
        self.assertEquals(cy_safe_slug(u'5454'), b'5454')

        self.assertEquals(cy_safe_slug(u'\xc3\xa9'), b'a?')
        self.assertEquals(cy_safe_slug(u'\xe9'), b'e')
        self.assertEquals(cy_safe_slug(b'`Error!'), b'`error!')

    def test_domain_from_email_lambda(self):
        self.assertEqual(cy_domain_from_email_lambda('x', missing='RR'), 'RR')
        self.assertEqual(cy_domain_from_email_lambda('xRy', missing=''), '')
        self.assertEqual(cy_domain_from_email_lambda('xRy', missing='', delimiter='R'), 'y')
        self.assertEqual(cy_domain_from_email_lambda('x@y.com'), 'y.com')
        self.assertEqual(cy_domain_from_email_lambda('qdfd@rrr@x@y.com'), 'y.com')
        self.assertEqual(cy_domain_from_email_lambda(u'qdfd@rrr@x@y.com'), u'y.com')
        self.assertEqual(type(cy_domain_from_email_lambda(u'qdfd@rrr@x@y.com')), six.text_type)

        with self.assertRaises(AttributeError) as e:
            _ = cy_domain_from_email_lambda(33)

    def test_kronii(self):
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

    def test_batch_contains_mask(self):
        a = set(['a', 'b', 3])

        pr1 = (np.array(['r', 'a', 'f', 'gg']), np.array([False, True, False, False]))
        pr2 = (np.array([1, 2, 3, 4, 5]), np.array([False, False, True, False, False]))
        pr3 = ([], np.array([], dtype=np.bool))

        for pr in [pr1, pr2, pr3]:
            np.testing.assert_array_equal(batch_contains_mask(pr[0], a), pr[1])
            np.testing.assert_array_equal(batch_contains_mask(pr[0], frozenset(a)), pr[1])

    def test_batch_is_exceptional_mask(self):
        a = set(['a', 'b', 3, np.iinfo(np.int).min, KarmaSparse])
        expectional_char = 'f'

        pr1 = (np.array(['r', 'af', 'f', 'gg']),
               np.array([False, False, True, False]))
        pr2 = (np.array([1, 2, 3, 4, 5, np.nan, np.iinfo(np.int).min]),
               np.array([False, False, True, False, False, True, True]))

        pr2_tuple = (np.array([(1,), (2, 'a'), (3,), [3, 3], (),
                               (np.nan, 'rr', 3), (3, 'fa'), [3, 'fa', 1], (3, ('fa', 'a'))]),
                     np.array([False, False, True, True, False, False, True, False, True]))

        pr3 = (np.array([1, 'ff', 3, KarmaSparse, -np.nan, np.iinfo(np.int).max]),
               np.array([False, True, True, True, True, False]))
        pr4 = (np.array([]), np.array([], dtype=np.bool))

        for pr in [pr1, pr2, pr2_tuple, pr3, pr4]:
            for b in [a, frozenset(a)]:
                np.testing.assert_array_equal(batch_is_exceptional_mask(pr[0], b, expectional_char), pr[1])
                np.testing.assert_array_equal(batch_is_exceptional_mask(pr[0].tolist(), b, expectional_char), pr[1])

    def test_kronii_random(self):
        for _ in range(30):
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


class IndexLookupTestCase(unittest.TestCase):
    def test_basic(self):
        source_indices = np.array([2, 3,
                                   1,
                                   0, 4, 5])
        source_indptr = np.array([0, 2, 3, 6])
        positions = np.array([1, -1, 2])
        data, indices, indptr = indices_lookup(positions, source_indices, source_indptr, 0)

        ks = KarmaSparse((data, indices, indptr), (len(indptr) - 1, source_indptr[-1]), 'csr')
        np.testing.assert_array_equal(ks.toarray(),
                                      np.array([[0., 1., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0.],
                                                [1., 0., 0., 0., 1., 1.]]))

    def test_last(self):
        source_indices = np.array([2, 3,
                                   1,
                                   0, 4, 5])
        source_indptr = np.array([0, 2, 3, 6])
        positions = np.array([1, -1, 2])
        data, indices, indptr = indices_lookup(positions, source_indices, source_indptr, 1)

        ks = KarmaSparse((data, indices, indptr), (len(indptr) - 1, source_indptr[-1]), 'csr')
        np.testing.assert_array_equal(ks.toarray(),
                                      np.array([[0., 1., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 1.]]))

    def test_first_2(self):
        source_indices = np.array([2, 3,
                                   1,
                                   0, 4, 5])
        source_indptr = np.array([0, 2, 3, 6])
        positions = np.array([1, -1, 2])
        data, indices, indptr = indices_lookup(positions, source_indices, source_indptr, -2)

        ks = KarmaSparse((data, indices, indptr), (len(indptr) - 1, source_indptr[-1]), 'csr')
        np.testing.assert_array_equal(ks.toarray(),
                                      np.array([[0., 1., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0.],
                                                [1., 0., 0., 0., 1., 0.]]))
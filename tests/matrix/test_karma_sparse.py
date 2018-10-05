import unittest
import itertools
import random
from numpy import allclose as eq
from numpy.testing import assert_array_almost_equal as _np_almost_equal
from cyperf.matrix.karma_sparse import KarmaSparse, is_karmasparse, DTYPE
from cyperf.matrix.karma_sparse import sp, np, ks_diag
from cyperf.matrix.karma_sparse import truncate_by_count_axis1_sparse, truncate_by_count_axis1_dense
from cyperf.matrix.karma_sparse import dense_pivot
from cyperf.matrix import linear_error
# from sklearn.preprocessing import normalize as sk_normalize
# from karma.learning.matrix_utils import normalize
# from karma.learning.matrix_utils import truncate_by_count
from cPickle import dumps, loads
# from karma.core.utils import run_in_subprocess, Parallel
# from karma.thread_setter import open_mp_threads

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)


def np_almost_equal(x, y, decimal=5):
    return _np_almost_equal(x, y, decimal)


def array_cast(x):
    return np.asarray(x, dtype=DTYPE)


class MatrixFabric(object):
    def __init__(self, shape=None, density=0.2, format=None, random_state=None):
        self.random_state = random_state
        np.random.seed(random_state)
        self.shape = shape if shape else (np.random.randint(1, 100), np.random.randint(1, 100))
        self.density = density
        self.format = format if format in ['csr', 'csc'] else random.choice(['csr', 'csc'])

    def random_positive(self):
        res = sp.rand(*self.shape, density=self.density,
                      format=self.format, random_state=self.random_state)
        return res

    def zero(self):
        return getattr(sp, self.format + "_matrix")(self.shape)

    def random_negative(self):
        return - self.random_positive()

    def random_negative_positive(self):
        res = self.random_positive().copy()
        np.random.shuffle(res.data)
        return res - 2 * self.random_positive()

    def unsorted_indices(self):
        res = self.random_negative()
        np.random.shuffle(res.indices)
        return res

    def with_few_zeros(self, nb=10):
        res = self.random_negative()
        for _ in xrange(nb):
            i = np.random.randint(self.shape[0])
            j = np.random.randint(self.shape[1])
            res[i, j] = 0
        return res

    def with_zero_column(self):
        res = self.random_negative()
        j = np.random.randint(self.shape[1])
        res[:, j] = 0
        return res

    def with_zero_row(self):
        res = self.random_negative()
        i = np.random.randint(self.shape[0])
        res[i, :] = 0
        return res

    def with_few_nonzero(self, nb=3):
        res = self.zero()
        for _ in xrange(nb):
            i = np.random.randint(self.shape[0])
            j = np.random.randint(self.shape[1])
            res[i, j] = np.random.randn()
        return res

    def iterator(self, dense=True):
        matrixes_gen = ('random_positive', 'zero', 'random_negative',
                        'unsorted_indices', 'with_few_zeros',
                        'with_zero_column', 'with_zero_row',
                        'with_few_nonzero', 'random_negative_positive')

        for gen in matrixes_gen:
            if dense:
                yield getattr(self, gen)().toarray()
            else:
                yield getattr(self, gen)()


class TestKarmaSparse(unittest.TestCase):

    mf = MatrixFabric(random_state=None, shape=(10, 6))
    matrixes = (np.array([[0., -2., 0., 1.], [0., 0., 4., 1.]]),
                np.array([[4., 2., 2., 1.], [3., 2., 4., 1.]]),
                np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]]))
    axis = (0, 1, None)
    is_csr = (True, False)

    def test_count_nonzero(self):
        results = (np.array([0., 1., 1., 2.]), np.array([2., 2.]),
                   np.array([2., 2.]), np.array([0., 1., 1., 2.]),
                   4, 4, np.array([2., 2., 2., 2.]),
                   np.array([4., 4.]), np.array([4., 4.]),
                   np.array([2., 2., 2., 2.]), 8, 8, np.array([0., 0., 0., 0.]),
                   np.array([0., 0.]), np.array([0., 0.]),
                   np.array([0., 0., 0., 0.]), 0, 0)

        for attrs, res in zip(itertools.product(self.matrixes, self.axis, self.is_csr), results):
            matrix, ax, csr = attrs
            ks_matrix = KarmaSparse(matrix)
            if not csr:
                ks_matrix = ks_matrix.transpose()
            np_almost_equal(ks_matrix.count_nonzero(ax), res)

    def assertMatrixEqual(self, compared, expected):
        if not isinstance(compared, KarmaSparse) or not isinstance(expected, KarmaSparse):
            raise Exception('Compared and expected matrix must be KarmaSparse instances')
        np_almost_equal(compared.toarray(), expected.toarray())

    def test_init_dense(self):
        for a in self.mf.iterator(dense=True):
            np_almost_equal(KarmaSparse(a).toarray(), a)
            np_almost_equal(KarmaSparse(a).tocsc().toarray(), a)
            np_almost_equal(KarmaSparse(a).tocsr().toarray(), a)
            np_almost_equal(KarmaSparse(a.T).toarray(), a.T)
            np_almost_equal(KarmaSparse(a).T.toarray(), a.T)

            # test_init_one_dimension
            np_almost_equal(KarmaSparse(a[0]).toarray(), np.atleast_2d(a[0]))

    def test_init_scipy_sparse(self):
        for a in self.mf.iterator(dense=False):
            np_almost_equal(KarmaSparse(a).toarray(), a.toarray())
            np_almost_equal(KarmaSparse(a).tocsc().toarray(), a.toarray())
            np_almost_equal(KarmaSparse(a.T).toarray(), a.toarray().T)
            np_almost_equal(KarmaSparse(a).T.toarray(), a.toarray().T)

    def test_init_karma_sparse(self):
        for a in self.mf.iterator(dense=False):
            np_almost_equal(KarmaSparse(KarmaSparse(a)).toarray(), a.toarray())
            np_almost_equal(KarmaSparse(KarmaSparse(a)).tocsc().toarray(), a.toarray())
            np_almost_equal(KarmaSparse(KarmaSparse(a).T).toarray(), a.toarray().T)
            np_almost_equal(KarmaSparse(KarmaSparse(a.T)).toarray(), a.toarray().T)

    def test_init_coo(self):
        for a in self.mf.iterator(dense=False):
            a = a.tocoo()
            np_almost_equal(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape).toarray(), a.toarray())
            np_almost_equal(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape,
                                        format="csc").toarray(), a.toarray())

        a = KarmaSparse((np.array([0]), np.array([0])), shape=(1, 1), format='csr')
        np_almost_equal(a.toarray(), np.array([[1]]))

    def test_init_mask_coo(self):
        for a in self.mf.iterator(dense=False):
            a = a.tocoo()
            b = a.copy()
            b.data[:] = 1.
            np_almost_equal(KarmaSparse((a.row, a.col), shape=a.shape).toarray(), b.toarray())
            np_almost_equal(KarmaSparse((a.row, a.col), shape=a.shape,
                                        format="csc").toarray(), b.toarray())

    def test_init_nondeduplicated_coo(self):
        for a in self.mf.iterator(dense=False):
            a = a.tocoo().copy()
            a.row = a.row % 2
            a.col = a.col % 3
            np.random.shuffle(a.row)
            np.random.shuffle(a.col)
            np_almost_equal(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape).toarray(), a.toarray())
            np_almost_equal(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape, format="csc").toarray(), a.toarray())
            # filter out nan values
            a.data[:] = np.nan
            ks = KarmaSparse((a.data, (a.row, a.col)), shape=a.shape)
            self.assertEqual(ks.nnz, 0)

            # only one element
            a.row[:] = 0
            a.col[:] = 0
            a.data[:] = np.random.rand(a.data.shape[0])
            ks = KarmaSparse((a.data, (a.row, a.col)), shape=a.shape)
            np_almost_equal(ks[0, 0], a.data.sum(), 5)
            np_almost_equal(ks.sum(), a.data.sum(), 5)
            np_almost_equal(ks.nnz, a.data.shape[0] > 0)

    def test_init_aggregator_coo(self):
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 1, 0, 0])
        values = np.array([2, 2, -1, -10])

        ks = KarmaSparse((x, y), shape=(2, 2))
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), 4)
        np_almost_equal(ks.toarray(), np.array([[2, 0], [1, 1]]))

        ks = KarmaSparse((x, y), shape=(2, 2), aggregator="max")
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), 3)
        np_almost_equal(ks.toarray(), np.array([[1, 0], [1, 1]]))

        ks = KarmaSparse((values, (x, y)), shape=(2, 2), aggregator="max")
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), 3)
        np_almost_equal(ks.toarray(), np.array([[2, 0], [-1, 2]]))

        ks = KarmaSparse((values, (x, y)), shape=(2, 2), aggregator="min")
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), -9)
        np_almost_equal(ks.toarray(), np.array([[-10, 0], [-1, 2]]))

        ks = KarmaSparse((values, (x, y)), shape=(2, 2), aggregator="multiply")
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), -19)
        np_almost_equal(ks.toarray(), np.array([[-20, 0], [-1, 2]]))

    def test_init_zeros(self):
        for a in self.mf.iterator(dense=False):
            np_almost_equal(KarmaSparse(a.shape).toarray(), np.zeros(a.shape))
            np_almost_equal(KarmaSparse(a.shape, format="csr").toarray(), np.zeros(a.shape))
            np_almost_equal(KarmaSparse(a.shape, format="csc").toarray(), np.zeros(a.shape))

    def test_init_scalar(self):
        for x in [1, 2.]:
            ks = KarmaSparse(x)
            self.assertEqual(ks.shape, (1, 1))
            np_almost_equal(ks, KarmaSparse(np.array([[x]])))

    def test_format_respect(self):
        a = sp.rand(4, 2, 0.3, format='coo', random_state=1208)
        # sparse
        np_almost_equal(KarmaSparse(a).toarray(), a.toarray())
        self.assertEqual(KarmaSparse(a, format="csc").format, 'csc')
        self.assertEqual(KarmaSparse(a, format="csr").format, 'csr')

        # dense
        np_almost_equal(KarmaSparse(a.toarray()).toarray(), a.toarray())
        self.assertEqual(KarmaSparse(a.toarray(), format="csc").format, 'csc')
        self.assertEqual(KarmaSparse(a.toarray(), format="csr").format, 'csr')

        # KS
        self.assertEqual(KarmaSparse(KarmaSparse(a), format="csc").format, 'csc')
        self.assertEqual(KarmaSparse(KarmaSparse(a), format="csr").format, 'csr')
        # coo
        np_almost_equal(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape).toarray(), a.toarray())
        self.assertEqual(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape).format, 'csr')
        self.assertEqual(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape, format="csr").format, 'csr')
        self.assertEqual(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape, format="csc").format, 'csc')
        # mask
        self.assertEqual(KarmaSparse((a.row, a.col), shape=a.shape, format="csr").format, 'csr')
        self.assertEqual(KarmaSparse((a.row, a.col), shape=a.shape, format="csc").format, 'csc')
        # zeros
        self.assertEqual(KarmaSparse(a.shape, format="csr").format, 'csr')
        self.assertEqual(KarmaSparse(a.shape, format="csc").format, 'csc')

    def test_tocsr_tocsc(self):
        for a in self.mf.iterator(dense=True):
            ks = KarmaSparse(a)
            self.assertEqual(ks.tocsr().format, 'csr')
            self.assertEqual(ks.tocsc().format, 'csc')
            np_almost_equal(ks.tocsr().toarray(), a)
            np_almost_equal(ks.tocsc().toarray(), ks.tocsr().toarray())

    def test_exception_in_init(self):
        a = sp.rand(4, 2, 0.5, format='csr', random_state=1111)
        self.assertRaisesRegexp(ValueError,
                                "Format should be one of",
                                lambda: KarmaSparse(a, format="cscc"))
        self.assertRaisesRegexp(ValueError,
                                "Shape values should be > 0",
                                lambda: KarmaSparse(a, shape=(3, -3)))
        self.assertRaisesRegexp(ValueError,
                                "Shape values should be > 0",
                                lambda: KarmaSparse(a, shape=(-1, 3)))
        self.assertRaisesRegexp(ValueError,
                                "Wrong shape parameter, got",
                                lambda: KarmaSparse(a, shape=(0, 3, 4)))
        self.assertRaisesRegexp(TypeError,
                                "Cannot cast to KarmaSparse",
                                lambda: KarmaSparse("string", shape=(3, 4)))
        self.assertRaisesRegexp(ValueError,
                                "Cannot cast to KarmaSparse",
                                lambda: KarmaSparse(("HAHA",)))
        self.assertRaisesRegexp(ValueError,
                                "format should be specified",
                                lambda: KarmaSparse((a.data, a.indices, a.indptr)))
        self.assertRaisesRegexp(TypeError,
                                "Shape should be provided, got",
                                lambda: KarmaSparse(([0, 1], [0, 1], [0, 1]), format="csr"))
        self.assertRaisesRegexp(ValueError,
                                "Wrong indptr shape: should be",
                                lambda: KarmaSparse(([0, 1], [0, 1], [0, 1]),
                                                    shape=(2, 3), format="csr"))
        b = a.tocoo()
        self.assertRaisesRegexp(ValueError,
                                "Coordinates are too large for provided shape",
                                lambda: KarmaSparse((b.data, (b.row, b.col)),
                                                    shape=(2, 3), format="csr"))

    def test_check_format(self):
        for a in self.mf.iterator():
            # first indptr element is not 0
            mat = KarmaSparse(a, copy=True)
            mat.indptr[0] = 10
            self.assertRaisesRegexp(ValueError,
                                    'First element of indptr should be == 0',
                                    mat.check_format)

            # last indptr element is negative
            mat = KarmaSparse(a, copy=True)
            mat.indptr[-1] = -2
            self.assertRaisesRegexp(ValueError,
                                    'Last element of indptr should be >= 0',
                                    mat.check_format)

            # too high last value of indptr
            mat = KarmaSparse(a, copy=True)
            mat.indptr[-1] = mat.indices.shape[0] + 1
            self.assertRaisesRegexp(ValueError,
                                    'indptr should be <= the size of indices',
                                    mat.check_format)

            # too high values in indices
            mat = KarmaSparse(a, copy=True)
            if mat.indices.size >= 2:
                mat.indices[1] = mat.shape[1] + 4 if mat.format == 'csr' \
                    else mat.shape[0] + 4
                self.assertRaisesRegexp(ValueError,
                                        'indices values should be < ',
                                        mat.check_format)

            # negative values in indices
            mat = KarmaSparse(a, copy=True)
            if mat.indices.size != 0:
                mat.indices[0] = -2
                self.assertRaisesRegexp(ValueError,
                                        'indices values should be >= than 0',
                                        mat.check_format)

            # decreasing sequence in indices
            mat = KarmaSparse(a, copy=True)
            mat.indptr[1] = mat.indptr[-1] + 5
            if np.count_nonzero(a) > 0:
                self.assertRaisesRegexp(ValueError,
                                        'non decreasing sequence',
                                        mat.check_format)

    def test_extend(self):
        axis_0, axis_1 = (13, 7)
        for i in self.mf.iterator():
            a = KarmaSparse(i)
            at = a.transpose()
            for mat in [a, at]:
                new_shape = (mat.shape[0] + axis_0, mat.shape[1] + axis_1)
                b = mat.extend(new_shape, copy=True)
                # some attributes do not change
                np.testing.assert_array_equal(b.indices, mat.indices)
                np.testing.assert_array_equal(b.data, mat.data)
                self.assertEqual(b.format, mat.format)
                # others change
                if mat.format == 'csr':
                    indptr_complete = axis_0
                else:
                    indptr_complete = axis_1
                self.assertEqual(b.shape, new_shape)
                new_indptr = np.hstack([mat.indptr, [mat.indptr[-1]] * indptr_complete])
                np.testing.assert_array_equal(b.indptr, new_indptr)
                new_mat = np.column_stack([mat.toarray(), np.zeros((mat.shape[0], axis_1))])
                new_mat = np.vstack([new_mat, np.zeros((axis_0, mat.shape[1] + axis_1))])
                np_almost_equal(b.toarray(), new_mat)
                # check data is really copied
                init_data = mat.data.copy()
                for i in xrange(len(mat.data)):
                    mat.data[i] = np.random.random()
                np.testing.assert_array_equal(b.data, init_data)
                self.assertRaises(AssertionError, mat.extend, (mat.shape[0] - 1, mat.shape[1]))

    def test_global_argmaxmin(self):
        n, m = 5, 23
        a = sp.rand(n, m, .1, format="csr")
        b = a.toarray()
        ks = KarmaSparse(a)

        self.assertEqual((b.argmax() / m, b.argmax() % m), ks.argmax(only_nonzero=False))
        self.assertEqual(ks.argmax(only_nonzero=False), ks.argmax(only_nonzero=True))
        self.assertEqual((b.argmin() / m, b.argmin() % m), ks.argmin(only_nonzero=False))
        self.assertNotEqual(ks.argmin(only_nonzero=False), ks.argmin(only_nonzero=True))

    def test_axis_argmaxmin(self):
        n, m = 5, 23
        a = sp.rand(n, m, .1, format="csr")
        b = a.toarray()
        ks = KarmaSparse(a)

        np_almost_equal(ks.argmax(1, False), b.argmax(1))
        np_almost_equal(ks.argmin(1, False), b.argmin(1))
        self.assertTrue(not eq(ks.argmin(1, True), b.argmin(1)))

        # np_almost_equal(ks.argmax(0, False), b.argmax(0))
        np_almost_equal(ks.argmin(0, False), b.argmin(0))

    def test_rank_axis_none(self):
        for i in self.mf.iterator(dense=True):
            a = KarmaSparse(i, format="csr")
            at = a.tocsc()
            for spmat in [a, at]:
                rank = spmat.rank(axis=None)
                flat_mat = np.reshape(i, (1, i.size))
                flat_non_zero_mat = flat_mat[np.nonzero(flat_mat)]
                expected_rank = flat_non_zero_mat.argsort(axis=0).argsort(axis=0) + 1
                np_almost_equal(expected_rank, rank.tocsr().data)
                self.assertEqual(spmat.format, rank.format)
                np.testing.assert_array_equal(spmat.indices, rank.indices)
                np.testing.assert_array_equal(spmat.indptr, rank.indptr)

    def test_rank_axis(self):
        for i in self.mf.iterator():
            a = KarmaSparse(i, format="csr")
            at = a.tocsc()
            for mat in [a, at]:
                for axis in (0, 1):
                    rank = mat.rank(axis)
                    self.assertEqual(mat.shape, rank.shape)
                    np.testing.assert_array_equal(mat.indptr, rank.indptr)
                    self.assertEqual(mat.format, rank.format)
                    np.testing.assert_array_equal(mat.indices, rank.indices)

    def test_truncate_by_count(self):
        ranks = [0, 5]
        for m in self.mf.iterator(dense=True):
            # for the negative values dense and sparse methods gives different result
            a = KarmaSparse(np.abs(m), format="csr")
            at = a.tocsc()
            for mat, nb, axis in itertools.product([a, at], ranks, [0, 1]):
                result = mat.truncate_by_count(axis=axis, nb=nb)
                self.assertEqual(result.format, mat.format)
                if axis == 1:
                    d_result = truncate_by_count_axis1_dense(np.ascontiguousarray(np.abs(m)),
                                                             np.full(m.shape[0], nb, dtype=np.int))
                else:
                    d_result = truncate_by_count_axis1_dense(np.ascontiguousarray(np.abs(m.T)),
                                                             np.full(m.shape[1], nb, dtype=np.int)).T
                np_almost_equal(result.toarray(), d_result)

            for mat, axis in itertools.product([a, at], [0, 1]):
                rank = np.random.randint(0, 5, size=mat.shape[1 - axis])
                if axis == 1:
                    d_result = truncate_by_count_axis1_dense(np.ascontiguousarray(np.abs(m)), rank)
                else:
                    d_result = truncate_by_count_axis1_dense(np.ascontiguousarray(np.abs(m.T)), rank).T
                result = mat.truncate_by_count(axis=axis, nb=rank)
                np_almost_equal(result.toarray(), d_result)
        # test for negative
        m = np.array([[-1., 0., 0., 0., -2., 0., 0.]])
        res = np.array([[-1., 0., 0., 0., 0., 0., 0.]])
        np_almost_equal(KarmaSparse(m).truncate_by_count(axis=1, nb=1).toarray(), res)
        np_almost_equal(KarmaSparse(m).truncate_by_count(axis=0, nb=1).toarray(), m)

    def test_truncate_by_cumulative(self):
        for mat in self.mf.iterator():
            sp_mat = KarmaSparse(mat).tocsr()
            tsp_mat = sp_mat.tocsc()
            for spmat in [sp_mat, tsp_mat]:
                for axis in (0, 1):
                    per = round(random.random(), 2)
                    trunc = spmat.truncate_by_cumulative(percentage=per, axis=axis)
                    self.assertTrue(spmat.shape, trunc.shape)
                    self.assertEqual(spmat.format, trunc.format)
                    # assert all data exists previously
                    self.assertTrue(np.all(np.in1d(trunc.data, spmat.data)))
                    if axis == 1:
                        dense_trunc = trunc.toarray()
                    elif axis == 0:
                        dense_trunc = trunc.transpose().toarray()
                        spmat = spmat.transpose()
                    for idx, trunc_row in enumerate(dense_trunc):
                        mat_row = spmat.toarray()[idx]
                        cutoff = np.sum(mat_row) * (1 - per)
                        tot = 0
                        expected_trunc = np.zeros(len(trunc_row))
                        for elem in np.argsort(mat_row)[::-1]:
                            if mat_row[elem] != 0:
                                tot += mat_row[elem]
                                expected_trunc[elem] = mat_row[elem]
                                if tot >= cutoff:
                                    break
                        np_almost_equal(expected_trunc, trunc_row)

    def test_dot(self):
        for _ in xrange(10):
            n, m, r = np.random.randint(1, 80, size=3)
            mf1 = MatrixFabric(shape=(n, r), density=np.random.rand())
            mf2 = MatrixFabric(shape=(r, m), density=np.random.rand())
            for x, y in zip(mf1.iterator(dense=True), mf2.iterator(dense=True)):
                ks1 = KarmaSparse(x, format="csr")
                ks2 = KarmaSparse(y, format="csr")
                res = ks1.dot(ks2)
                np_almost_equal(res.toarray(), x.dot(y), 5)
                self.assertEqual(res.format, "csr")

                np_almost_equal(ks1.dot(ks1.T).toarray(), x.dot(x.T), 4)

                res = ks1.tocsc().dot(ks2.tocsc())
                np_almost_equal(res.toarray(), x.dot(y), 5)
                self.assertEqual(res.format, "csc")

    def test_div_scalar_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            KarmaSparse(np.array([[0, 2], [4, 1]])) / 0

    def test_div_with_inequal_shape(self):
        with self.assertRaises(ValueError) as e:
            a = KarmaSparse(np.array([[0, 2], [4, 1]]))
            b = KarmaSparse(np.array([[0, 2, 3], [4, 0, 1]]))
            a / b
        self.assertIn('Incompatible shape,', str(e.exception))

    def test_div_scalar(self):
        original_matrix = np.array([[0, 2], [4, 1]])
        expected_matrix = np.array([[0, 1], [2, 0.5]])
        self.assertMatrixEqual(KarmaSparse(original_matrix) / 2,
                               KarmaSparse(expected_matrix))

    def test_div(self):
        original_matrix = np.array([[0, 2], [4, 1]])
        divider_matrix = np.array([[1, 2], [0, 2]])
        expected_matrix = np.array([[0, 1], [0, 0.5]])
        self.assertMatrixEqual(KarmaSparse(original_matrix) / KarmaSparse(divider_matrix),
                               KarmaSparse(expected_matrix))

    def test_idiv(self):
        the_matrix = KarmaSparse(np.array([[0, 2], [4, 1]]))
        expected_matrix = KarmaSparse(np.array([[0, 1], [2, 0.5]]))
        the_matrix /= 2
        self.assertMatrixEqual(the_matrix, expected_matrix)

    def test_div_other_with_zero(self):
        original_matrix = np.array([[0, 2], [4, 1]])
        divider_matrix = KarmaSparse(([1, 2, 0, 2], [0, 1, 0, 1], [0, 2, 4]),
                                     shape=(2, 2), format='csr')
        expected_matrix = np.array([[0, 1], [0, 0.5]])
        self.assertMatrixEqual(KarmaSparse(original_matrix) / divider_matrix,
                               KarmaSparse(expected_matrix))

    # def test_normalize(self):
    #     norms = ('l1', 'l2', 'linf')
    #     axis = (0, 1, None)
    #     is_csr = (True, False)

    #     for norm, ax, matrix, csr in itertools.product(norms, axis, self.mf.iterator(), is_csr):
    #         k = KarmaSparse(matrix)
    #         if csr:
    #             k = k.transpose()
    #             matrix = matrix.transpose()
    #         copy = k.copy()
    #         norm_ks = k.normalize(norm=norm, axis=ax)

    #         # for linf and None axis we use our own normalize method to test
    #         if norm == 'linf' or ax is None:
    #             norm_np = normalize(matrix, norm=norm, axis=ax)
    #         else:
    #             norm_np = sk_normalize(matrix, norm=norm, axis=ax)

    #         np_almost_equal(norm_ks.toarray(), norm_np)

    #         # check that the KS.normalize does not modify KS object
    #         np_almost_equal(copy.toarray(), k.toarray())

    def test_abs(self):
        k = KarmaSparse(np.array([[0., -2., 0., 1.], [0., 0., 4., -1.]]))
        copy = k.copy()
        k_abs = abs(k)
        np_abs = np.array([[0., 2., 0., 1.], [0., 0., 4., 1.]])

        np_almost_equal(k_abs.toarray(), np_abs)
        np_almost_equal(copy.toarray(), k.toarray())

    def test_abs_empty(self):
        k = KarmaSparse(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        copy = k.copy()
        k_abs = abs(k)
        np_abs = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        np_almost_equal(k_abs.toarray(), np_abs)
        np_almost_equal(copy.toarray(), k.toarray())

    def test_add(self):
        for matrix1, matrix2 in itertools.combinations(self.matrixes, 2):
            result = matrix1 + matrix2
            np_almost_equal(KarmaSparse(matrix1) + KarmaSparse(matrix2), result)
            np_almost_equal(KarmaSparse(matrix1, format="csc") + matrix2, result)
            np_almost_equal(KarmaSparse(matrix1) + matrix2, result)
            np_almost_equal(KarmaSparse(matrix1) + matrix2.astype(np.float32), result)
            np_almost_equal(matrix1 + KarmaSparse(matrix2), result)
            np_almost_equal(matrix1 + KarmaSparse(matrix2, format="csc"), result)

            np_almost_equal(KarmaSparse(matrix1) + matrix2[0], matrix1 + matrix2[0])
            np_almost_equal(KarmaSparse(matrix1) + matrix2[0:1,:], matrix1 + matrix2[0:1,:])
            np_almost_equal(KarmaSparse(matrix1) + matrix2[:,0:1], matrix1 + matrix2[:,0:1])

    def test_sum(self):
        axis = (0, 1, None)
        for matrix, axis in itertools.product(self.matrixes, axis):
            np_almost_equal(KarmaSparse(matrix).sum(axis=axis), matrix.sum(axis))

    def test_mean(self):
        axis = (0, 1, None)
        for matrix, axis in itertools.product(self.matrixes, axis):
            np_almost_equal(KarmaSparse(matrix).mean(axis=axis), matrix.mean(axis))

    def test_var(self):
        for matrix, ax, csr in itertools.product(self.matrixes, self.axis, self.is_csr):
            if csr:
                matrix = matrix.transpose()
            np_almost_equal(KarmaSparse(matrix).var(axis=ax), matrix.var(axis=ax))

    def test_std(self):
        for matrix, ax, csr in itertools.product(self.mf.iterator(), self.axis, self.is_csr):
            if csr:
                matrix = matrix.transpose()
            np_almost_equal(KarmaSparse(matrix).std(axis=ax), matrix.std(axis=ax))

    def test_max(self):
        for matrix, ax, csr in itertools.product(self.mf.iterator(), self.axis, self.is_csr):
            if csr:
                matrix = matrix.transpose()
            np_almost_equal(KarmaSparse(matrix).max(axis=ax), matrix.max(axis=ax))

    def test_dense_dot_right(self):
        for _ in xrange(3):
            n, m, r = np.random.randint(1, 80, size=3)
            mf1 = MatrixFabric(shape=(n, r), density=0.6)
            y = np.random.randn(r, m)
            for x in mf1.iterator(dense=True):
                ks1 = KarmaSparse(x, format="csr")
                np_almost_equal(ks1.dense_dot_right(y), x.dot(y))
                ks1 = KarmaSparse(x, format="csc")
                np_almost_equal(ks1.dense_dot_right(y), x.dot(y))

    def test_dense_dot_left(self):
        for _ in xrange(3):
            n, m, r = np.random.randint(1, 80, size=3)
            mf1 = MatrixFabric(shape=(r, n), density=0.6)
            y = np.random.randn(m, r)
            for x in mf1.iterator(dense=True):
                ks1 = KarmaSparse(x, format="csr")
                np_almost_equal(ks1.dense_dot_left(y), y.dot(x))
                ks1 = KarmaSparse(x, format="csc")
                np_almost_equal(ks1.dense_dot_left(y), y.dot(x))

    def test_mask_dot(self):
        n, m, r = 5, 4, 3
        mf1 = MatrixFabric(shape=(n, r), density=0.8)
        mf2 = MatrixFabric(shape=(r, m), density=0.5)
        mat_mask = sp.rand(n, m, 0.3)
        for x, y in zip(mf1.iterator(dense=True), mf2.iterator(dense=True)):
            true_dot = np.zeros((n, m))
            true_dot[mat_mask.nonzero()] = x.dot(y)[mat_mask.nonzero()]
            for my_mat_mask, my_x, my_y in itertools.product(
                [KarmaSparse(mat_mask, format="csr"), KarmaSparse(mat_mask, format="csc")],
                [x, KarmaSparse(x, format="csr"), KarmaSparse(x, format="csc")],
                [y, KarmaSparse(y, format="csr"), KarmaSparse(y, format="csc")]):
                res = my_mat_mask.mask_dot(my_x, my_y)
                np_almost_equal(res.toarray(), true_dot)
                self.assertEqual(res.format, my_mat_mask.format)

    def test_getitem(self):
        for a in self.mf.iterator(dense=True):
            ks = KarmaSparse(a)
            b = sp.csr_matrix(a)
            # single element
            for _ in xrange(10):
                i = random.randint(-a.shape[0], a.shape[0] - 1)
                j = random.randint(-a.shape[1], a.shape[1] - 1)
                np_almost_equal(ks[i, j], a[i, j])
                np_almost_equal(ks.tocsr()[i, j], a[i, j])
                np_almost_equal(ks.tocsc()[i, j], a[i, j])
                np_almost_equal(ks.T[j, i], a[i, j])
            # row slice
            for _ in xrange(15):
                i = random.randint(0, a.shape[0] - 1)
                j = random.randint(0, a.shape[0] - 1)
                k = random.choice([None, 1, 2, 3])
                if random.choice([True, False]):
                    i -= a.shape[0]
                    j -= a.shape[0]
                if i <= j:
                    np_almost_equal(ks[i:j:k].toarray(), a[i:j:k])
                    np_almost_equal(ks.tocsc()[i:j:k].toarray(), a[i:j:k])
                    np_almost_equal(ks.tocsr()[i:j:k].toarray(), a[i:j:k])
                else:
                    self.assertRaisesRegexp(IndexError, "", lambda: ks[i:j])
            # single row
            for _ in xrange(10):
                i = random.randint(-a.shape[0] - 10, a.shape[0] + 10)
                if -a.shape[0] <= i < a.shape[0]:
                    np_almost_equal(ks[i].toarray(), b[i].toarray())
                    np_almost_equal(ks.tocsc()[i].toarray(), b[i].toarray())
                else:
                    self.assertRaisesRegexp(IndexError, "", lambda: ks[i])
            # single col
            for _ in xrange(10):
                i = random.randint(-a.shape[1] - 10, a.shape[1] + 10)
                if -a.shape[1] <= i < a.shape[1]:
                    np_almost_equal(ks[:, i].toarray(), b[:, i].toarray())
                    np_almost_equal(ks.tocsc()[:, i].toarray(), b[:, i].toarray())
                else:
                    self.assertRaisesRegexp(IndexError, "", lambda: ks[:, i])

            # col indices
            for _ in xrange(10):
                ind = [random.randint(-a.shape[1], a.shape[1] - 1) for o in range(20)]
                np_almost_equal(ks[:, ind].toarray(), b[:, ind].toarray())
                np_almost_equal(ks.tocsc()[:, ind].toarray(), b[:, ind].toarray())
                np_almost_equal(ks.T[ind].toarray(), b.T[ind].toarray())

            # col indices
            for _ in xrange(10):
                ind = [random.randint(-a.shape[0], a.shape[0] - 1) for o in range(20)]
                np_almost_equal(ks[ind].toarray(), b[ind].toarray())
                np_almost_equal(ks.tocsc()[ind].toarray(), b[ind].toarray())
                np_almost_equal(ks.T[:, ind].toarray(), b.T[:, ind].toarray())

    def test_median(self):
        for matrix, axis, csr in itertools.product(self.mf.iterator(dense=True),
                                                   self.axis, self.is_csr):
            ks = KarmaSparse(matrix, format="csr" if csr else "csc")
            np_almost_equal(ks.median(axis=axis), np.median(matrix, axis=axis))

        self.assertRaisesRegexp(ValueError,
                                'quantile should be in',
                                lambda: ks.quantile(axis=0, q=2))
        self.assertRaisesRegexp(TypeError,
                                'a float is required',
                                lambda: ks.quantile(axis=0, q="csr"))
        self.assertRaisesRegexp(TypeError,
                                'Axis should be of integer type, got',
                                lambda: ks.quantile(axis="cc", q=0.3))

    def test_complement(self):
        for a in self.mf.iterator(dense=True):
            ks = KarmaSparse(a)
            # an element
            for _ in xrange(20):
                x = np.random.randint(ks.shape[0] - 1)
                y = np.random.randint(ks.shape[1] - 1)
                res = ks.complement(([x], [y]))
                self.assertEqual(res[x, y], 0)
                self.assertEqual(res[x - 1, y - 1], ks[x - 1, y - 1])
            # whole line
            rows = np.repeat(1, ks.shape[1])
            cols = range(ks.shape[1])
            res = ks.complement((rows, cols))
            self.assertTrue(all(elem == 0 for elem
                                in res.toarray()[rows, cols]))
            # diagonal
            res = ks.complement((cols, cols))
            diag_mat = ks_diag(ks.diagonal()).extend(ks.shape)
            np_almost_equal(res.toarray(), (ks - diag_mat).toarray())

    def test_diagonal(self):
        for a in self.mf.iterator(dense=True):
            ks = KarmaSparse(a)
            ks.check_format()
            np_almost_equal(ks.diagonal(), ks.toarray().diagonal())

    def test_pickle(self):
        for mat in self.matrixes:
            spmat = KarmaSparse(mat)
            np_almost_equal(loads(dumps(spmat.tocsr())).toarray(), mat)
            np_almost_equal(loads(dumps(spmat.tocsc())).toarray(), mat)
            np_almost_equal(loads(dumps(spmat.tocsr(), protocol=2)).toarray(), mat)
            np_almost_equal(loads(dumps(spmat.tocsc(), protocol=2)).toarray(), mat)

    def test_load_64pickle(self):
        """
            The matrix in karma_sparse_64.pickle was saved from an environment where they were all in 64bit
            we just want to make sure we can still load them
        """
        ks32 = KarmaSparse(np.array([[0., -2., 0., 1.], [0., 0., 4., 1.]]))
        path = './tests/matrix/karma_sparse_64.pickle'
        with open(path, "r") as _f:
            ks64 = loads(_f.read())
        self.assertEqual(ks32.dtype, ks64.dtype)
        np_almost_equal(ks32, ks64)

    # def test_run_in_subprocess(self):
    #     a = KarmaSparse(sp.rand(100, 100, 0.9, format="csr", random_state=100))

    #     @run_in_subprocess
    #     def ks_from_subprocess():
    #         a.check_format()
    #         a.toarray()
    #         b = a.dot(a).truncate_by_count(5, 1)
    #         b.check_format()
    #         b.toarray()
    #         return a.sum()

    #     _ = a.dot(a)
    #     for _ in xrange(15):
    #         self.assertAlmostEqual(a.sum(), ks_from_subprocess())

    #     with open_mp_threads(2):
    #         for _ in xrange(15):
    #             self.assertAlmostEqual(a.sum(), ks_from_subprocess())

    # def test_run_in_parallel(self):
    #     # to check whether OpenMP is not freezing with multiprocessing
    #     a = KarmaSparse(sp.rand(100, 100, 0.9, format="csr", random_state=100))
    #     with open_mp_threads(2):
    #         _ = a.dot(a)  # to init openmp threads

    #     def ks_lambda(mat):
    #         mat.check_format()
    #         mat.toarray()
    #         b = mat.dot(mat).truncate_by_count(5, 1)
    #         b.check_format()
    #         return round(mat.toarray().sum(), 5)

    #     with open_mp_threads(1):
    #         self.assertEqual(Parallel(4).map(ks_lambda, 10 * [a]), 10 * [round(a.sum(), 5)])
    #     with open_mp_threads(2):
    #         self.assertEqual(Parallel(4).map(ks_lambda, 10 * [a]), 10 * [round(a.sum(), 5)])

    # def test_thread_setter_non_freeze(self):
    #     a = KarmaSparse(sp.rand(10**2, 10**2, 0.01))
    #     a.toarray().shape
    #     a.toarray().sum()
    #     for _ in xrange(5):
    #         with open_mp_threads(2):
    #             a.toarray().shape
    #             a.toarray().sum()

    def test_array_interface(self):
        x = np.arange(6).reshape(2, 3)
        ks = KarmaSparse(np.arange(6).reshape(2, 3))
        y = np.asarray(ks)
        self.assertEqual(y.dtype, DTYPE)
        self.assertTrue(np.all(x == y))

        self.assertEqual(np.array(ks, dtype=np.int32).dtype, np.int32)
        # string types
        z = np.array(ks, dtype='S5')
        zz = np.array([['0.0', '1.0', '2.0'], ['3.0', '4.0', '5.0']], dtype='|S5')
        self.assertTrue(np.all(z == zz))
        self.assertEqual(z.dtype, np.dtype('S5'))
        # other usage cases
        xx = np.random.rand(2, 3)
        xx[:] = ks
        self.assertTrue(np.all(x == xx))
        # ufunc
        xx += ks
        self.assertTrue(np.all(2 * x == xx))

    def test_truncate_by_axis(self):
        for _ in xrange(10):
            a = np.random.rand(5, 100)
            aa = a.copy()
            rank = np.array([3, 1, 0, 6, 10])
            b = truncate_by_count_axis1_dense(a, rank)
            bb = truncate_by_count_axis1_sparse(a, rank)
            np_almost_equal(bb, b)
            self.assertTrue(isinstance(b, np.ndarray))
            self.assertTrue(isinstance(bb, KarmaSparse))
            np_almost_equal(aa, a)
            np_almost_equal(b.astype(np.bool).sum(axis=1), rank)
            np_almost_equal(a[b != 0], b[b != 0])
            np_almost_equal(a.max(axis=1)[rank != 0], b.max(axis=1)[rank != 0])
            arg_sort = a.argsort(axis=1)[:, ::-1]
            for i in xrange(len(a)):
                ind = arg_sort[i, :rank[i]]
                np_almost_equal(a[i, ind], b[i, ind])

    def test_dense_pivot1(self):
        x = np.array([0, 0, 0, 1], dtype=np.int32)
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        val = np.array([12., 7., -1., 5.])
        shape = (2, 2)

        res1 = dense_pivot(x, y, val, shape=shape, aggregator='add', default=-2.4)
        expected1 = np.array([[19., -1.], [-2.4, 5.]])
        np_almost_equal(res1, expected1)

        res_32 = dense_pivot(x, y, val.astype(np.float32), shape=shape, aggregator='add', default=-2.4)
        expected_32 = np.array([[19., -1.], [-2.4, 5.]], dtype=np.float32)
        np.testing.assert_equal(res_32, expected_32)
        self.assertEqual(res_32.dtype, np.float32)

        res2 = dense_pivot(x, y, val, shape=shape, aggregator='max', default=3.3)
        expected2 = np.array([[12., -1.], [3.3, 5.]])
        np_almost_equal(res2, expected2)

        res3 = dense_pivot(x, y, val, shape=shape, aggregator='min', default=np.nan)
        expected3 = np.array([[7., -1.], [np.nan, 5.]])
        np_almost_equal(res3, expected3)

        res4 = dense_pivot(x, y, val, shape=shape, aggregator='first', default=-np.inf)
        expected4 = np.array([[12., -1.], [-np.inf, 5.]])
        np_almost_equal(res4, expected4)

        res5 = dense_pivot(x, y, val, shape=shape, aggregator='last', default=0.)
        expected5 = np.array([[7., -1.], [0., 5.]])
        np_almost_equal(res5, expected5)

    def test_dense_pivot2(self):
        for _ in xrange(50):
            size = np.random.randint(0, 10 ** 3)
            shape = (np.random.randint(1, 100), np.random.randint(1, 100))
            x = np.random.randint(0, shape[0], size).astype(np.int32)
            y = np.random.randint(0, shape[1], size).astype(np.int32)
            val = np.random.rand(size)
            for agg in ['add', 'min', 'max', 'last', 'first']:
                res_dense = dense_pivot(x, y, val, shape=shape, aggregator=agg, default=0.)
                res_sparse = KarmaSparse((val, (x, y)), shape=shape, format="csr", aggregator=agg)
                np_almost_equal(res_dense, res_sparse)

    def test_dot_vector(self):
        for a, dtype in itertools.product(self.mf.iterator(dense=True),
                                          [np.float, np.float32, np.int, np.int32]):
            ks = KarmaSparse(a)
            vec1 = np.random.randn(ks.shape[1]).astype(dtype)
            vec0 = np.random.randn(ks.shape[0]).astype(dtype)
            np_almost_equal(ks.dot(vec1), ks.toarray().dot(vec1))
            np_almost_equal(ks.tocsc().dot(vec1), ks.toarray().dot(vec1))
            np_almost_equal(ks.T.dot(vec0), ks.toarray().T.dot(vec0))
            np_almost_equal(ks.tocsc().T.dot(vec0), ks.toarray().T.dot(vec0))

    def test_agg_multiplication_raises(self):
        a = np.array([[1, -1, 0],
                      [0, -2, -1],
                      [2, 0, 0]], dtype=np.float)
        b = np.array([[0, 1, 3],
                      [-1, 1, 1],
                      [5, 0, 2]], dtype=np.float)
        a_sp = KarmaSparse(a, format='csr')
        b_sp = KarmaSparse(b, format='csr')

        with self.assertRaises(ValueError) as e:
            _ = a_sp.dense_shadow(b)
        self.assertEqual('KarmaSparse contains negative values while only positive are expected',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = a_sp.sparse_shadow(b_sp)
        self.assertEqual('KarmaSparse contains negative values while only positive are expected',
                         str(e.exception))

        a[a < 0] = 0
        a_sp = KarmaSparse(a, format='csr')

        with self.assertRaises(ValueError) as e:
            _ = a_sp.dense_shadow(b)
        self.assertEqual('Numpy matrix contains negative values while only positive are expected',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = a_sp.tocsc().dense_shadow(b)
        self.assertEqual('Numpy matrix contains negative values while only positive are expected',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = a_sp.sparse_shadow(b_sp)
        self.assertEqual('KarmaSparse contains negative values while only positive are expected',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = a_sp.dense_shadow(b, reducer='foo')
        self.assertEqual('Unsupported reducer "foo", choose one from max, add',
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ = a_sp.sparse_shadow(b_sp, reducer='min')
        self.assertEqual('Unsupported reducer "min", choose one from max, add',
                         str(e.exception))

    def test_agg_multiplication(self):
        a = np.array([[1, 1, 0],
                      [1, 2, 1],
                      [2, 0, 0]], dtype=np.float)
        b = np.array([[0, 1, 3],
                      [1, 1, 1],
                      [5, 0, 2]], dtype=np.float)
        a_sp = KarmaSparse(a, format='csr')
        b_sp = KarmaSparse(b, format='csr')

        np_almost_equal(a_sp.dense_shadow(b), [[1, 1, 3], [5, 2, 3], [0, 2, 6]])
        np_almost_equal(a_sp.sparse_shadow(b_sp), [[1, 1, 3], [5, 2, 3], [0, 2, 6]])

        # # FIXME the result should be [[0, 0, 0], [0, 0, 2], [0, 0, 0]]
        # np_almost_equal(a_sp.dense_shadow(b, reducer='min'),
        #                                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # np_almost_equal(a_sp.sparse_shadow(b_sp, reducer='min'),
        #                                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        np_almost_equal(a_sp.dense_shadow(b, reducer='add'), a.dot(b))
        np_almost_equal(a_sp.sparse_shadow(b_sp, reducer='add'), a.dot(b))

    def test_agg_multiplication_random(self):
        for a in self.mf.iterator(dense=True):
            a[a < 0] = 0
            ks = KarmaSparse(a, format='csr')
            for other_matrix in [np.random.randn(*ks.shape).T, 2 * np.random.randint(-1, 2, size=ks.shape).T,
                                 np.random.poisson(0.01, size=ks.shape).T]:
                other_matrix[other_matrix < 0] = 0
                other = KarmaSparse(other_matrix, format='csr')
                for agg in ['max', 'add']:
                    dense_res = ks.dense_shadow(other_matrix, reducer=agg)
                    np_almost_equal(ks.sparse_shadow(other, reducer=agg), dense_res)
                    np_almost_equal(ks.sparse_shadow(other.tocsc(), reducer=agg), dense_res)
                    np_almost_equal(ks.tocsc().sparse_shadow(other, reducer=agg), dense_res)
                    np_almost_equal(ks.tocsc().sparse_shadow(other.tocsc(), reducer=agg), dense_res)

    def test_pointwise_multiplication(self):
        for a in self.mf.iterator(dense=True):
            b1 = np.random.randint(0, 3, size=a.shape)
            b2 = np.random.randint(0, 3, size=a.shape[1])
            b3 = np.random.randint(0, 3, size=(1, a.shape[1]))
            b4 = np.random.randint(0, 3, size=(a.shape[0], 1))

            for b in [b1, b2, b3, b4]:
                expected_result = a * b
                self.assertTrue(is_karmasparse(KarmaSparse(a) * b))
                np_almost_equal(expected_result, KarmaSparse(a) * b)
                np_almost_equal(expected_result, KarmaSparse(a) * b.astype(np.float32))
                np_almost_equal(expected_result, KarmaSparse(a).tocsc() * b)
                np_almost_equal(expected_result, a * KarmaSparse(b))
                np_almost_equal(expected_result, KarmaSparse(a) * KarmaSparse(b))
                np_almost_equal(expected_result, KarmaSparse(b) * KarmaSparse(a))

    def test_scale_along_axis(self):
        for a in self.mf.iterator(dense=False):
            a = KarmaSparse(a)
            for _ in xrange(10):
                w1 = np.random.rand(a.shape[0])
                w2 = np.random.randint(-2, 3, size=a.shape[1])
                expected = w2 * (w1 * a.toarray().T).T
                for mat in [a, a.tocsc()]:
                    result = mat.scale_along_axis(w1, axis=1) \
                        .scale_along_axis(w2, axis=0)
                    np_almost_equal(result, expected)
                    self.assertTrue(isinstance(result, KarmaSparse))

    def test_kronii_test(self):
        for a in self.mf.iterator(dense=True):
            sparse = KarmaSparse(a, copy=False)
            dense = np.random.rand(a.shape[0], np.random.randint(1, 30))
            factor = np.random.rand(a.shape[1] * dense.shape[1])
            expected_result = sparse.kronii(dense).dot(factor)
            expected_result2 = (sparse.kronii(dense) ** 2).dot(factor)
            for b in [dense, dense.astype(np.float32), KarmaSparse(dense)]:
                np_almost_equal(expected_result, sparse.kronii_dot(b, factor), 5)
                np_almost_equal(expected_result2, sparse.kronii_dot(b, factor, 2), 5)

    def test_kronii_test_transpose(self):
        for a in self.mf.iterator(dense=True):
            sparse = KarmaSparse(a, copy=False)
            dense = np.random.rand(a.shape[0], np.random.randint(1, 30))
            factor = np.random.rand(a.shape[0])
            expected_result = sparse.kronii(dense).dense_vector_dot_left(factor)
            expected_result2 = (sparse.kronii(dense) ** 2).dense_vector_dot_left(factor)
            for b in [dense, dense.astype(np.float32), KarmaSparse(dense)]:
                np_almost_equal(expected_result, sparse.kronii_dot_transpose(b, factor))
                np_almost_equal(expected_result2, sparse.kronii_dot_transpose(b, factor, 2))

    def test_kronii_second_moment(self):
        for a in self.mf.iterator(dense=True):
            sparse = KarmaSparse(a, copy=False)
            dense = np.random.rand(a.shape[0], np.random.randint(1, 3))
            c = sparse.kronii(dense)
            expected_result = c.T.dot(c)
            expected_result_dense = np.einsum('ij, ik, il, im -> jklm', a, dense, a, dense)\
                                      .reshape(a.shape[1] * dense.shape[1], -1)
            for b in [dense, dense.astype(np.float32), KarmaSparse(dense)]:
                np_almost_equal(sparse.kronii_second_moment(b), sparse.kronii_second_moment(b).T)
                np_almost_equal(expected_result, sparse.kronii_second_moment(b), 5)
                np_almost_equal(expected_result_dense, sparse.kronii_second_moment(b), 5)

    def test_linear_error(self):
        for dense in self.mf.iterator(dense=True):
            sparse = KarmaSparse(dense, copy=False)
            regressor = np.random.rand(sparse.shape[1], np.random.randint(1, 30))
            target = np.random.rand(sparse.shape[0])
            intercept = np.random.rand(regressor.shape[1])

            expected_result = dense.dot(regressor)
            expected_result += intercept
            expected_result += target[:, np.newaxis]
            expected_result **= 2
            expected_result = array_cast(expected_result.sum(axis=0))

            for b in [sparse, dense, dense.astype(np.float32)]:
                np_almost_equal(expected_result, linear_error(b, regressor, intercept, target), 5)

    def test_apply_apply_pointwise_function(self):
        for dense in self.mf.iterator(dense=True):
            sparse = KarmaSparse(dense, copy=False)
            for method, args, kwargs in [(np.exp, [], {}), (np.power, [2], {}),
                                         (np.clip, [], {'a_min': 0, 'a_max': 1})]:
                expected = method(dense, *args, **kwargs)
                expected[dense == 0] = 0
                actual = sparse.apply_pointwise_function(method, args, kwargs)
                np_almost_equal(actual, expected)

    def test_is_one_hot(self):
        # one hot
        matrix = np.zeros((15, 7))
        idx = np.random.randint(0, 7, 15)
        matrix[np.arange(15), idx] = 1
        one_hot_sparse = KarmaSparse(matrix, format='csr')
        for m, axis in [(one_hot_sparse, 1), (one_hot_sparse.tocsc(), 1),
                        (one_hot_sparse.T, 0), (one_hot_sparse.T.tocsr(), 0)]:
            self.assertTrue(m.is_one_hot(axis))
            self.assertFalse(m.is_one_hot(1 - axis))

        matrix[0, idx[0]] = 2
        not_one_sparse = KarmaSparse(matrix, format='csr')
        for m, axis in [(not_one_sparse, 1), (not_one_sparse.tocsc(), 1),
                        (not_one_sparse.T, 0), (not_one_sparse.T.tocsr(), 0)]:
            self.assertFalse(m.is_one_hot(axis))
            self.assertFalse(m.is_one_hot(1 - axis))

        matrix[0, idx[0]] = 1
        matrix[0, idx[0] - 1] = 1
        not_one_hot_sparse = KarmaSparse(matrix, format='csr')
        for m, axis in [(not_one_hot_sparse, 1), (not_one_hot_sparse.tocsc(), 1),
                        (not_one_hot_sparse.T, 0), (not_one_hot_sparse.T.tocsr(), 0)]:
            self.assertFalse(m.is_one_hot(axis))
            self.assertFalse(m.is_one_hot(1 - axis))

    def test_quantile_boundaries(self):
        data = np.tile([0.78, 0.78, 0.2, 0.87, 0.87, 0.87, 0.5, 0.6, 0.9, 0.9], 5)
        indptr = np.arange(6) * 10
        indices = np.hstack([np.random.choice(100, 10, replace=False) for _ in xrange(5)])
        matrix1 = KarmaSparse((data, indices, indptr), shape=(100, 5), format='csc')

        for m in [matrix1, matrix1.tocsr()]:
            np.testing.assert_equal(m.quantile_boundaries(4, axis=0), array_cast([[0.6] * 5, [0.78] * 5, [0.87] * 5]))
            np.testing.assert_equal(m.T.quantile_boundaries(4, axis=1), array_cast([[0.6, 0.78, 0.87]] * 5))

        data = np.hstack((data, [0.3, 0.5, 0.6]))
        indptr = np.hstack((indptr, [53]))
        indices = np.hstack((indices, np.random.choice(100, 3, replace=False)))
        matrix2 = KarmaSparse((data, indices, indptr), shape=(100, 6), format='csc')

        for m in [matrix2, matrix2.tocsr()]:
            np.testing.assert_equal(m.quantile_boundaries(4, axis=0)[:, -1], array_cast([0.3, 0.5, 0.5]))
            np.testing.assert_equal(m.T.quantile_boundaries(4, axis=1)[-1], array_cast([0.3, 0.5, 0.5]))

        data = np.hstack((data, [1, 2]))
        indptr = np.hstack((indptr, [55]))
        indices = np.hstack((indices, np.random.choice(100, 2, replace=False)))
        matrix3 = KarmaSparse((data, indices, indptr), shape=(100, 7), format='csc')

        for m in [matrix3, matrix3.tocsr()]:
            np.testing.assert_equal(m.quantile_boundaries(4, axis=0)[:, -1], [1, 1, 2])
            np.testing.assert_equal(m.T.quantile_boundaries(4, axis=1)[-1], [1, 1, 2])

        matrix4 = KarmaSparse(np.repeat([0.78, 0.5], 5).reshape(2, 5))
        for m in [matrix4.tocsr(), matrix4.tocsc()]:
            np.testing.assert_equal(m.quantile_boundaries(4, axis=0), [[0.5] * 5, [0.5] * 5])
            np.testing.assert_equal(m.T.quantile_boundaries(4, axis=1), [[0.5, 0.5]] * 5)

        with self.assertRaises(NotImplementedError):
            matrix1.quantile_boundaries(4, axis=None)

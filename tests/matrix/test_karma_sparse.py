import unittest
import doctest
import itertools
import random
from numpy import allclose as eq
from matrix.karma_sparse import KarmaSparse
from matrix.karma_sparse import sp, np, ks_diag
from matrix.karma_sparse import truncate_by_count_axis1_sparse, truncate_by_count_axis1_dense
from sklearn.preprocessing import normalize as sk_normalize
# from karma.learning.matrix_utils import normalize
# from karma.learning.matrix_utils import truncate_by_count
from cPickle import dumps, loads
# from karma.core.utils import run_in_subprocess, Parallel
# from karma.thread_setter import open_mp_threads

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)


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
            self.assertTrue(eq(ks_matrix.count_nonzero(ax), res))

    def assertMatrixEqual(self, compared, expected):
        if not isinstance(compared, KarmaSparse) or not isinstance(expected, KarmaSparse):
            raise Exception('Compared and expected matrix must be KarmaSparse instances')
        self.assertTrue(eq(compared.toarray(), expected.toarray()))

    def test_init_dense(self):
        for a in self.mf.iterator(dense=True):
            self.assertTrue(eq(KarmaSparse(a).toarray(), a))
            self.assertTrue(eq(KarmaSparse(a).tocsc().toarray(), a))
            self.assertTrue(eq(KarmaSparse(a).tocsr().toarray(), a))
            self.assertTrue(eq(KarmaSparse(a.T).toarray(), a.T))
            self.assertTrue(eq(KarmaSparse(a).T.toarray(), a.T))

            # test_init_one_dimension
            self.assertTrue(eq(KarmaSparse(a[0]).toarray(), np.atleast_2d(a[0])))

    def test_init_scipy_sparse(self):
        for a in self.mf.iterator(dense=False):
            self.assertTrue(eq(KarmaSparse(a).toarray(), a.toarray()))
            self.assertTrue(eq(KarmaSparse(a).tocsc().toarray(), a.toarray()))
            self.assertTrue(eq(KarmaSparse(a.T).toarray(), a.toarray().T))
            self.assertTrue(eq(KarmaSparse(a).T.toarray(), a.toarray().T))

    def test_init_karma_sparse(self):
        for a in self.mf.iterator(dense=False):
            self.assertTrue(eq(KarmaSparse(KarmaSparse(a)).toarray(), a.toarray()))
            self.assertTrue(eq(KarmaSparse(KarmaSparse(a)).tocsc().toarray(), a.toarray()))
            self.assertTrue(eq(KarmaSparse(KarmaSparse(a).T).toarray(), a.toarray().T))
            self.assertTrue(eq(KarmaSparse(KarmaSparse(a.T)).toarray(), a.toarray().T))

    def test_init_coo(self):
        for a in self.mf.iterator(dense=False):
            a = a.tocoo()
            self.assertTrue(eq(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape).toarray(),
                               a.toarray()))
            self.assertTrue(eq(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape,
                                           format="csc").toarray(),
                               a.toarray()))

        a = KarmaSparse((np.array([0]), np.array([0])), shape=(1, 1), format='csr')
        self.assertTrue(a.toarray(), np.array([[1]]))

    def test_init_mask_coo(self):
        for a in self.mf.iterator(dense=False):
            a = a.tocoo()
            b = a.copy()
            b.data[:] = 1.
            self.assertTrue(eq(KarmaSparse((a.row, a.col), shape=a.shape).toarray(),
                               b.toarray()))
            self.assertTrue(eq(KarmaSparse((a.row, a.col), shape=a.shape,
                                           format="csc").toarray(),
                               b.toarray()))

    def test_init_nondeduplicated_coo(self):
        for a in self.mf.iterator(dense=False):
            a = a.tocoo().copy()
            a.row = a.row % 2
            a.col = a.col % 3
            np.random.shuffle(a.row)
            np.random.shuffle(a.col)
            self.assertTrue(eq(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape).toarray(),
                               a.toarray()))
            self.assertTrue(eq(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape, format="csc").toarray(),
                               a.toarray()))
            # filter out nan values
            a.data[:] = np.nan
            ks = KarmaSparse((a.data, (a.row, a.col)), shape=a.shape)
            self.assertEqual(ks.nnz, 0)

            # only one element
            a.row[:] = 0
            a.col[:] = 0
            a.data[:] = np.random.rand(a.data.shape[0])
            ks = KarmaSparse((a.data, (a.row, a.col)), shape=a.shape)
            self.assertAlmostEqual(ks[0, 0], a.data.sum())
            self.assertAlmostEqual(ks.sum(), a.data.sum())
            self.assertTrue(eq(ks.nnz, a.data.shape[0] > 0))

    def test_init_aggregator_coo(self):
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 1, 0, 0])
        values = np.array([2, 2, -1, -10])

        ks = KarmaSparse((x, y), shape=(2, 2))
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), 4)
        self.assertTrue(eq(ks.toarray(), np.array([[2, 0], [1, 1]])))

        ks = KarmaSparse((x, y), shape=(2, 2), aggregator="max")
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), 3)
        self.assertTrue(eq(ks.toarray(), np.array([[1, 0], [1, 1]])))

        ks = KarmaSparse((values, (x, y)), shape=(2, 2), aggregator="max")
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), 3)
        self.assertTrue(eq(ks.toarray(), np.array([[2, 0], [-1, 2]])))

        ks = KarmaSparse((values, (x, y)), shape=(2, 2), aggregator="min")
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), -9)
        self.assertTrue(eq(ks.toarray(), np.array([[-10, 0], [-1, 2]])))

        ks = KarmaSparse((values, (x, y)), shape=(2, 2), aggregator="multiply")
        self.assertEqual(ks.nnz, 3)
        self.assertEqual(ks.sum(), -19)
        self.assertTrue(eq(ks.toarray(), np.array([[-20, 0], [-1, 2]])))

    def test_init_zeros(self):
        for a in self.mf.iterator(dense=False):
            self.assertTrue(eq(KarmaSparse(a.shape).toarray(), np.zeros(a.shape)))
            self.assertTrue(eq(KarmaSparse(a.shape, format="csr").toarray(), np.zeros(a.shape)))
            self.assertTrue(eq(KarmaSparse(a.shape, format="csc").toarray(), np.zeros(a.shape)))

    def test_format_respect(self):
        a = sp.rand(4, 2, 0.3, format='coo', random_state=1208)
        # sparse
        self.assertTrue(eq(KarmaSparse(a).toarray(), a.toarray()))
        self.assertEqual(KarmaSparse(a, format="csc").format, 'csc')
        self.assertEqual(KarmaSparse(a, format="csr").format, 'csr')

        # dense
        self.assertTrue(eq(KarmaSparse(a.toarray()).toarray(), a.toarray()))
        self.assertEqual(KarmaSparse(a.toarray(), format="csc").format, 'csc')
        self.assertEqual(KarmaSparse(a.toarray(), format="csr").format, 'csr')

        # KS
        self.assertEqual(KarmaSparse(KarmaSparse(a), format="csc").format, 'csc')
        self.assertEqual(KarmaSparse(KarmaSparse(a), format="csr").format, 'csr')
        # coo
        self.assertTrue(eq(KarmaSparse((a.data, (a.row, a.col)), shape=a.shape).toarray(),
                           a.toarray()))
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
            self.assertTrue(eq(ks.tocsr().toarray(), a))
            self.assertTrue(eq(ks.tocsc().toarray(), ks.tocsr().toarray()))

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
                self.assertTrue(np.array_equal(b.indices, mat.indices))
                self.assertTrue(np.array_equal(b.data, mat.data))
                self.assertEqual(b.format, mat.format)
                # others change
                if mat.format == 'csr':
                    indptr_complete = axis_0
                else:
                    indptr_complete = axis_1
                self.assertEqual(b.shape, new_shape)
                new_indptr = np.hstack([mat.indptr, [mat.indptr[-1]] * indptr_complete])
                self.assertTrue(np.array_equal(b.indptr, new_indptr))
                new_mat = np.column_stack([mat.toarray(), np.zeros((mat.shape[0], axis_1))])
                new_mat = np.vstack([new_mat, np.zeros((axis_0, mat.shape[1] + axis_1))])
                self.assertTrue(eq(b.toarray(), new_mat))
                # check data is really copied
                init_data = mat.data.copy()
                for i in xrange(len(mat.data)):
                    mat.data[i] = np.random.random()
                self.assertTrue(np.array_equal(b.data, init_data))
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

        self.assertTrue(eq(ks.argmax(1, False), b.argmax(1)))
        self.assertTrue(eq(ks.argmin(1, False), b.argmin(1)))
        self.assertTrue(not eq(ks.argmin(1, True), b.argmin(1)))

        # self.assertTrue(eq(ks.argmax(0, False), b.argmax(0)))
        self.assertTrue(eq(ks.argmin(0, False), b.argmin(0)))

    def test_rank_axis_none(self):
        for i in self.mf.iterator(dense=True):
            a = KarmaSparse(i, format="csr")
            at = a.tocsc()
            for spmat in [a, at]:
                rank = spmat.rank(axis=None)
                flat_mat = np.reshape(i, (1, i.size))
                flat_non_zero_mat = flat_mat[np.nonzero(flat_mat)]
                expected_rank = flat_non_zero_mat.argsort(axis=0).argsort(axis=0) + 1
                self.assertTrue(eq(expected_rank, rank.tocsr().data))
                self.assertEqual(spmat.format, rank.format)
                self.assertTrue(np.array_equal(spmat.indices, rank.indices))
                self.assertTrue(np.array_equal(spmat.indptr, rank.indptr))

    def test_rank_axis(self):
        for i in self.mf.iterator():
            a = KarmaSparse(i, format="csr")
            at = a.tocsc()
            for mat in [a, at]:
                for axis in (0, 1):
                    rank = mat.rank(axis)
                    self.assertEqual(mat.shape, rank.shape)
                    self.assertTrue(np.array_equal(mat.indptr, rank.indptr))
                    self.assertEqual(mat.format, rank.format)
                    self.assertTrue(np.array_equal(mat.indices, rank.indices))

    # def test_truncate_by_count(self):
    #     ranks = [0, 5]
    #     for m in self.mf.iterator(dense=True):
    #         # for the negative values dense and sparse methods gives different result
    #         a = KarmaSparse(np.abs(m), format="csr")
    #         at = a.tocsc()
    #         for mat, nb, axis in itertools.product([a, at], ranks, [0, 1]):
    #             result = mat.truncate_by_count(axis=axis, nb=nb)
    #             self.assertEqual(result.format, mat.format)
    #             self.assertTrue(eq(result.toarray(),
    #                             truncate_by_count(np.abs(m), axis=axis, max_rank=nb)))
    #     # test for negative
    #     m = np.array([[-1., 0., 0., 0., -2., 0., 0.]])
    #     res = np.array([[-1., 0., 0., 0., 0., 0., 0.]])
    #     self.assertTrue(eq(KarmaSparse(m).truncate_by_count(axis=1, nb=1).toarray(), res))
    #     self.assertTrue(eq(KarmaSparse(m).truncate_by_count(axis=0, nb=1).toarray(), m))

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
                        self.assertTrue(eq(expected_trunc, trunc_row))

    def test_dot(self):
        for _ in xrange(10):
            n, m, r = np.random.randint(1, 80, size=3)
            mf1 = MatrixFabric(shape=(n, r), density=np.random.rand())
            mf2 = MatrixFabric(shape=(r, m), density=np.random.rand())
            for x, y in zip(mf1.iterator(dense=True), mf2.iterator(dense=True)):
                ks1 = KarmaSparse(x, format="csr")
                ks2 = KarmaSparse(y, format="csr")
                res = ks1.dot(ks2)
                self.assertTrue(eq(res.toarray(), x.dot(y)))
                self.assertEqual(res.format, "csr")

                self.assertTrue(eq(ks1.dot(ks1.T).toarray(), x.dot(x.T)))

                res = ks1.tocsc().dot(ks2.tocsc())
                self.assertTrue(eq(res.toarray(), x.dot(y)))
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

    #         self.assertTrue(eq(norm_ks.toarray(), norm_np))

    #         # check that the KS.normalize does not modify KS object
    #         self.assertTrue(eq(copy.toarray(), k.toarray()))

    def test_abs(self):
        k = KarmaSparse(np.array([[0., -2., 0., 1.], [0., 0., 4., -1.]]))
        copy = k.copy()
        k_abs = abs(k)
        np_abs = np.array([[0., 2., 0., 1.], [0., 0., 4., 1.]])

        self.assertTrue(eq(k_abs.toarray(), np_abs))
        self.assertTrue(eq(copy.toarray(), k.toarray()))

    def test_abs_empty(self):
        k = KarmaSparse(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        copy = k.copy()
        k_abs = abs(k)
        np_abs = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        self.assertTrue(eq(k_abs.toarray(), np_abs))
        self.assertTrue(eq(copy.toarray(), k.toarray()))

    def test_add(self):
        for matrix1, matrix2 in itertools.combinations(self.matrixes, 2):
            self.assertTrue(eq((KarmaSparse(matrix1) + KarmaSparse(matrix2)).toarray(), matrix1 + matrix2))

    def test_sum(self):
        axis = (0, 1, None)
        for matrix, axis in itertools.product(self.matrixes, axis):
            self.assertTrue(eq(KarmaSparse(matrix).sum(axis=axis), matrix.sum(axis)))

    def test_mean(self):
        axis = (0, 1, None)
        for matrix, axis in itertools.product(self.matrixes, axis):
            self.assertTrue(eq(KarmaSparse(matrix).mean(axis=axis), matrix.mean(axis)))

    def test_var(self):
        for matrix, ax, csr in itertools.product(self.matrixes, self.axis, self.is_csr):
            if csr:
                matrix = matrix.transpose()
            self.assertTrue(eq(KarmaSparse(matrix).var(axis=ax), matrix.var(axis=ax)))

    def test_std(self):
        for matrix, ax, csr in itertools.product(self.mf.iterator(), self.axis, self.is_csr):
            if csr:
                matrix = matrix.transpose()
            self.assertTrue(eq(KarmaSparse(matrix).std(axis=ax), matrix.std(axis=ax)))

    def test_max(self):
        for matrix, ax, csr in itertools.product(self.mf.iterator(), self.axis, self.is_csr):
            if csr:
                matrix = matrix.transpose()
            self.assertTrue(eq(KarmaSparse(matrix).max(axis=ax), matrix.max(axis=ax)))

    def test_dense_dot_right(self):
        for _ in xrange(3):
            n, m, r = np.random.randint(1, 80, size=3)
            mf1 = MatrixFabric(shape=(n, r), density=0.6)
            y = np.random.randn(r, m)
            for x in mf1.iterator(dense=True):
                ks1 = KarmaSparse(x, format="csr")
                self.assertTrue(eq(ks1.dense_dot_right(y), x.dot(y)))
                ks1 = KarmaSparse(x, format="csc")
                self.assertTrue(eq(ks1.dense_dot_right(y), x.dot(y)))

    def test_dense_dot_left(self):
        for _ in xrange(3):
            n, m, r = np.random.randint(1, 80, size=3)
            mf1 = MatrixFabric(shape=(r, n), density=0.6)
            y = np.random.randn(m, r)
            for x in mf1.iterator(dense=True):
                ks1 = KarmaSparse(x, format="csr")
                self.assertTrue(eq(ks1.dense_dot_left(y), y.dot(x)))
                ks1 = KarmaSparse(x, format="csc")
                self.assertTrue(eq(ks1.dense_dot_left(y), y.dot(x)))

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
                self.assertTrue(eq(res.toarray(), true_dot))
                self.assertEqual(res.format, my_mat_mask.format)

    def test_getitem(self):
        for a in self.mf.iterator(dense=True):
            ks = KarmaSparse(a)
            b = sp.csr_matrix(a)
            # single element
            for _ in xrange(10):
                i = random.randint(-a.shape[0], a.shape[0] - 1)
                j = random.randint(-a.shape[1], a.shape[1] - 1)
                self.assertEqual(ks[i, j], a[i, j])
                self.assertEqual(ks.tocsr()[i, j], a[i, j])
                self.assertEqual(ks.tocsc()[i, j], a[i, j])
                self.assertEqual(ks.T[j, i], a[i, j])
            # row slice
            for _ in xrange(15):
                i = random.randint(0, a.shape[0] - 1)
                j = random.randint(0, a.shape[0] - 1)
                k = random.choice([None, 1, 2, 3])
                if random.choice([True, False]):
                    i -= a.shape[0]
                    j -= a.shape[0]
                if i <= j:
                    self.assertTrue(eq(ks[i:j:k].toarray(), a[i:j:k]))
                    self.assertTrue(eq(ks.tocsc()[i:j:k].toarray(), a[i:j:k]))
                    self.assertTrue(eq(ks.tocsr()[i:j:k].toarray(), a[i:j:k]))
                else:
                    self.assertRaisesRegexp(IndexError, "", lambda: ks[i:j])
            # single row
            for _ in xrange(10):
                i = random.randint(-a.shape[0] - 10, a.shape[0] + 10)
                if -a.shape[0] <= i < a.shape[0]:
                    self.assertTrue(eq(ks[i].toarray(), b[i].toarray()))
                    self.assertTrue(eq(ks.tocsc()[i].toarray(), b[i].toarray()))
                else:
                    self.assertRaisesRegexp(IndexError, "", lambda: ks[i])
            # single col
            for _ in xrange(10):
                i = random.randint(-a.shape[1] - 10, a.shape[1] + 10)
                if -a.shape[1] <= i < a.shape[1]:
                    self.assertTrue(eq(ks[:, i].toarray(), b[:, i].toarray()))
                    self.assertTrue(eq(ks.tocsc()[:, i].toarray(), b[:, i].toarray()))
                else:
                    self.assertRaisesRegexp(IndexError, "", lambda: ks[:, i])

            # col indices
            for _ in xrange(10):
                ind = [random.randint(-a.shape[1], a.shape[1] - 1) for o in range(20)]
                self.assertTrue(eq(ks[:, ind].toarray(), b[:, ind].toarray()))
                self.assertTrue(eq(ks.tocsc()[:, ind].toarray(), b[:, ind].toarray()))
                self.assertTrue(eq(ks.T[ind].toarray(), b.T[ind].toarray()))

            # col indices
            for _ in xrange(10):
                ind = [random.randint(-a.shape[0], a.shape[0] - 1) for o in range(20)]
                self.assertTrue(eq(ks[ind].toarray(), b[ind].toarray()))
                self.assertTrue(eq(ks.tocsc()[ind].toarray(), b[ind].toarray()))
                self.assertTrue(eq(ks.T[:, ind].toarray(), b.T[:, ind].toarray()))

    def test_median(self):
        for matrix, axis, csr in itertools.product(self.mf.iterator(dense=True),
                                                   self.axis, self.is_csr):
            ks = KarmaSparse(matrix, format="csr" if csr else "csc")
            self.assertTrue(eq(ks.median(axis=axis), np.median(matrix, axis=axis)))

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
            self.assertTrue(eq(res.toarray(),
                            (ks - diag_mat).toarray()))

    def test_diagonal(self):
        for a in self.mf.iterator(dense=True):
            ks = KarmaSparse(a)
            ks.check_format()
            self.assertTrue(eq(ks.diagonal(), ks.toarray().diagonal()))

    def test_pickle(self):
        for mat in self.matrixes:
            spmat = KarmaSparse(mat)
            self.assertTrue(eq(loads(dumps(spmat.tocsr())).toarray(), mat))
            self.assertTrue(eq(loads(dumps(spmat.tocsc())).toarray(), mat))
            self.assertTrue(eq(loads(dumps(spmat.tocsr(), protocol=2)).toarray(), mat))
            self.assertTrue(eq(loads(dumps(spmat.tocsc(), protocol=2)).toarray(), mat))

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
        self.assertEqual(y.dtype, np.float64)
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
            self.assertTrue(eq(bb, b))
            self.assertTrue(isinstance(b, np.ndarray))
            self.assertTrue(isinstance(bb, KarmaSparse))
            self.assertTrue(eq(aa, a))
            self.assertTrue(eq(b.astype(np.bool).sum(axis=1), rank))
            self.assertTrue(eq(a[b != 0], b[b != 0]))
            self.assertTrue(eq(a.max(axis=1)[rank != 0], b.max(axis=1)[rank != 0]))
            arg_sort = a.argsort(axis=1)[:, ::-1]
            for i in xrange(len(a)):
                ind = arg_sort[i, :rank[i]]
                self.assertTrue(eq(a[i, ind], b[i, ind]))

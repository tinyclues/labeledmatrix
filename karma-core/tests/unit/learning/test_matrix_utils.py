import unittest
import numpy as np
from karma.core.utils.utils import use_seed
from scipy.sparse import rand
from cyperf.matrix.karma_sparse import KarmaSparse
from karma.learning.matrix_utils import gram_quantiles

class MatrixUtilsTestCase(unittest.TestCase):
    def test_gram_quantiles(self):
        with use_seed(42):
            sparse_G = KarmaSparse(rand(100, 10, 0.1))
        np.testing.assert_almost_equal(gram_quantiles(sparse_G, 0.1), np.array([0.]))
        np.testing.assert_almost_equal(gram_quantiles(sparse_G, 0.9),
                                       np.array([ 0.20081272]))
        np.testing.assert_almost_equal(gram_quantiles(sparse_G, [0.1, 0.9]),
                                       np.array([ 0., 0.20081272]))

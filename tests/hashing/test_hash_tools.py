import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cyperf.hashing import (hash_numpy_string, hash_numpy_string_with_many_seeds, hasher_numpy_string,
                            randomizer_string, increment_over_numpy_string)


class HashToolsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1234567)
        cls.string_array = np.random.randint(97, 123, size=(10, 8)).astype('uint8').view('S8')[:, 0]

    @classmethod
    def tearDownClass(cls):
        np.random.seed(None)

    def test_hash_numpy_string(self):
        actual = hash_numpy_string(self.string_array, 1234)
        expected = np.asarray([1317099945, 2793735997, 3694782527, 570304382, 3393746913,
                               2686037843, 2288594406, 1035344568, 297607909, 1693388205], dtype='uint32')
        assert_array_equal(expected, actual)

    def test_hash_numpy_string_with_many_seeds(self):
        actual = hash_numpy_string_with_many_seeds(self.string_array, (1, 2, 3, 4))
        expected = np.asarray([[142871393, 621915129, 752048583, 1850162155],
                               [1983106724, 2421898445, 2920017667, 4011385710],
                               [435505789, 1154401381, 1406030201, 3053319577],
                               [2355580926, 3258510167, 3516800653, 1306164583],
                               [3180565836, 2109580468, 1018165361, 2244821960],
                               [1767866879, 4161758788, 2100311021, 2688001248],
                               [2615956031, 4004313904, 1643549433, 1007161801],
                               [1895094328, 3163203076, 4050748942, 2992326503],
                               [3020603039, 473885690, 2835359068, 1856770423],
                               [1843225487, 2478737487, 145633316, 694184351]], dtype='int')
        assert_array_equal(expected, actual)

    def test_hasher_numpy_string(self):
        actual = hasher_numpy_string(self.string_array, 4, 1234)
        expected = [1, 1, 3, 2, 1, 3, 2, 0, 1, 1]
        assert_array_equal(expected, actual)

    def test_randomizer_numpy_string(self):
        actual = randomizer_string(self.string_array, (1, 1, 2), 1234)
        expected = [1, 1, 2, 2, 1, 2, 2, 0, 1, 1]
        assert_array_equal(expected, actual)

    def test_randomizer_python_string(self):
        actual = randomizer_string(self.string_array.tolist(), (1, 1, 2), 1234)
        expected = [1, 1, 2, 2, 1, 2, 2, 0, 1, 1]
        assert_array_equal(expected, actual)

    def test_increment_over_numpy_string_1d(self):
        actual = increment_over_numpy_string(self.string_array,
                                             [0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                             np.arange(10.) + 1.,
                                             [1, 2, 3, 4],
                                             (80, 20),
                                             4)
        expected = np.asarray([[0.8, -0.2, 0.8, -0.2],
                               [1.2, 1.2, -1.8, 5.2],
                               [-3.6, 2.4, -3.6, -3.6],
                               [4.6, 21.6, -5.4, -5.4]])
        assert_array_almost_equal(expected, actual)

    def test_increment_over_numpy_string_2d(self):
        actual = increment_over_numpy_string(self.string_array,
                                             [0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                             (np.arange(30.) + 1.).reshape(10, 3),
                                             [1, 2, 3, 4],
                                             (80, 20),
                                             4)
        expected = np.asarray([[[  0.8,  1.6,  2.4],
                                [ -0.2, -0.4, -0.6],
                                [  0.8,  1.6,  2.4],
                                [ -0.2, -0.4, -0.6]],
                               [[  2.8,  3.2,  3.6],
                                [  2.8,  3.2,  3.6],
                                [ -4.2, -4.8, -5.4],
                                [ 12.8, 14.2, 15.6]],
                               [[ -9.6,-10.2,-10.8],
                                [  6.4,  6.8,  7.2],
                                [ -9.6,-10.2,-10.8],
                                [ -9.6,-10.2,-10.8]],
                               [[ 13.,  13.4, 13.8],
                                [ 60.,  62.4, 64.8],
                                [-15., -15.6,-16.2],
                                [-15., -15.6,-16.2]]])
        assert_array_almost_equal(expected, actual)

import unittest
import doctest


class testDoctest(unittest.TestCase):

    def test_indexed_list(self):
        import perf.indexing.indexed_list
        self.assertEqual(doctest.testmod(perf.indexing.indexed_list).failed, 0)

    def test_karma_sparse(self):
        import perf.matrix.karma_sparse
        self.assertEqual(doctest.testmod(perf.matrix.karma_sparse).failed, 0)

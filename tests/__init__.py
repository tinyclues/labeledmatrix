import unittest
import doctest


class testDoctest(unittest.TestCase):

    def test_indexed_list(self):
        import perf.indexing.indexed_list
        self.assertEqual(doctest.testmod(perf.indexing.indexed_list).failed, 0)

    def test_karma_sparse(self):
        import perf.matrix.karma_sparse
        self.assertEqual(doctest.testmod(perf.matrix.karma_sparse).failed, 0)

    def test_routine(self):
        import perf.matrix.routine
        self.assertEqual(doctest.testmod(perf.matrix.routine).failed, 0)

    def test_sort_tools(self):
        import perf.tools.sort_tools
        self.assertEqual(doctest.testmod(perf.tools.sort_tools).failed, 0)

    def test_space_tools(self):
        import perf.clustering.space_tools
        self.assertEqual(doctest.testmod(perf.clustering.space_tools).failed, 0)

    def test_getter(self):
        import perf.tools.getter
        self.assertEqual(doctest.testmod(perf.tools.getter).failed, 0)

    def test_curve(self):
        import perf.tools.curve
        self.assertEqual(doctest.testmod(perf.tools.curve).failed, 0)

    def test_doctest(self):
        import perf.clustering.hierarchical
        self.assertEqual(doctest.testmod(perf.clustering.hierarchical).failed, 0)

    def test_doctest_types(self):
        import perf.tools.types
        self.assertEqual(doctest.testmod(perf.tools.types).failed, 0)

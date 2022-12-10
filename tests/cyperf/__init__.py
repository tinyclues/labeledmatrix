import unittest
import doctest


class testDoctest(unittest.TestCase):

    def test_indexed_list(self):
        import cyperf.indexing.indexed_list
        self.assertEqual(doctest.testmod(cyperf.indexing.indexed_list).failed, 0)

    def test_karma_sparse(self):
        import cyperf.matrix.karma_sparse
        self.assertEqual(doctest.testmod(cyperf.matrix.karma_sparse).failed, 0)

    def test_argmax_dispatch(self):
        import cyperf.matrix.argmax_dispatch
        self.assertEqual(doctest.testmod(cyperf.matrix.argmax_dispatch).failed, 0)

    def test_routine(self):
        import cyperf.matrix.routine
        self.assertEqual(doctest.testmod(cyperf.matrix.routine).failed, 0)

    def test_sort_tools(self):
        import cyperf.tools.sort_tools
        self.assertEqual(doctest.testmod(cyperf.tools.sort_tools).failed, 0)

    def test_space_tools(self):
        import cyperf.clustering.space_tools
        self.assertEqual(doctest.testmod(cyperf.clustering.space_tools).failed, 0)

    def test_getter(self):
        import cyperf.tools.getter
        self.assertEqual(doctest.testmod(cyperf.tools.getter).failed, 0)

    def test_curve(self):
        import cyperf.tools.curve
        self.assertEqual(doctest.testmod(cyperf.tools.curve).failed, 0)

    def test_doctest(self):
        import cyperf.clustering.hierarchical
        self.assertEqual(doctest.testmod(cyperf.clustering.hierarchical).failed, 0)

    def test_doctest_types(self):
        import cyperf.tools.types
        self.assertEqual(doctest.testmod(cyperf.tools.types).failed, 0)

    def test_doctest_indexing(self):
        import cyperf.indexing
        self.assertEqual(doctest.testmod(cyperf.indexing).failed, 0)

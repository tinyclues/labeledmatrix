#
# Copyright tinyclues, All rights reserved
#

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster.hierarchical import AgglomerativeClustering as Ward
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import normalized_mutual_info_score
import operator

from perf.clustering.hierarchical import WardTree, traversal, huffman_encoding, huffman_encoding_reordering
from perf.clustering.space_tools import (pairwise_flat, vector_distance, METRIC_LIST,
                                         pairwise_square, fast_buddies)


class testHierarchical(unittest.TestCase):

    def test_ward_tree(self):
        X = np.array([[0.92, 0.1], [1.0, 0.0], [0.1, 0.9],
                      [0.0, 1.0], [0.51, 0.49]])
        wt = WardTree(X)
        mat = np.array([[0., 1., 0.12806248, 2.],
                        [2., 3., 0.14142136, 2.],
                        [4., 5., 0.72672783, 3.],
                        [6., 7., 1.65778969, 5.]])
        assert_array_almost_equal(wt.build_linkage(False), mat)

        l1 = WardTree(X, n_clusters=3).build_labels()
        assert_array_equal(l1, np.array([1, 1, 3, 3, 4]))

        # build_labels returns labels of clusters.
        X = np.random.rand(20, 4)
        l1 = WardTree(X, n_clusters=3).build_labels()
        l2 = Ward(n_clusters=3).fit(X).labels_
        self.assertAlmostEqual(normalized_mutual_info_score(l1, l2), 1.)

        # No weights yields same output as Scipy's linkage.
        X = np.random.rand(20, 5)
        wt = WardTree(X)
        link = wt.build_linkage(False)
        hc_link = hc.linkage(X, method='ward')
        assert_array_almost_equal(link, hc_link)

        # Testing chain method
        link_chain = WardTree(X).build_linkage()  # not exactly the same order
        self.assertTrue(np.allclose(link_chain[-1, 2:], hc_link[-1, 2:]))
        # Two *ordered* tree should be the same (up to inversion)
        d = pairwise_flat(X)
        hc1 = huffman_encoding_reordering(link_chain, d)
        hc2 = huffman_encoding_reordering(link, d)
        ll1 = sorted(hc1.iteritems(), key=operator.itemgetter(1))
        ll2 = sorted(hc2.iteritems(), key=operator.itemgetter(1))
        ord1 = np.array([x for x, v in ll1])
        ord2 = np.array([x for x, v in ll2])
        self.assertTrue(np.all(ord1 == ord2) or np.all(ord1 == ord2[::-1]))
        # No weights is equivalent to setting weights = np.ones(n)
        wtw = WardTree(X, weights=np.ones(20))
        assert_array_almost_equal(link, wtw.build_linkage(False))

        # Testing correctness of weights.
        # Constant redundancy is the same as constant weights.
        Y = np.array(list(np.random.rand(100))*3).reshape([75, 4])
        wt = WardTree(Y[:25], weights=[3]*25)
        link = wt.build_linkage(False)
        assert_array_almost_equal(link[:, 2:], hc.linkage(Y, method="ward")[-24:,2:])
        # Other input.
        Y = np.array([[0., 0.], [0.9, 0.0], [1.1, 1.1],
                      [0.0, 1.0], [0.0, 1.1]])
        standard_link = WardTree(Y).build_linkage(False)
        mat = np.array([[ 3.,  4.,  0.1       ,  2.],
                        [ 0.,  1.,  0.9       ,  2.],
                        [ 2.,  5.,  1.27148207,  3.],
                        [ 6.,  7.,  1.65750817,  5.]])
        assert_array_almost_equal(standard_link, mat)

        # Here adding weights changes the output.
        link_with_weights = WardTree(Y, weights=[40] * 2 + [1] * 3).build_linkage(False)
        self.assertTrue((np.sum(link_with_weights[:, :2] != standard_link[:, 2:]) > 0))

        # Same output with redundancy.
        Z = np.zeros([83, 2])
        Z[:80] = np.array([[0., 0.] * 40, [0.9, 0.0] * 40]).reshape(80, 2)
        Z[-3:] = np.array([[1.1, 1.1], [0.0, 1.0], [0.0, 1.1]])
        link_with_redundancy = hc.ward(Z)[-4:, 2:]
        assert_array_almost_equal(link_with_weights[:, 2:], link_with_redundancy)

        # Test using synthetic data.
        centers = [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, -1, -1]]
        X, labels_true = make_blobs(n_samples=100, centers=centers,
                                    cluster_std=0.4, random_state=0)
        labels_pred = WardTree(X, n_clusters=3).build_labels()
        self.assertAlmostEqual(normalized_mutual_info_score(labels_true, labels_pred), 1.)

        # Test of huffman_encoding
        Y = np.array([[0.92, 0.1], [1.0, 0.0], [0.1, 0.9],
                      [0.0, 1.0], [0.51, 0.49]])
        d = huffman_encoding(WardTree(Y).build_linkage(False))
        self.assertEqual(d, {0: '110', 1: '111', 2: '00', 3: '01', 4: '10'})
        d = WardTree(Y).build_huffman_ordering()
        self.assertEqual(d, {0: '110', 1: '111', 2: '01', 3: '00', 4: '10'})

        # Test transversal
        self.assertEqual(traversal(WardTree(Y).build_linkage().astype(np.int32), 8),
                         [1, 0, 4, 3, 2])
        self.assertEqual(traversal(WardTree(Y).build_linkage().astype(np.int32), 6),
                         [3, 2])

        # Test ordering exactness
        Y = np.random.rand(20, 1)  # one dim vectors with natural order
        ll = sorted(WardTree(Y).build_huffman_ordering().iteritems(),
                    key=operator.itemgetter(1))
        ord1 = np.array([x for x, v in ll])
        ord2 = np.argsort(Y.T, axis=1)[0]
        self.assertTrue(np.all(ord1 == ord2) or np.all(ord1 == ord2[::-1]))

    def test_vector_distance(self):
        a = np.random.rand(10, 3)
        v = np.random.rand(1, 3)
        for m in METRIC_LIST:
            assert_array_almost_equal(vector_distance(v[0], a, m), cdist(v, a, m)[0])

        assert_array_almost_equal(vector_distance(v[0], a, 'idiv'),
                                  np.sum(v[0] * np.log(v[0] / a), axis=1))

    def test_pairwise_flat(self):
        a = np.random.rand(10, 3)
        for m in METRIC_LIST:
            assert_array_almost_equal(pairwise_flat(a, m), pdist(a, m))

    def test_pairwise_square(self):
        a = np.random.rand(10, 3)
        for m in METRIC_LIST:
            assert_array_almost_equal(pairwise_square(a, m), cdist(a, a, m))

    def test_fast_buddies(self):
        a = np.random.rand(10, 3)
        for m in METRIC_LIST:
            b1 = fast_buddies(a, m)
            x = cdist(a, a, m)
            np.fill_diagonal(x, np.inf)
            b2 = np.argmin(x, axis=1)
            assert_array_almost_equal(b1, b2)

        b1 = fast_buddies(a, 'idiv')
        for i in xrange(10):
            x = np.sum(a[i] * np.log(a[i] / a), axis=1)
            x[i] = np.inf
            self.assertEqual(b1[i], np.argmin(x))

#
# Copyright tinyclues, All rights reserved
#


from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

import cyperf.clustering.hierarchical as perf_hc
from cyperf.clustering.space_tools import pairwise_flat, METRIC_LIST

__all__ = ['clustering_dispatcher']


def clustering_dispatcher(matrix, weights=None, method="ward",
                          metric="euclidean", ordering=True):
    """
        Dispatching the clustering algorithm.
        >>> import numpy as np
        >>> X = np.array([[0.92, 0.1], [1.0 ,0.0], [0.1, 0.9],
        ...               [0.0, 1.0], [0.51, 0.49]])
        >>> weights = np.ones(5)
        >>> d = clustering_dispatcher(X)
        >>> d
        {0: '110', 1: '111', 2: '01', 3: '00', 4: '10'}
        >>> clustering_dispatcher(X, method="average") == d
        True
        >>> clustering_dispatcher(X, method="average", metric="cosine") == d
        True
    """
    if method == 'ward':
        wt = perf_hc.WardTree(matrix, weights=weights)
        if ordering:
            return wt.build_huffman_ordering()
        else:
            return wt.build_huffman()

    if metric in METRIC_LIST:
        distance_matrix = pairwise_flat(matrix, metric)
    else:
        distance_matrix = pdist(matrix, metric)

    try:
        link = linkage(distance_matrix, method=method, metric=metric)
    except ValueError:
        link = linkage(matrix, method=method, metric=metric)

    if ordering:
        return perf_hc.huffman_encoding_reordering(link, distance_matrix)
    else:
        return perf_hc.huffman_encoding(link)

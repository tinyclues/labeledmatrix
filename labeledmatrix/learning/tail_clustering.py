#
# Copyright tinyclues, All rights reserved
#
import numpy as np

from cyperf.clustering.hierarchical import TailTree


def tail_clustering(vectors, multiplicities, k):
    """
    Fast implementation using cython
    Inputs:
      - vectors: enumerable of vectors or np.matrix
      - multiplicities: enumerable of multiplicities of each vector
      - k: requested number of clusters

    Outputs:
      - the list of assignments of vectors to clusters
        (clusters are labelled by elements in range(k))

    Examples: ::

        >>> from sklearn.metrics import normalized_mutual_info_score
        >>> v = np.random.rand(20, 4)
        >>> m = np.random.rand(20)
        >>> l1 = tail_clustering(v, m, 5)
        >>> l2 = old_tail_clustering(v, m, 5)
        >>> np.allclose(normalized_mutual_info_score(l1, l2), 1.0)
        True
        >>> v = np.array([[7, 0, 3],
        ...               [5, 0, 5],
        ...               [3, 3, 4],
        ...               [0, 10, 0],
        ...               [0, 9, 1]])
        >>> tail_clustering(v, [1, 2, 4, 4, 1], 2)
        [2, 2, 2, 3, 3]
    """
    return TailTree(vectors, multiplicities, k).build_labels().tolist()


def old_tail_clustering(vectors, multiplicities, k):
    """
    Examples: ::
        >>> v = np.array([[1,0,0],
        ...               [0,0,2],
        ...               [0,0,0],
        ...               [0,1,0],
        ...               [0,1,1]])
        >>> old_tail_clustering(v, [1, 2, 3, 4, 1], 2)
        [0, 0, 0, 1, 1]
        >>> nb = 100
        >>> k = 8
        >>> v = np.random.rand(nb,10)
        >>> m = range(nb)
        >>> c = old_tail_clustering(v, m, k)
        >>> len(c)
        100
        >>> set(c) == set(range(k))
        True
        >>> # uncomment this for benchmarking with nosetests
        >>> # import time
        >>> # import numpy as np
        >>> # t = time.time()
        >>> # n = 1000
        >>> # x = old_tail_clustering(np.random.rand(n,20), range(n), 20)
        >>> # time.time() - t
    """
    vecs = np.array([1.0 * x for x in vectors])
    indices = np.arange(len(vecs))
    n = len(vecs)
    mults = np.array([1.0 * x for x in multiplicities])
    clusters = {}
    if k >= n:
        return list(range(n))
    for each in range(n - k):
        activemults = mults.compress(mults)
        activeindices = indices.compress(mults)
        j = activemults.argmin()
        m = activemults[j]
        absolute_j = activeindices[j]
        v = vecs[absolute_j]
        ### now working without v
        mults[absolute_j] = 0.0
        activevectors = vecs.compress(mults, axis=0)
        activeindices = indices.compress(mults)
        norms = np.sum((activevectors - v) ** 2, axis=1)
        jj = norms.argmin()
        vv = activevectors[jj]
        absolute_jj = activeindices[jj]
        mm = mults[absolute_jj]
        clusters[absolute_j] = absolute_jj
        vecs[absolute_jj] = (m * v + mm * vv) / (m + mm)
        mults[absolute_jj] = m + mm
    res = []
    for each in range(n):
        a = each
        while a in clusters:
            a = clusters[a]
        res.append(a)
    distincts = set(res)
    correct = {x: i for i, x in enumerate(list(distincts))}
    return [correct[x] for x in res]

#
# Copyright tinyclues, All rights reserved
#

import numpy as np
from karma import KarmaSetup
from cyperf.clustering.sparse_affinity_propagation import SAFP
from cyperf.matrix.karma_sparse import is_karmasparse
from scipy.sparse import isspmatrix as is_scipy_sparse


def affinity_propagation(similarity, preference=None, max_iter=200, damping=0.6):
    """
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> from sklearn.metrics import euclidean_distances
    >>> from cyperf.matrix.karma_sparse import KarmaSparse
    >>> centers = [[1, 1], [-1, -1], [1, -1]]
    >>> X, labels_true = make_blobs(n_samples=100, centers=centers,
    ...                             cluster_std=0.4, random_state=0)
    >>> similarity = - euclidean_distances(X, squared=True)
    >>> len(np.unique(affinity_propagation(similarity, preference=-20)))
    3
    >>> similarity = np.array([[3, 5, 1, 1],
    ...                        [5, 2, 2, 1],
    ...                        [1, 1, 2, 6],
    ...                        [1, 2, 4, 3]])
    >>> np.all(affinity_propagation(similarity) == np.array([0, 0, 3, 3]))
    True
    >>> np.all(affinity_propagation(KarmaSparse(similarity)) == np.array([0, 0, 3, 3]))
    True
    >>> dense_availability = np.array([[ 3., -3., -4.,  0.],
    ...                                [-0.,  0., -4.,  0.],
    ...                                [ 0., -3.,  0., -0.],
    ...                                [ 0., -3., -4.,  4.]])
    >>> dense_responsibility = np.array([[ 1., -1., -5., -5.],
    ...                                  [ 3., -3., -3., -4.],
    ...                                  [-5., -5., -4.,  4.],
    ...                                  [-6., -5., -3.,  2.]])
    >>> safp = SAFP(similarity)
    >>> clusters = safp.build()
    >>> np.allclose(np.transpose(np.reshape(np.asarray(safp.availability), (4,4))), dense_availability)
    True
    >>> np.allclose(np.transpose(np.reshape(np.asarray(safp.responsibility), (4,4))), dense_responsibility)
    True
    >>> similarity = np.array([[7, 8, 0, 0],
    ...                        [8, 6, 0, 0],
    ...                        [0, 0, 4, 4],
    ...                        [0, 0, 5, 4]])
    >>> np.all(affinity_propagation(similarity) == np.array([0, 0, 2, 2]))
    True
    >>> np.all(affinity_propagation(KarmaSparse(similarity)) == np.array([0, 0, 2, 2]))
    True
    >>> np.random.seed(1000)
    >>> similarity = np.random.rand(100, 100)
    >>> np.all(affinity_propagation(similarity, 0.1) == affinity_propagation(KarmaSparse(similarity), 0.1))
    True
    """
    if similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Similarity must be a square matrix (shape=%s)" % repr(similarity.shape))
    if is_karmasparse(similarity) or is_scipy_sparse(similarity):
        return sparse_affinity_propagation(similarity, preference, max_iter, damping)
    else:
        return dense_affinity_propagation(similarity, preference, max_iter, damping)


def sparse_affinity_propagation(similarity, preference=None, max_iter=200, damping=0.6):
    return SAFP(similarity, preference).build(max_iter=max_iter, damping=damping, verbose=KarmaSetup.verbose)


def dense_affinity_propagation(similarity, preference=None,
                               max_iter=200, damping=0.6,
                               max_repeated_iter=200, eps=0.001):
    n = similarity.shape[0]
    if preference == "mean":
        similarity.flat[::(n + 1)] = np.mean(similarity)
    elif preference == "median":
        similarity.flat[::(n + 1)] = np.median(similarity)
    elif isinstance(preference, (np.ndarray, np.number, float, int)):
        similarity.flat[::(n + 1)] = preference

    damping = max(min(damping, 0.8), 0.2)
    ind = np.arange(n)
    availability = np.zeros(similarity.shape)
    responsibility = np.zeros(similarity.shape)

    similarity = similarity + 10 ** (-6) * similarity * np.random.randn(*similarity.shape)
    is_exemplar = np.zeros(n) > 0
    same_exemplar_iter = 0
    min_eps = np.max(np.abs(np.diff(similarity, axis=1))) * eps
    # Main loop
    for it in xrange(max_iter):
        damping_rand = damping + (0.5 - np.random.rand()) * 0.1
        # Compute responsibility
        responsibility_old = responsibility.copy()
        av_sim = availability + similarity
        max_ind = np.argmax(av_sim, axis=1)
        max_val = av_sim[ind, max_ind[ind]]  # np.max(av_sim, axis=1)

        av_sim[ind, max_ind[ind]] = - np.finfo(np.double).max
        max_out = np.max(av_sim, axis=1)
        responsibility = similarity - max_val[:, np.newaxis]
        responsibility[ind, max_ind[ind]] = similarity[ind, max_ind[ind]] - max_out[ind]
        responsibility = (1 - damping_rand) * responsibility + damping_rand * responsibility_old

        # Compute availabilities
        availability_old = availability.copy()
        responsibility_pos = np.maximum(responsibility, 0)
        responsibility_pos.flat[::n + 1] = responsibility.flat[::n + 1]

        availability = np.sum(responsibility_pos, axis=0)[np.newaxis, :] - responsibility_pos
        diag_av = np.diag(availability)
        availability = np.minimum(availability, 0)
        availability.flat[::n + 1] = diag_av
        availability = (1 - damping_rand) * availability + damping_rand * availability_old

        # Check for convergence
        is_old_exemplar = is_exemplar
        is_exemplar = (np.diag(availability) + np.diag(responsibility)) > 0
        if np.all(is_old_exemplar == is_exemplar):
            same_exemplar_iter += 1
        else:
            same_exemplar_iter = 0
        err = np.max(np.abs(responsibility - responsibility_old)
                     + np.abs(availability - availability_old)) / damping
        if (same_exemplar_iter > max_repeated_iter) or (err < min_eps):
            if KarmaSetup.verbose:
                print "Early stopping condition after {} iterations".format(it)
            break

    # recomute clusters
    exemplar_ind = np.where(is_exemplar)[0]
    nb_exem = np.sum(is_exemplar, axis=0)
    if nb_exem > 0:
        clust = np.argmax(similarity[:, exemplar_ind], axis=1)
        clust[exemplar_ind] = np.arange(nb_exem)
        for k in range(nb_exem):
            ii = np.where(clust == k)[0]
            j = np.argmax(np.sum(similarity[ii[:, np.newaxis], ii], axis=0))
            exemplar_ind[k] = ii[j]
        clust = np.argmax(similarity[:, exemplar_ind], axis=1)
        clust[exemplar_ind] = np.arange(nb_exem)
        labels = exemplar_ind[clust]
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        labels = cluster_centers_indices = ind
    return np.array([cluster_centers_indices[x] for x in labels])

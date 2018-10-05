#cython: embedsignature=True
#cython: wraparound=False
#cython: boundscheck=False


import numpy as np
from cyperf.tools import argsort
from cyperf.matrix.karma_sparse import KarmaSparse, ITYPE, DTYPE, LTYPE
from cyperf.where.indices_where_long import Vector

cimport numpy as np
from cyperf.matrix.karma_sparse cimport KarmaSparse, ITYPE_t, DTYPE_t, LTYPE_t
from cyperf.tools.types cimport INT1, INT2
from cyperf.where.indices_where_long cimport Vector


def sparse_argmax_dispatch(KarmaSparse matrix, int maximum_pressure, INT1[:] max_rank, INT2[:] max_volume):
    """
    Return KarmaSparse matrix of dispatched rows over columns according to argmax score.

    >>> from cyperf.matrix.karma_sparse import KarmaSparse
    >>> matrix = np.array([[0.8, 0.3], [0.4, 0.5], [0., 0.1]])
    >>> volumes = np.array([1, 1])
    >>> ranks = np.array([2, 2])
    >>> res = sparse_argmax_dispatch(KarmaSparse(matrix), maximum_pressure=1, max_rank=ranks, max_volume=volumes)
    >>> res.dtype == DTYPE
    True
    >>> print(res.toarray())
    [[0.8 0. ]
     [0.  0.5]
     [0.  0. ]]
    """

    cdef ITYPE_t nb_user, nb_topic, nnz = matrix.nnz
    nb_user, nb_topic = matrix.shape

    assert nb_topic == max_volume.shape[0]
    assert nb_topic == max_rank.shape[0]
    assert maximum_pressure > 0

    cdef:
        ITYPE_t[::1] user, topic
        ITYPE_t[::1] topic_volume = np.zeros(nb_topic, dtype=ITYPE)
        ITYPE_t[::1] topic_user_rank = np.zeros(nb_topic, dtype=ITYPE)
        ITYPE_t[::1] user_pressure = np.zeros(nb_user, dtype=ITYPE)
        char[::1] is_active_topic = np.full(nb_topic, 1, dtype=np.int8)
        Vector choice = Vector(nb_user * maximum_pressure)
        LTYPE_t[::1] sorted_indices = argsort(matrix.data, reverse=True).astype(LTYPE, copy=False)
        LTYPE_t i, ind
        ITYPE_t u, t, j

    user, topic = matrix.nonzero()

    with nogil:
        for i in xrange(nnz):
            ind = sorted_indices[i]
            u, t = user[ind], topic[ind]
            if topic_user_rank[t] >= max_rank[t] or topic_volume[t] >= max_volume[t]:
                is_active_topic[t] = 0
            if is_active_topic[t]:
                if user_pressure[u] < maximum_pressure:
                    choice.append(ind)
                    user_pressure[u] += 1
                    topic_volume[t] += 1
            topic_user_rank[t] += 1

            for j in xrange(nb_topic):
                if is_active_topic[j]:
                    break
            else:
                break

    cdef np.ndarray[dtype=long, ndim=1] xx = choice.asarray()
    return KarmaSparse((np.asarray(matrix.data)[xx], (np.asarray(user)[xx], np.asarray(topic)[xx])),
                       shape=matrix.shape)

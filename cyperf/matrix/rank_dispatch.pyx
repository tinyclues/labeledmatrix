#cython: embedsignature=True
#cython: nonecheck=True
#cython: overflowcheck=True
#cython: unraisable_tracebacks=True
#cython: wraparound=False
#cython: boundscheck=False


from cython.parallel import prange
from cyperf.matrix.karma_sparse cimport KarmaSparse, np, DTYPE_t, ITYPE_t, LTYPE_t, BOOL_t
from cyperf.tools.types cimport INT1, INT2, A
from cyperf.matrix.karma_sparse import KarmaSparse, np, LTYPE, DTYPE, ITYPE, BOOL
from cyperf.tools.sort_tools cimport partial_sort_decreasing_quick
from libcpp.vector cimport vector
from libc.string cimport memcpy

cdef DTYPE_t MINF = - np.inf # to be removed from candidate

cpdef KarmaSparse matrix_rank_dispatch(np.ndarray[A, ndim=2] raw_matrix,
                                       const ITYPE_t maximum_pressure,
                                       INT1[::1] max_rank,
                                       INT2[::1] max_volume):

    cdef A[:,::1] matrix = np.array(raw_matrix.transpose(), order="C", copy=True)
    cdef ITYPE_t nb_topic = matrix.shape[0]
    cdef ITYPE_t length = matrix.shape[1]

    assert nb_topic == max_rank.shape[0]
    assert nb_topic == max_volume.shape[0]
    assert maximum_pressure >= 0

    cdef ITYPE_t[:, ::1] rank_matrix = np.repeat(np.atleast_2d(np.arange(length,
                                                                         dtype=ITYPE)),
                                                 nb_topic, axis=0)
    cdef ITYPE_t rank, topic, candidate, nb_full = 0
    cdef ITYPE_t[::1] pressure = np.zeros(length, dtype=ITYPE)
    cdef ITYPE_t[::1] count = np.zeros(nb_topic, dtype=ITYPE)
    cdef BOOL_t[::1] full = np.zeros(nb_topic, dtype=BOOL)
    cdef vector[vector[ITYPE_t]] choice

    with nogil:
        # partial sort on columns
        choice.resize(nb_topic)
        for topic in prange(nb_topic):
            choice[topic].reserve(min(max_rank[topic], max_volume[topic]))
            partial_sort_decreasing_quick(&matrix[topic, 0], &rank_matrix[topic, 0],
                                          length, min(max_rank[topic], length))

        for rank in xrange(length):
            for topic in xrange(nb_topic):
                if full[topic]:
                    continue
                candidate = rank_matrix[topic, rank]
                if pressure[candidate] < maximum_pressure and \
                   raw_matrix[candidate, topic] != MINF:
                    choice[topic].push_back(candidate)
                    pressure[candidate] += 1
                    count[topic] += 1
                if count[topic] >= max_volume[topic] or rank + 1 >= max_rank[topic]:
                    full[topic] = 1
                    nb_full += 1
            if nb_full >= nb_topic:
                break

    # convert to KarmaSparse
    cdef LTYPE_t[::1] indptr = np.zeros(nb_topic + 1, dtype=LTYPE)

    for topic in xrange(nb_topic):
        indptr[topic + 1] = indptr[topic] + choice[topic].size()

    cdef ITYPE_t[::1] indices = np.zeros(indptr[nb_topic], dtype=ITYPE)
    cdef DTYPE_t[::1] data = np.zeros(indptr[nb_topic], dtype=DTYPE)

    for topic in prange(nb_topic, nogil=True):
        memcpy(&indices[indptr[topic]], choice[topic].data(),
               sizeof(ITYPE_t) * choice[topic].size())
        choice[topic].clear()

    # copy data
    for topic in prange(nb_topic, nogil=True):
        for candidate in xrange(indptr[topic], indptr[topic+1]):
            data[candidate] = raw_matrix[indices[candidate], topic]

    # free memory
    choice.clear()
    cdef KarmaSparse ks = KarmaSparse((data, indices, indptr),
                                      shape=(length, nb_topic),
                                      format="csc", copy=False)
    return ks

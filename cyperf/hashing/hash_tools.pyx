#cython: boundscheck=False, wraparound=False, unraisable_tracebacks=True

import numpy as np
from cython.parallel import prange, parallel
cimport cython
from cyperf.tools.types cimport A


cdef np.ndarray[char, ndim=2, mode="c"] safe_convertor(np.ndarray keys):
    assert keys.dtype.kind == 'S'
    return np.ascontiguousarray(keys).view(np.int8).reshape(len(keys), keys.dtype.itemsize)


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] hash_numpy_string(np.ndarray keys, signed int seed):
    cdef char[:,::1] keys_str = safe_convertor(keys)
    cdef long k, n = len(keys), size = keys.dtype.itemsize
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)

    with nogil:
        for k in xrange(n):
            result[k] = Hash32WithSeed(&keys_str[k, 0], size, seed)
    return np.array(result)


cpdef np.ndarray[np.uint32_t, ndim=2, mode="c"] hash_numpy_string_with_many_seeds(np.ndarray keys,
                                                                                  np.uint32_t[::1] seed):
    cdef char[:,::1] keys_str = safe_convertor(keys)
    cdef char * ch
    cdef long k, i, n = len(keys), s = len(seed), size = keys.dtype.itemsize
    cdef np.uint32_t[:,::1] result = np.zeros((n, s), dtype=np.uint32)

    with nogil:
        for k in xrange(n):
            ch = &keys_str[k, 0]
            for i in xrange(s):
                result[k, i] = Hash32WithSeed(ch, size, seed[i])
    return np.array(result)


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] hasher_numpy_string(np.ndarray keys,
                                                                    signed int nb_feature,
                                                                    unsigned int seed):
    cdef char[:,::1] keys_str = safe_convertor(keys)
    cdef long k, n = len(keys), size = keys.dtype.itemsize
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)

    with nogil:
        for k in xrange(n):
            result[k] = Hash32WithSeed(&keys_str[k, 0], size, seed) % nb_feature
    return np.array(result)


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] randomizer_numpy_string(np.ndarray keys,
                                                                        np.uint32_t[::1] composition,
                                                                        unsigned int seed):
    cdef char[:,::1] keys_str = safe_convertor(keys)
    cdef long k, n = len(keys), size = keys.dtype.itemsize
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)
    cdef signed int total = np.array(composition).sum()

    with nogil:
        for k in prange(n):
            result[k] = composition_part(Hash32WithSeed(&keys_str[k, 0], size, seed) % total, &composition[0])
    return np.array(result)


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] randomizer_python_string(list keys,
                                                                         np.uint32_t[::1] composition,
                                                                         unsigned int seed):
    cdef long k, n = len(keys)
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)
    cdef signed int total = np.array(composition).sum()
    cdef char* ch
    cdef object x

    for k in xrange(n):
        x = keys[k]
        if not PyString_Check(x):
            try:
                x = str(x)
            except:
                x = ""

        ch = PyString_AsString(x)
        size = len(x)
        result[k] = composition_part(Hash32WithSeed(ch, size, seed) % total, &composition[0])
    return np.array(result)


def increment_over_numpy_string(np.ndarray keys,
                                cython.integral[:] segments,
                                A[:, ::1] values,
                                np.uint32_t[::1] seeds,
                                np.uint32_t[::1] composition,
                                int nb_segments):
    assert len(keys) == segments.shape[0] == values.shape[0]
    assert np.array(segments).min() >= 0
    assert len(composition) == 2
    assert nb_segments >= np.array(segments).max() + 1

    cdef char[:,::1] keys_str = safe_convertor(keys)
    cdef long n = len(keys), size = keys.dtype.itemsize
    cdef long d = values.shape[1]
    cdef long nb_seeds = len(seeds)
    cdef long k, i, j, seg, group
    cdef signed int total = np.array(composition).sum()
    cdef A val
    cdef double ratio = composition[0] * 1. / composition[1], extrapolate = (ratio + 1.) / ratio
    cdef double[:, :, ::1] increment = np.zeros((nb_segments, nb_seeds, d), dtype='float64')


    with nogil:
        for k in xrange(n):
            seg = segments[k]
            for i in xrange(nb_seeds):
                group = composition_part(Hash32WithSeed(&keys_str[k, 0], size, seeds[i]) % total, &composition[0])
                for j in xrange(d):
                    val = values[k, j]
                    if group:
                        increment[seg, i, j] += val * extrapolate
                    else:
                        increment[seg, i, j] -= val * ratio * extrapolate
    return np.array(increment)

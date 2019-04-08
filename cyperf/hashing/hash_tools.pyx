# cython: boundscheck=False, wraparound=False, unraisable_tracebacks=True
# distutils: include_dirs = cyperf/hashing/include


cimport cython
from cython.parallel cimport prange

cimport numpy as np
import numpy as np

from cyperf.tools.types cimport A, ITER, get_c_string, safe_numpy_string_convertor
from libc.string cimport strlen
from cpython.string cimport PyString_Check
from cpython.bytes cimport PyBytes_Check, PyBytes_AsString
from cpython.object cimport PyObject


cdef inline const char* char_array_from_python_object(object x) except NULL:
    if PyBytes_Check(x):
        return PyBytes_AsString(x)

    if PyString_Check(x):
        return get_c_string(<PyObject*>x)

    try:
        x = str(x)
    except:
        x = ""
    return get_c_string(<PyObject*>x)


cdef extern from "city.cc" nogil:
    # our current usage makes the downcast to uint32
    cdef unsigned long CityHash64WithSeed(char *buff, size_t len, unsigned long seed)


cdef inline long python_string_length(const char* ch, long size) nogil:
    cdef long i = size - 1

    while ch[i] == 0 and i >= 0:
        i -= 1
    return i + 1


cdef inline unsigned int composition_part(int residue, const np.uint32_t* composition) nogil:
    """
    nb_group = len(composition) > 0
    total = sum(composition)
    """
    # cdef int residue = x % total
    cdef unsigned int i = 0
    residue -= composition[0]
    while residue >= 0:
        i += 1
        residue -= composition[i]
    return i


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] hash_numpy_string(np.ndarray keys, unsigned int seed):
    cdef const char[:,::1] keys_str = safe_numpy_string_convertor(keys)
    cdef long k, n = len(keys), size = keys.dtype.itemsize
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)
    cdef const char* ch

    with nogil:
        for k in range(n):
            ch = &keys_str[k, 0]
            result[k] = CityHash64WithSeed(ch, python_string_length(ch, size), seed)
    return np.asarray(result)


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] hash_generic_string(ITER keys, unsigned int seed):
    cdef long k, n = len(keys)
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)
    cdef const char* ch

    for k in range(n):
        ch = char_array_from_python_object(keys[k])
        result[k] = CityHash64WithSeed(ch, strlen(ch), seed)
    return np.asarray(result)


cpdef np.ndarray[np.uint32_t, ndim=2, mode="c"] hash_numpy_string_with_many_seeds(np.ndarray keys,
                                                                                  const np.uint32_t[::1] seed):
    cdef const char[:,::1] keys_str = safe_numpy_string_convertor(keys)
    cdef const char * ch
    cdef long k, i, n = len(keys), s = len(seed), size = keys.dtype.itemsize
    cdef np.uint32_t[:,::1] result = np.zeros((n, s), dtype=np.uint32)

    with nogil:
        for k in range(n):
            ch = &keys_str[k, 0]
            for i in range(s):
                result[k, i] = CityHash64WithSeed(ch, size, seed[i])
    return np.asarray(result)


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] hasher_numpy_string(np.ndarray keys,
                                                                    signed int nb_feature,
                                                                    unsigned int seed):
    cdef const char[:,::1] keys_str = safe_numpy_string_convertor(keys)
    cdef long k, n = len(keys), size = keys.dtype.itemsize
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)

    with nogil:
        for k in range(n):
            result[k] = CityHash64WithSeed(&keys_str[k, 0], size, seed) % nb_feature
    return np.asarray(result)


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] randomizer_numpy_string(np.ndarray keys,
                                                                        const np.uint32_t[::1] composition,
                                                                        unsigned int seed):
    cdef const char[:,::1] keys_str = safe_numpy_string_convertor(keys)
    cdef long k, n = len(keys), size = keys.dtype.itemsize
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)
    cdef signed int total = np.asarray(composition).sum()

    with nogil:
        for k in prange(n):
            result[k] = composition_part(<unsigned int>CityHash64WithSeed(&keys_str[k, 0], size, seed) % total,
                                         &composition[0])
    return np.asarray(result)


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] randomizer_python_string(list keys,
                                                                         const np.uint32_t[::1] composition,
                                                                         unsigned int seed):
    cdef long k, n = len(keys)
    cdef np.uint32_t[::1] result = np.zeros(n, dtype=np.uint32)
    cdef signed int total = np.asarray(composition).sum()
    cdef const char* ch
    cdef object x

    for k in range(n):
        ch = char_array_from_python_object(keys[k])
        result[k] = composition_part(<unsigned int>CityHash64WithSeed(ch, strlen(ch), seed) % total, &composition[0])
    return np.asarray(result)


def increment_over_numpy_string(np.ndarray keys,
                                np.ndarray[cython.integral, ndim=1] segments,
                                np.ndarray[A, ndim=2, mode="c"] values,
                                const np.uint32_t[::1] seeds,
                                const np.uint32_t[::1] composition,
                                int nb_segments):
    assert len(keys) == segments.shape[0] == values.shape[0]
    assert np.asarray(segments).min() >= 0
    assert len(composition) == 2
    assert nb_segments >= np.asarray(segments).max() + 1

    cdef const char[:,::1] keys_str = safe_numpy_string_convertor(keys)
    cdef long n = len(keys), size = keys.dtype.itemsize
    cdef long d = values.shape[1]
    cdef long nb_seeds = len(seeds)
    cdef long k, i, j, seg, group
    cdef signed int total = np.asarray(composition).sum()
    cdef A val
    cdef double ratio = composition[0] * 1. / composition[1], extrapolate = (ratio + 1.) / ratio
    cdef double[:, :, ::1] increment = np.zeros((nb_segments, nb_seeds, d), dtype='float64')


    with nogil:
        for k in range(n):
            seg = segments[k]
            for i in range(nb_seeds):
                group = composition_part(<unsigned int>CityHash64WithSeed(&keys_str[k, 0], size, seeds[i]) % total,
                                         &composition[0])
                for j in range(d):
                    val = values[k, j]
                    if group:
                        increment[seg, i, j] -= val * ratio * extrapolate
                    else:
                        increment[seg, i, j] += val * extrapolate
    return np.asarray(increment)

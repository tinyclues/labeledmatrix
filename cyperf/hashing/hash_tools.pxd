cimport cython

cimport numpy as np

from cyperf.tools.types cimport A, ITER


cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] hash_numpy_string(np.ndarray keys, unsigned int seed)

cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] hash_generic_string(ITER keys, unsigned int seed)

cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] hasher_numpy_string(np.ndarray keys,
                                                                    signed int nb_feature,
                                                                    unsigned int seed)

cpdef np.ndarray[np.uint32_t, ndim=2, mode="c"] hash_numpy_string_with_many_seeds(np.ndarray keys,
                                                                                  const np.uint32_t[::1] seed)

cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] randomizer_numpy_string(np.ndarray keys,
                                                                        const np.uint32_t[::1] composition,
                                                                        unsigned int seed)

cpdef np.ndarray[np.uint32_t, ndim=1, mode="c"] randomizer_python_string(list keys,
                                                                         const np.uint32_t[::1] composition,
                                                                         unsigned int seed)

cpdef increment_over_numpy_string(np.ndarray keys,
                                  np.ndarray[cython.integral, ndim=1] segments,
                                  np.ndarray[A, ndim=2, mode="c"] values,
                                  const np.uint32_t[::1] seeds,
                                  const np.uint32_t[::1] composition,
                                  int nb_segments)

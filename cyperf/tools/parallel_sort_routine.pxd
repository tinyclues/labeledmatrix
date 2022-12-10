cimport numpy as np
from cyperf.tools.types cimport A, get_c_string, safe_numpy_string_convertor
from libcpp.functional cimport function


cdef extern from "<parallel/algorithm>" namespace "__gnu_parallel":
    cdef cppclass parallel_tag:
        parallel_tag()
    cdef cppclass sequential_tag:
        sequential_tag()
    cdef cppclass multiway_mergesort_sampling_tag(parallel_tag):
        multiway_mergesort_sampling_tag()
    cdef void sort[T](T first, T last, parallel_tag tag) nogil
    cdef void sort[T, Compare](T first, T last, Compare comp, parallel_tag tag) nogil
    cdef void stable_sort[T, Compare](T first, T last, Compare comp, parallel_tag tag) nogil


ctypedef fused NUMERIC:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t


ctypedef fused FLOATING:
    np.float32_t
    np.float64_t


ctypedef fused PY_ITER_CONTI:
    np.ndarray[object, ndim=1, mode='c']
    list


cpdef void inplace_numerical_parallel_sort(NUMERIC[::1] a, bint reverse=*)
cpdef void inplace_string_parallel_sort(PY_ITER_CONTI arr, bint reverse=*) except *

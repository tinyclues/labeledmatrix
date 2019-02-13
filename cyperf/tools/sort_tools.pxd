cimport numpy as np
from cyperf.tools.types cimport DTYPE_t, ITYPE_t, LTYPE_t, bool, A, B, INT1


cdef void partial_sort(A* dist, B* idx, LTYPE_t size, LTYPE_t m, bool reverse=*) nogil
cdef void partial_unordered_sort(A* dist, B* idx, LTYPE_t size, LTYPE_t m, bool reverse=*) nogil
cdef void partial_sort_decreasing_quick(A* dist, B* idx, LTYPE_t size, LTYPE_t m) nogil
cdef void partial_sort_increasing_quick(A* dist, B* idx, LTYPE_t size, LTYPE_t m) nogil
cpdef np.ndarray[dtype=ITYPE_t, ndim=1] cython_argsort(A[:] xx, ITYPE_t nb,  bool reverse)
cpdef inplace_parallel_sort(A[::1] a)
cpdef np.ndarray[A, ndim=1, mode="c"] parallel_sort(A[:] a)


cdef inline void dual_swap(A * darr, B * iarr, LTYPE_t i1, LTYPE_t i2) nogil:
    """swap the values at index i1 and i2 of both darr and iarr"""
    cdef A dtmp = darr[i1]
    darr[i1] = darr[i2]
    darr[i2] = dtmp

    cdef B itmp = iarr[i1]
    iarr[i1] = iarr[i2]
    iarr[i2] = itmp


cdef inline void selectsort_increasing(A* xx, B* yy, long size, long nb) nogil:
    """
    Implementation from bottleneck.argpartsort
    """
    cdef long i, j, l = 0, r = size - 1, k = nb - 1
    cdef A pivot

    if nb < 1 or nb > size:
        return

    while l < r:
        pivot = xx[k]
        i = l
        j = r
        while 1:
            while xx[i] < pivot: i += 1
            while pivot < xx[j]: j -= 1
            if i <= j:
                dual_swap(xx, yy, i, j)
                i += 1
                j -= 1
            if i > j: break
        if j < k: l = i
        if k < i: r = j


cdef inline void selectsort_decreasing(A* xx, B* yy, long size, long nb) nogil:
    cdef long i, j, l = 0, r = size - 1, k = nb - 1
    cdef A pivot

    if nb < 1 or nb > size:
        return

    while l < r:
        pivot = xx[k]
        i = l
        j = r
        while 1:
            while xx[i] > pivot: i += 1
            while pivot > xx[j]: j -= 1
            if i <= j:
                dual_swap(xx, yy, i, j)
                i += 1
                j -= 1
            if i > j: break
        if j < k: l = i
        if k < i: r = j


cdef inline void inplace_reordering(A* arr, INT1* ind, long size) nogil:
    """
    Performing inplace a reverse reordering of `arr` according to the permutation `ind`.
    **Warning** : `ind` cannot be used once inplace_reordering applied

    Algorithm is linear O(size) in time and constant O(1) in space, see :
    http://blog.merovius.de/2014/08/12/applying-permutation-in-constant.html
    http://stackoverflow.com/questions/13102277/sort-an-array-by-an-index-array-in-c?noredirect=1
    http://stackoverflow.com/questions/7365814/in-place-array-reordering
    http://stackoverflow.com/questions/1683020/is-it-possible-to-rearrange-an-array-in-place-in-on

    >>> a = np.random.rand(100)
    >>> aa = a.copy()
    >>> b = np.arange(len(a), dtype=np.int32)
    >>> np.random.shuffle(b)
    >>> bb = b.copy()
    >>> _inplace_permutation(a, b)
    >>> np.all(a[bb] == aa)
    True
    # that will be equivalent to `arr[argsort(ind)]`
    >>> ind = np.argsort(bb)
    >>> np.all(aa[ind][bb] == aa)
    True
    """
    cdef A x
    cdef long i
    cdef INT1 k, j

    for i in range(size):
        j = ind[i]
        while j != i:
            x = arr[i]; arr[i] = arr[j]; arr[j] = x
            k = ind[j]; ind[j] = j; j = k

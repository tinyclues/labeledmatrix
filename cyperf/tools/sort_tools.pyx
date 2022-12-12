# distutils: language = c++
#cython: overflowcheck=True
#cython: unraisable_tracebacks=True
#cython: wraparound=False
#cython: boundscheck=False


import numpy as np
from cyperf.tools.types import ITYPE, DTYPE, LTYPE

cpdef np.ndarray[dtype=ITYPE_t, ndim=1] cython_argpartition(A[:] xx, ITYPE_t nb,  bool reverse):
    """
    It is here for testing reasons
    >>> cython_argpartition(np.array([1, 2, 1, 2, 1, 2, 1]), 3, True)
    array([3, 5, 1, 6, 4, 2, 0], dtype=int32)
    >>> cython_argpartition(np.array([1, 1, 1, 1, 1, 2, 2]), 3, True)
    array([6, 5, 4, 3, 2, 1, 0], dtype=int32)
    >>> cython_argpartition(np.array([3, 3, 1, 1, 1, 2, 2]), 4, True)
    array([0, 1, 5, 6, 4, 3, 2], dtype=int32)
    >>> cython_argpartition(np.array([3, 3, 1, 1, 1, 6, 2]), 1, True)
    array([5, 1, 2, 3, 4, 0, 6], dtype=int32)
    >>> cython_argpartition(np.array([3, 3, 1, 1, 1, 6, 2]), 4, False)
    array([4, 3, 2, 6, 0, 5, 1], dtype=int32)
    >>> cython_argpartition(np.array([3, 3, 1, 1, 1, 2, 2]), 3, False)
    array([4, 3, 2, 1, 0, 5, 6], dtype=int32)
    """
    cdef:
        A[::1] dist = np.array(xx, copy=True)
        ITYPE_t size = dist.shape[0]
        ITYPE_t[::1] idx = np.arange(size, dtype=ITYPE)
    if nb == -1: nb = size
    assert nb >= 0
    with nogil:
        partial_unordered_sort(&dist[0], &idx[0], size, nb, reverse)
    return np.asarray(idx)


def _inplace_permutation(A[::1] xx, INT1[::1] ind):
    """
    It is here for testing reasons
    >>> a = np.random.rand(100)
    >>> aa = a.copy()
    >>> b = np.arange(len(a))
    >>> np.random.shuffle(b)
    >>> bb = b.copy()
    >>> _inplace_permutation(a, b)
    >>> np.all(a[bb] == aa)
    True
    """
    assert xx.shape[0] == ind.shape[0]
    inplace_reordering(&xx[0], &ind[0], ind.shape[0])


cpdef np.ndarray[dtype=ITYPE_t, ndim=1] cython_argsort(A[:] xx, ITYPE_t nb,  bool reverse):
    """
    It is here for testing reasons
    >>> cython_argsort(np.array([1, 2, 1, 2, 1, 2, 1]), 7, True)
    array([3, 5, 1, 4, 0, 2, 6], dtype=int32)
    >>> cython_argsort(np.array([1, 1, 1, 1, 1, 2, 2]), 7, True)
    array([6, 5, 4, 0, 2, 1, 3], dtype=int32)
    >>> cython_argsort(np.array([3, 3, 1, 1, 1, 2, 2]), 7, True)
    array([0, 1, 5, 6, 4, 2, 3], dtype=int32)
    >>> cython_argsort(np.array([3, 3, 1, 1, 1, 2, 2]), 7, False)
    array([3, 2, 4, 5, 6, 1, 0], dtype=int32)
    >>> cython_argsort(np.array([3, 3, 1, 1, 1, 2, 2]), 3, False)
    array([3, 2, 4, 5, 6, 1, 0], dtype=int32)
    """
    cdef:
        A[::1] dist = np.array(xx, copy=True)
        ITYPE_t size = dist.shape[0]
        ITYPE_t[::1] idx = np.arange(size, dtype=ITYPE)
    if nb == -1: nb = size
    assert nb >= 0
    with nogil:
        partial_sort(&dist[0], &idx[0], size, nb, reverse)
    return np.asarray(idx)


cpdef void cython_simaltanious_sort(A[::1] xx, B[::1] yy,  bool reverse=True):
    assert xx.shape[0] == yy.shape[0]
    with nogil:
        partial_sort(&xx[0], &yy[0], xx.shape[0], xx.shape[0], reverse)


cdef void partial_sort(A* dist, B* idx, LTYPE_t size, LTYPE_t m, bool reverse=True) nogil:
    if reverse:
        partial_sort_decreasing_quick(dist, idx, size, m)
    else:
        partial_sort_increasing_quick(dist, idx, size, m)


cdef void partial_unordered_sort(A* dist, B* idx, LTYPE_t size, LTYPE_t m, bool reverse=True) nogil:
    if reverse:
        selectsort_decreasing(dist, idx, size, m)
    else:
        selectsort_increasing(dist, idx, size, m)


cdef void partial_sort_decreasing_quick(A* dist, B* idx, LTYPE_t size, LTYPE_t m) nogil:
    cdef LTYPE_t left, right
    cdef A pivot_val

    if size <= 1:
        pass
    elif size == 2:
        if dist[0] < dist[1]:
            dual_swap(dist, idx, 0, 1)
    elif size == 3:
        if dist[0] < dist[1]:
            dual_swap(dist, idx, 0, 1)
        if dist[1] < dist[2]:
            dual_swap(dist, idx, 1, 2)
            if dist[0] < dist[1]:
                dual_swap(dist, idx, 0, 1)
    else:
        left = size // 2
        if dist[0] < dist[left]:
            dual_swap(dist, idx, 0, left)
        if dist[left] < dist[size - 1]:
            dual_swap(dist, idx, left, size - 1)
            if dist[0] < dist[left]:
                dual_swap(dist, idx, 0, left)
        pivot_val = dist[left]

        left = 1
        right = size - 2
        while right >= left:
            while dist[left] > pivot_val:
                left += 1
            while dist[right] < pivot_val:
                right -= 1
            if left < right:
                dual_swap(dist, idx, left, right)
            if left <= right:
                left += 1
                right -= 1
        # recursively sort each side of the pivot
        if right > 0:
            partial_sort_decreasing_quick(dist, idx, right + 1, m)
        if left < m:
            partial_sort_decreasing_quick(dist + left, idx + left, size - left, m - left)


cdef void partial_sort_increasing_quick(A* dist, B* idx, LTYPE_t size, LTYPE_t m) nogil:
    cdef LTYPE_t left, right
    cdef A pivot_val

    if size <= 1:
        pass
    elif size == 2:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
    elif size == 3:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
        if dist[1] > dist[2]:
            dual_swap(dist, idx, 1, 2)
            if dist[0] > dist[1]:
                dual_swap(dist, idx, 0, 1)
    else:
        left = size // 2
        if dist[0] > dist[left]:
            dual_swap(dist, idx, 0, left)
        if dist[left] > dist[size - 1]:
            dual_swap(dist, idx, left, size - 1)
            if dist[0] > dist[left]:
                dual_swap(dist, idx, 0, left)
        pivot_val = dist[left]

        left = 1
        right = size - 2
        while right >= left:
            while dist[left] < pivot_val:
                left += 1
            while dist[right] > pivot_val:
                right -= 1
            if left < right:
                dual_swap(dist, idx, left, right)
            if left <= right:
                left += 1
                right -= 1
        # recursively sort each side of the pivot
        if right > 0:
            partial_sort_increasing_quick(dist, idx, right + 1, m)
        if left < m:
            partial_sort_increasing_quick(dist + left, idx + left, size - left, m - left)

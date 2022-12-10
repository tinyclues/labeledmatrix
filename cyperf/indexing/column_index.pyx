# cython: nonecheck=True
# cython: overflowcheck=True
# cython: unraisable_tracebacks=True
# cython: wraparound=False
# cython: boundscheck=False

import numpy as np

from libc.string cimport memcpy
from cpython.sequence cimport PySequence_Check
from cpython.ref cimport PyObject
from cpython.set cimport PySet_Contains, PySet_Add
from cpython.dict cimport PyDict_GetItem
from cpython.tuple cimport PyTuple_CheckExact
from cyperf.tools import inplace_numerical_parallel_sort, argsort
from cyperf.tools.types import BOOL



cpdef INDICES_NP merge_sort(INDICES_NP arr1, INDICES_NP arr2):
    """
    arr1 and arr2 are assumed to have unique&sorted indices
    return is equivalent to np.unique(np.concatenate([arr1, arr2]))

    >>> a = np.array([1, 2, 3, 10])
    >>> b = np.array([-10, 2, 3, 10])
    >>> merge_sort(a, b)
    array([-10,   1,   2,   3,  10])
    """
    cdef long i = 0, j= 0, n1 = arr1.shape[0], n2 = arr2.shape[0], c = 0
    cdef INDICES_NP result = np.empty(n1 + n2, dtype=arr1.dtype)
    cdef int size = arr1.itemsize

    with nogil:
        while i < n1 and j < n2:
            if arr1[i] > arr2[j]:
                result[c] = arr2[j]
                j += 1
            elif arr1[i] < arr2[j]:
                result[c] = arr1[i]
                i += 1
            else:
                result[c] = arr1[i]
                i += 1
                j += 1
            c += 1

        if n1 > i:
            memcpy(&result[c], &arr1[i], (n1 - i) * size)
            c += n1 - i
        else:
            memcpy(&result[c], &arr2[j], (n2 - j) * size)
            c += n2 - j
    result.resize((c,), refcheck=False)
    return result


cpdef INDICES_NP get_unique_indices(INDICES_NP_BIS positions, INDICES_NP indptr, INDICES_NP indices):
    cdef long i, nb = positions.shape[0], pos, size, total_size = 0, count = 0, k
    cdef INDICES_NP result
    cdef BOOL_t[::1] mask

    with nogil:
        for i in range(nb):
            pos = positions[i]
            if pos != -1:
                size = indptr[pos + 1] - indptr[pos]
                total_size += size
                count += 1

    # we use counter-sort for many values. It scales linearly but has constant penalty O(len(indices))
    # and requires O(len(indices)) memory (BOOL_t flags)
    if count > 5 + len(indices) / (1 + total_size):  # constant 5 is arbitrary, but works well for uniform distribution
        mask = np.zeros(len(indices), dtype=BOOL)
        with nogil:
            for i in range(nb):  # this can be done in parallel
                pos = positions[i]
                if pos != -1:
                    for k in range(indptr[pos], indptr[pos + 1]):
                        mask[indices[k]] = 1
        return np.where(mask)[0]
    else:  # merge sort for small number of values (scales quadratically)
        result = np.zeros(0, dtype=indices.dtype)
        for i in range(nb):
            pos = positions[i]
            if pos != -1:
                # TODO: we can merge different slices following the binary tree
                result = merge_sort(result, <INDICES_NP>indices[indptr[pos]:indptr[pos + 1]])
        return result


cpdef long unique_indices_inplace(ITER values, bool reverse, INDICES_NP position):
    assert PySequence_Check(values)
    cdef set seen = set()
    cdef long i, n_keys = 0, nb = len(values)
    cdef object value

    if reverse:
        for i in range(nb):
            value = values[i]
            if PySet_Contains(seen, value) != 1:
                PySet_Add(seen, value)
                position[n_keys] = i
                n_keys += 1
    else:
        for i in range(nb-1, -1, -1):
            value = values[i]
            if PySet_Contains(seen, value) != 1:
                PySet_Add(seen, value)
                position[n_keys] = i
                n_keys += 1
        position[:n_keys] = position[:n_keys][::-1]
    return n_keys


cpdef dict factorize_inplace(ITER values, INDICES_NP reversed_indices):
    assert PySequence_Check(values)

    cdef dict key_position = {}
    cdef long nb = len(values), i, key_ind, n_keys = 0
    cdef PyObject *obj
    cdef object val

    for i in range(nb):
        val = values[i]
        obj = PyDict_GetItem(key_position, val)
        if obj is not NULL:
            # if a key have been already seen we put its position into reversed_indices
            # and increment its counter
            key_ind = <object>obj
            reversed_indices[i] = key_ind
        else:
            # if we see a key for a first time we add it to dict and initialize its counter
            key_position[val] = n_keys
            reversed_indices[i] = n_keys
            n_keys += 1

    return key_position



# TODO : find good rule to make a switch between groupsort_indexer and groupsort_indexer_as_parallel_argsort
cpdef tuple groupsort_indexer_as_parallel_argsort(INDICES_NP indptr, INDICES_NP_BIS reversed_indices):
    indptr[1:] = np.bincount(reversed_indices, minlength=len(indptr)-1)
    nb_unique = np.count_nonzero(indptr)
    np.cumsum(indptr, out=indptr)
    indices = argsort(reversed_indices)
    return nb_unique, indices


cpdef tuple groupsort_indexer(INDICES_NP indptr, INDICES_NP_BIS reversed_indices):
    """
    Args:
        reversed_indices: numpy array representing values using dict of positions
        indptr: numpy array with index pointers (should be np.zeros of good dtype)

    Equivalent to pandas.algo.groupsort_indexer

    Returns: (nb_distinct_values, compact numpy array with indices of each position)
    """
    cdef long nb = reversed_indices.shape[0], n_keys = indptr.shape[0] - 1, length = 0
    cdef long i, ind, pos
    # by construction maximal value in both indices and indptr equals to size of reversed_indices
    # so indices and indptr will have the same dtype
    cdef INDICES_NP indices = np.empty(nb, dtype=indptr.dtype)

    with nogil:
        for i in range(nb):
            ind = reversed_indices[i] + 1
            indptr[ind] += 1

        for i in range(n_keys):
            if indptr[i + 1] > 0:
                length += 1
            indptr[i + 1] += indptr[i]


        # while filling in indices array we increment indptrs
        for i in range(nb):
            ind = reversed_indices[i]
            indices[indptr[ind]] = i
            indptr[ind] += 1

        # at the end instead of array [0, a, a+b, a+b+c, ...] we have [a, a+b, a+b+c, ...]
        # we move indptr array one step right
        pos = 0
        for i in range(n_keys):
            ind = indptr[i]
            indptr[i] = pos
            pos = ind

    return length, indices


cpdef INDICES_NP get_positions(dict position, ITER values, INDICES_NP reversed_indices):
    """
    Returns positions for every item in values; reversed_indices are needed to determine a dtype of an output array
    """
    cdef long i, nb = len(values), pos
    cdef INDICES_NP positions = np.empty(nb, dtype=reversed_indices.dtype)

    for i in range(nb):
        val = values[i]
        obj = PyDict_GetItem(position, val)
        if obj is not NULL:
            pos = <object>obj
            positions[i] = pos
        else:
            positions[i] = -1

    return positions


cpdef INDICES_NP positions_select_inplace(INDICES_NP positions, INDICES_NP_BIS indptr):
    """
    Switch positions of values not contained in selection to -1
    """
    cdef long i, nb = positions.shape[0], pos

    with nogil:
        for i in range(nb):
            pos = positions[i]
            if pos != -1:
                if indptr[pos + 1] - indptr[pos] == 0:
                    positions[i] = -1

    return positions


cpdef INDICES_NP get_positions_multiindex(dict position, INDICES_NP reversed_indices,
                                          long coeff, INDICES_NP_BIS positions_0, INDICES_NP_TER positions_1):
    """
    Returns positions of tuples of elements given by positions _0 and _1;
    reversed_indices are needed to determine a dtype of an output array;
    we suppose that reversed_indices, positions_0, positions_1 have all same length
    """
    cdef long i, nb = positions_0.shape[0], val, pos_0, pos_1
    cdef INDICES_NP positions = np.empty(nb, dtype=reversed_indices.dtype)
    cdef PyObject *obj

    for i in range(nb):
        pos_0 = positions_0[i]
        pos_1 = positions_1[i]

        if pos_0 == -1 or pos_1 == -1:
            positions[i] = -1
            continue

        val = coeff * pos_0 + pos_1
        obj = PyDict_GetItem(position, val)
        if obj is not NULL:
            pos = <object>obj
            positions[i] = pos
        else:
            positions[i] = -1

    return positions


cpdef INDICES_NP get_size_batch(INDICES_NP_BIS positions, INDICES_NP indptr):
    cdef long i, nb = positions.shape[0], pos
    cdef INDICES_NP result = np.zeros(nb, dtype=indptr.dtype)

    with nogil:
        for i in range(nb):
            pos = positions[i]
            if pos != -1:
                result[i] = indptr[pos + 1] - indptr[pos]
    return result


cpdef tuple get_batch_indices(INDICES_NP_BIS positions, INDICES_NP indptr, INDICES_NP indices):
    cdef long i, nb = positions.shape[0], pos, size
    cdef long[::1] res_indptr = np.zeros(nb + 1, dtype=np.int64)
    cdef int itemsize = indices.dtype.itemsize
    cdef INDICES_NP result

    with nogil:
        for i in range(nb):
            pos = positions[i]
            if pos != -1:
                size = indptr[pos + 1] - indptr[pos]
            else:
                size = 0
            res_indptr[i + 1] = res_indptr[i] + size

    result = np.zeros(res_indptr[nb], dtype=indices.dtype)
    # TODO if we use Vector instead of an result array we can do everything in only one loop
    with nogil:
        for i in range(nb):
            pos = positions[i]
            if pos != -1:
                memcpy(&result[res_indptr[i]], &indices[indptr[pos]], (res_indptr[i + 1] - res_indptr[i]) * itemsize)

    return result, np.asarray(res_indptr)


cpdef INDICES_NP get_first_batch(INDICES_NP_BIS positions, INDICES_NP indptr, INDICES_NP indices):
    cdef long i, nb = positions.shape[0], pos
    cdef INDICES_NP result = np.empty(nb, dtype=indptr.dtype)

    with nogil:
        for i in range(nb):
            pos = positions[i]
            if pos != -1:
                result[i] = indices[indptr[pos]]
            else:
                result[i] = -1
    return result


cpdef INDICES_NP sorted_indices(INDICES_NP_BIS sorted_keys_positions, INDICES_NP indptr, INDICES_NP indices):
    cdef long pos, size, i, count = 0, n_keys = sorted_keys_positions.shape[0]
    cdef INDICES_NP result = np.zeros(indptr[indptr.shape[0] - 1], dtype=indices.dtype)
    cdef int itemsize = indices.dtype.itemsize

    with nogil:
        for i in range(n_keys):
            pos = sorted_keys_positions[i]
            size = indptr[pos + 1] - indptr[pos]
            memcpy(&result[count], &indices[indptr[pos]], size * itemsize)
            count += size

    return result


cpdef INDICES_NP quantiles_indices(INDICES_NP_BIS sorted_keys_positions, INDICES_NP actual_indptr,
                                   INDICES_NP actual_indices, INDICES_NP_BIS indptr, long nb_of_quantiles):
    """
    Calculates indices of values given by actual_indices and actual_indptr for quantiles boundaries
    Args:
        sorted_keys_positions: iterable of positions of keys in sorted order
        actual_indptr: indptr array to retrieve a slice with indices in _values (parent_indptr for SelectIndex)
        actual_indices: array with indices to retrieve an index of a key in _values (parent_indices for SelectIndex)
        indptr: indptr array to check value's size
        nb_of_quantiles: number of quantiles to calculate

    Returns: indices of quantile boundaries in _values (obtained by actual_indices[actual_indptr[position[key]]])
    so it should be the same type as actual_indices
    """
    cdef long pos, size, i, count = 0, current_indice = 0
    cdef long n_keys = sorted_keys_positions.shape[0]
    cdef float step = 1.0 * indptr[indptr.shape[0] - 1] / nb_of_quantiles
    cdef INDICES_NP boundary_indices = np.zeros(nb_of_quantiles - 1, dtype=actual_indices.dtype)

    with nogil:
        for i in range(n_keys):
            pos = sorted_keys_positions[i]
            if pos != -1:
                size = indptr[pos + 1] - indptr[pos]
                while count < nb_of_quantiles - 1 and current_indice + size > <int>(0.5 + (count + 1) * step) - 1:
                    boundary_indices[count] = actual_indices[actual_indptr[pos]]
                    count += 1

                current_indice += size

    return boundary_indices


cpdef tuple quantiles_indices_with_first(INDICES_NP_BIS sorted_keys_positions, INDICES_NP actual_indptr,
                                         INDICES_NP actual_indices, INDICES_NP_BIS indptr, long nb_of_quantiles):
    cdef long pos, size, i, count = 0, current_indice = 0
    cdef long n_keys = sorted_keys_positions.shape[0]
    cdef long first_in_quantile = -1
    cdef float step = 1.0 * indptr[indptr.shape[0] - 1] / nb_of_quantiles
    cdef INDICES_NP boundary_indices = np.zeros(nb_of_quantiles - 1, dtype=actual_indices.dtype)
    cdef INDICES_NP labels_indices = np.zeros(nb_of_quantiles, dtype=actual_indices.dtype)

    with nogil:
        for i in range(n_keys):
            pos = sorted_keys_positions[i]
            if pos != -1:
                size = indptr[pos + 1] - indptr[pos]
                if first_in_quantile == -1:
                    first_in_quantile = actual_indices[actual_indptr[pos]]

                while count < nb_of_quantiles - 1 and current_indice + size > <int>(0.5 + (count + 1) * step) - 1:
                    boundary_indices[count] = actual_indices[actual_indptr[pos]]
                    labels_indices[count] = first_in_quantile
                    if current_indice + size > <int>(0.5 + (count + 1) * step):
                        first_in_quantile = actual_indices[actual_indptr[pos]]
                    else:
                        first_in_quantile = -1
                    count += 1
                current_indice += size

        labels_indices[count] = first_in_quantile

    return boundary_indices, labels_indices


cpdef tuple quantiles_indices_with_most_common(INDICES_NP_BIS sorted_keys_positions, INDICES_NP actual_indptr,
                                               INDICES_NP actual_indices, INDICES_NP_BIS indptr, long nb_of_quantiles):
    cdef long pos, size, i, count = 0, current_indice = 0
    cdef long n_keys = sorted_keys_positions.shape[0]
    cdef float step = 1.0 * indptr[indptr.shape[0] - 1] / nb_of_quantiles
    cdef long most_common_index = 0, most_common_occurrence = 0, occurrence = 0
    cdef INDICES_NP boundary_indices = np.zeros(nb_of_quantiles - 1, dtype=actual_indices.dtype)
    cdef INDICES_NP labels_indices = np.zeros(nb_of_quantiles, dtype=actual_indices.dtype)

    with nogil:
        for i in range(n_keys):
            pos = sorted_keys_positions[i]
            if pos != -1:
                size = indptr[pos + 1] - indptr[pos]

                if current_indice + size > <int>(0.5 + (count + 1) * step) - 1:
                    occurrence = <int>(0.5 + (count + 1) * step) - current_indice
                else:
                    occurrence = size

                if occurrence > most_common_occurrence:
                    most_common_occurrence = occurrence
                    most_common_index = actual_indices[actual_indptr[pos]]

                while count < nb_of_quantiles - 1 and current_indice + size > <int>(0.5 + (count + 1) * step) - 1:
                    boundary_indices[count] = actual_indices[actual_indptr[pos]]
                    labels_indices[count] = most_common_index
                    most_common_index = actual_indices[actual_indptr[pos]]
                    most_common_occurrence = current_indice + size - <int>(0.5 + (count + 1) * step)
                    count += 1
                current_indice += size

        labels_indices[count] = most_common_index

    return boundary_indices, labels_indices


cpdef dict count(ITER keys, INDICES_NP indptr, INDICES_NP indices):
    """
    We need to pass keys iterable indexed in order of indptr
    """
    cdef dict result = {}
    cdef long n_keys = len(keys), pos, start

    for pos in range(n_keys):
        start = indptr[pos]
        result[keys[pos]] = indptr[pos + 1] - start

    return result


cpdef INDICES_NP key_indices_multiindex(INDICES_NP indptr, INDICES_NP indices):
    cdef long nb = indptr.shape[0] - 1, pos
    cdef INDICES_NP result = np.empty(nb, dtype=indices.dtype)

    with nogil:
        for pos in range(nb):
            result[pos] = indices[indptr[pos]]

    return result


# In several methods for SelectIndex the following iteration procedure is used:
# while i < len(self.indices):
#     pos = self.reversed_indices[self.indices[i]]
#     do something
#     i = self.indptr[pos + 1]
# here we iterate over all unique key positions used in selection
# order of iteration is that given by the parent index
# TODO find a way to move this pattern outside from: count_select, deduplicate_indices_select, reversed_index_select,
# keys_select, items_select, values_select


cpdef dict count_select(ITER keys, INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices):
    """
    We need to pass keys iterable indexed in order of indptr
    """
    cdef dict result = {}
    cdef long nb = indices.shape[0], i = 0, pos, start, count = 0

    while i < nb:
        pos = reversed_indices[indices[i]]
        i = indptr[pos + 1]
        result[keys[count]] = i - indptr[pos]
        count += 1

    return result


cpdef INDICES_NP deduplicate_indices(INDICES_NP indptr, INDICES_NP indices, str take):
    cdef long i = 0, nb = indptr.shape[0] - 1
    cdef INDICES_NP unique_indices = np.zeros(nb, dtype=indices.dtype)

    if take != 'first' and take != 'last':
        raise ValueError('"take" must be either "first" or "last"')

    if take == 'first':
        for i in range(nb):
            unique_indices[i] = indices[indptr[i]]
    else:
        for i in range(nb):
            unique_indices[i] = indices[indptr[i + 1] - 1]
        inplace_numerical_parallel_sort(unique_indices)
    return unique_indices


cpdef INDICES_NP deduplicate_indices_select(INDICES_NP indptr, INDICES_NP indices,
                                            INDICES_NP_BIS reversed_indices, long length, str take):
    cdef long i = 0, nb = indices.shape[0], count = 0
    cdef INDICES_NP unique_indices = np.zeros(length, dtype=indices.dtype)

    if take != 'first' and take != 'last':
        raise ValueError('"take" must be either "first" or "last"')

    while i < nb:
        if take == 'first':
            unique_indices[count] = indices[i]
            i = indptr[reversed_indices[indices[i]] + 1]
        else:
            i = indptr[reversed_indices[indices[i]] + 1]
            unique_indices[count] = indices[i - 1]
        count += 1
    inplace_numerical_parallel_sort(unique_indices)
    return unique_indices


cpdef INDICES_NP get_keys_indices(INDICES_NP indptr, INDICES_NP indices):
    cdef long pos, nb = indptr.shape[0] - 1
    cdef INDICES_NP keys_indices = np.zeros(nb, dtype=indices.dtype)
    with nogil:
        for pos in range(nb):
            keys_indices[pos] = indices[indptr[pos]]
    return keys_indices


cpdef tuple reversed_index_select(INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices,
                                  INDICES_NP_TER parent_indices, INDICES_NP_TER parent_indptr, long length):
    cdef long pos, nb = indices.shape[0], i = 0, j, count = 0, ind
    cdef INDICES_NP keys_positions = np.zeros(length, dtype=indices.dtype)
    cdef INDICES_NP_BIS new_reversed_indices = np.zeros(nb, dtype=reversed_indices.dtype)

    with nogil:
        while i < nb:
            pos = reversed_indices[indices[i]]
            keys_positions[count] = parent_indices[parent_indptr[pos]]
            for j in range(indptr[pos], indptr[pos + 1]):
                ind = indices[j]
                new_reversed_indices[ind] = count
            i = indptr[pos + 1]
            count += 1
    return keys_positions, new_reversed_indices


cpdef INDICES_NP_TER get_keys_indices_select(INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices,
                                             INDICES_NP_TER parent_indices, INDICES_NP_TER parent_indptr, long length):
    cdef long nb = indices.shape[0], i = 0, pos, count = 0
    cdef INDICES_NP_TER keys_indices = np.zeros(length, dtype=parent_indices.dtype)

    with nogil:
        while i < nb:
            pos = reversed_indices[indices[i]]
            keys_indices[count] = parent_indices[parent_indptr[pos]]
            i = indptr[pos + 1]
            count += 1
    return keys_indices


cpdef tuple compact_select(ITER values, INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices):
    """
    This method should launched with already unlazied values, so values correspond exactly to reversed_indices,
    so we don't need parent_indices and parent_indptr
    """
    cdef long nb = reversed_indices.shape[0], i, j, count = 0, pos, ind, size, value_ind
    cdef INDICES_NP_BIS new_reversed_indices
    cdef INDICES_NP new_indices, new_indptr
    cdef dict new_position = {}
    cdef int itemsize = indices.dtype.itemsize

    new_reversed_indices = np.full(nb, -1, dtype=reversed_indices.dtype)
    new_indices = np.zeros(indices.shape[0], dtype=indices.dtype)
    new_indptr = np.zeros(nb + 1, dtype=indptr.dtype)

    for i in range(nb):
        if new_reversed_indices[i] == -1:
            pos = reversed_indices[i]
            new_position[values[i]] = count
            for j in range(indptr[pos], indptr[pos + 1]):
                ind = indices[j]
                new_reversed_indices[ind] = count
            size = indptr[pos + 1] - indptr[pos]
            new_indptr[count + 1] = new_indptr[count] + size

            memcpy(&new_indices[new_indptr[count]], &indices[indptr[pos]], size * itemsize)
            count += 1

    return new_position, new_indices, new_reversed_indices, new_indptr[:count + 1].copy()


cpdef dict compact_multiindex(ITER values, INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices):
    cdef dict new_position = {}
    cdef long nb = indptr.shape[0] - 1, i, j

    for i in range(nb):
        j = indices[indptr[i]]
        new_position[values[j]] = reversed_indices[j]

    return new_position


cpdef tuple dispatch_tupled_values(list values, object default_value):
    cdef long nb = len(values), i
    cdef list result0 = [], result1 = []
    cdef object x, y, val

    for i in range(nb):
        val = values[i]
        if PyTuple_CheckExact(val) and len(val) == 2:
            x, y = val
            result0.append(x)
            result1.append(y)
        else:
            result0.append(default_value)
            result1.append(default_value)

    return result0, result1

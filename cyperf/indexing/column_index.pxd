cimport cython
cimport numpy as np
from cyperf.tools.types cimport BOOL_t, bool, ITER

ctypedef fused INDICES_NP:
    np.ndarray[dtype=int, ndim=1, mode="c"]
    np.ndarray[dtype=long, ndim=1, mode="c"]

ctypedef fused INDICES_NP_BIS:
    np.ndarray[dtype=int, ndim=1, mode="c"]
    np.ndarray[dtype=long, ndim=1, mode="c"]

ctypedef fused INDICES_NP_TER:
    np.ndarray[dtype=int, ndim=1, mode="c"]
    np.ndarray[dtype=long, ndim=1, mode="c"]

cpdef INDICES_NP merge_sort(INDICES_NP arr1, INDICES_NP arr2)

cpdef long unique_indices_inplace(ITER values, bool reverse, INDICES_NP position)

cpdef dict factorize_inplace(ITER values, INDICES_NP reversed_indices)

cpdef INDICES_NP get_positions(dict position, ITER values, INDICES_NP reversed_indices)

cpdef INDICES_NP positions_select_inplace(INDICES_NP positions, INDICES_NP_BIS indptr)

cpdef INDICES_NP get_positions_multiindex(dict position, INDICES_NP reversed_indices,
                                          long coeff, INDICES_NP_BIS positions_0, INDICES_NP_TER positions_1)

cpdef INDICES_NP get_size_batch(INDICES_NP_BIS positions, INDICES_NP indptr)

cpdef tuple get_batch_indices(INDICES_NP_BIS positions, INDICES_NP indptr, INDICES_NP indices)

cpdef INDICES_NP get_unique_indices(INDICES_NP_BIS positions, INDICES_NP indptr, INDICES_NP indices)

cpdef INDICES_NP get_first_batch(INDICES_NP_BIS positions, INDICES_NP indptr, INDICES_NP indices)

cpdef INDICES_NP sorted_indices(INDICES_NP_BIS sorted_keys_positions, INDICES_NP indptr, INDICES_NP indices)

cpdef INDICES_NP quantiles_indices(INDICES_NP_BIS sorted_keys_positions, INDICES_NP actual_indptr,
                                   INDICES_NP actual_indices, INDICES_NP_BIS indptr, long nb_of_quantiles)

cpdef tuple quantiles_indices_with_first(INDICES_NP_BIS sorted_keys_positions, INDICES_NP actual_indptr,
                                         INDICES_NP actual_indices, INDICES_NP_BIS indptr, long nb_of_quantiles)

cpdef tuple quantiles_indices_with_most_common(INDICES_NP_BIS sorted_keys_positions, INDICES_NP actual_indptr,
                                               INDICES_NP actual_indices, INDICES_NP_BIS indptr, long nb_of_quantiles)

cpdef dict count(ITER keys, INDICES_NP indptr, INDICES_NP indices)

cpdef INDICES_NP key_indices_multiindex(INDICES_NP indptr, INDICES_NP indices)

cpdef dict count_select(ITER values, INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices)

cpdef INDICES_NP deduplicate_indices(INDICES_NP indptr, INDICES_NP indices, str take)

cpdef INDICES_NP deduplicate_indices_select(INDICES_NP indptr, INDICES_NP indices,
                                            INDICES_NP_BIS reversed_indices, long length, str take)

cpdef INDICES_NP get_keys_indices(INDICES_NP indptr, INDICES_NP indices)

cpdef tuple reversed_index_select(INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices,
                                  INDICES_NP_TER parent_indices, INDICES_NP_TER parent_indptr, long length)

cpdef INDICES_NP_TER get_keys_indices_select(INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices,
                                             INDICES_NP_TER parent_indices, INDICES_NP_TER parent_indptr, long length)

cpdef tuple compact_select(ITER values, INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices)

cpdef dict compact_multiindex(ITER values, INDICES_NP indptr, INDICES_NP indices, INDICES_NP_BIS reversed_indices)

cpdef tuple dispatch_tupled_values(list values, object default_value)

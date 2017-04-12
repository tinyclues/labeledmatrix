
from cyperf.indexing.indexed_list import is_increasing
from cyperf.matrix.routine import indices_truncation, first_indices, last_indices
import numpy as np
from cyperf.matrix.karma_sparse import KarmaSparse
from pandas.hashtable import Int64HashTable


def safe_datetime64_cast(date_values):
    return np.asarray(np.asarray(date_values, dtype="S10"), dtype="datetime64[D]")


def sorted_unique(array):
    """
    Equivalent to np.unique(array, return_index=True)
    Assumes that array is already sorted, so works faster than np.unique
    """
    if len(array) == 0:
        return array, np.array([], dtype=np.int)
    else:
        index = np.where(array[1:] > array[:-1])[0]
        index += 1
        index = np.hstack([[0], index])
    return array[index], index


class SortedDateIndex(object):

    def __init__(self, date_values):
        self.date_values = safe_datetime64_cast(date_values)
        assert is_increasing(self.date_values)
        self.min_date, self.max_date = self.date_values[0], self.date_values[-1]
        self._create_index()

    @property
    def is_without_holes(self):
        return len(self.unique_date) == (self.max_date - self.min_date).view(int) + 1

    def _create_index(self):
        self.unique_date, self.interval_indices = sorted_unique(self.date_values)
        if not self.is_without_holes:
            full_unique_date = np.arange(self.min_date, self.max_date + 1)
            full_interval_indices = np.zeros(len(full_unique_date), dtype=np.int)

            # It's not too big - no need to cythonize it
            j = 0
            for i, d in enumerate(self.unique_date):
                while full_unique_date[j] < d:
                    full_interval_indices[j] = self.interval_indices[i]
                    j += 1

                full_interval_indices[j] = self.interval_indices[i]
                j += 1

            self.unique_date = full_unique_date
            self.interval_indices = full_interval_indices

        self.interval_indices = np.hstack([self.interval_indices,
                                           [len(self.date_values)]])

    def get_window_indices(self, d, lower=-1, upper=0):
        assert lower < upper
        # indices for day such that lower <= day - d  < upper
        d = safe_datetime64_cast(d)

        d_lower = (d + lower).clip(self.min_date, self.max_date + 1)
        d_lower = (d_lower - self.min_date).view(np.int)

        d_upper = (d + upper).clip(self.min_date, self.max_date + 1)
        d_upper = (d_upper - self.min_date).view(np.int)

        lower_bound = self.interval_indices[d_lower]
        upper_bound = self.interval_indices[d_upper]
        return lower_bound, upper_bound

    def decay(self, d, half_life):
        d = safe_datetime64_cast(d)

        min_date = min(d.min(), self.min_date)
        max_date = max(d.max(), self.max_date)
        delta = (max_date - min_date + 1).astype(int)

        unique_decayed = 2. ** (-np.arange(delta, dtype=np.float) / half_life)
        ind_d = (d - min_date).view(np.int)
        ind_self = (self.date_values - min_date).view(np.int)
        row_decay = unique_decayed[ind_d]
        column_decay = (1. / unique_decayed)[ind_self]
        return row_decay, column_decay


class PastTruncatedIndex(object):

    def __init__(self, user_index, sorted_date_index):
        self.user_index = user_index
        self.sorted_date_index = sorted_date_index

    def get_batch_window_indices(self, u, d, lower=-1, upper=0):
        assert len(d) == len(u)
        lower_bound, upper_bound = self.sorted_date_index.get_window_indices(d, lower, upper)

        # This query can be factorized by unique users !
        indices, indptr = self.user_index.get_batch_indices(u)
        # individual indices are ordered !
        truncated_indices, truncated_indptr = indices_truncation(indices, indptr,
                                                                 lower_bound, upper_bound)
        return truncated_indices, truncated_indptr

    def get_first_batch_window_indices_with_intensities(self, u, d, lower=-1, upper=0, half_life=None):
        assert len(d) == len(u)
        lower_bound, upper_bound = self.sorted_date_index.get_window_indices(d, lower, upper)

        # This query can be factorized by unique users !
        indices, indptr = self.user_index.get_batch_indices(u)
        # individual indices are ordered !
        indices, indptr = first_indices(indices, indptr, lower_bound, upper_bound)
        data = np.ones(len(indices), dtype=np.float64)
        ks = KarmaSparse((data, indices, indptr), format="csr",
                         shape=(len(u), self.user_index.indptr[-1]),
                         copy=False, has_sorted_indices=True,
                         has_canonical_format=True)

        if half_life is not None:
            row_decay, column_decay = self.sorted_date_index.decay(d, half_life=half_life)
            ks = ks.scale_along_axis_inplace(row_decay, axis=1)\
                   .scale_along_axis_inplace(column_decay, axis=0)
        return compactify_on_right(ks)

    def get_last_batch_window_indices_with_intensities(self, u, d, lower=-1, upper=0, half_life=None):
        assert len(d) == len(u)
        lower_bound, upper_bound = self.sorted_date_index.get_window_indices(d, lower, upper)

        # This query can be factorized by unique users !
        indices, indptr = self.user_index.get_batch_indices(u)
        # individual indices are ordered !
        indices, indptr = last_indices(indices, indptr, lower_bound, upper_bound)
        data = np.ones(len(indices), dtype=np.float64)
        ks = KarmaSparse((data, indices, indptr), format="csr",
                         shape=(len(u), self.user_index.indptr[-1]),
                         copy=False, has_sorted_indices=True,
                         has_canonical_format=True)

        if half_life is not None:
            row_decay, column_decay = self.sorted_date_index.decay(d, half_life=half_life)
            ks = ks.scale_along_axis_inplace(row_decay, axis=1)\
                   .scale_along_axis_inplace(column_decay, axis=0)
        return compactify_on_right(ks)

    def get_batch_window_indices_with_intensity(self, u, d, lower=-1, upper=0, half_life=None):
        indices, indptr = self.get_batch_window_indices(u, d, lower, upper)
        data = np.ones(len(indices), dtype=np.float64)
        ks = KarmaSparse((data, indices, indptr), format="csr",
                         shape=(len(u), self.user_index.indptr[-1]),
                         copy=False, has_sorted_indices=True,
                         has_canonical_format=True)

        if half_life is not None:
            row_decay, column_decay = self.sorted_date_index.decay(d, half_life=half_life)
            ks = ks.scale_along_axis_inplace(row_decay, axis=1)\
                   .scale_along_axis_inplace(column_decay, axis=0)
        return compactify_on_right(ks)


def compactify_on_right(ks):
    """
    >>> from cyperf.matrix.karma_sparse import KarmaSparse, np, sp
    >>> ks = KarmaSparse(sp.rand(3, 10, density=0.1))
    >>> out = np.arange(20).reshape(10, 2)
    >>> ind, ks_compact = compactify_on_right(ks)
    >>> np.allclose(ks.dot(out), ks_compact.dot(out[ind]))
    True
    """
    if ks.nnz / (ks.shape[1] + 1.) < 0.1:
        if ks.format == "csr":
            required_indices = np.unique(ks.indices)
        else:
            required_indices = np.unique(ks.nonzero()[1])
    else:
        required_indices = np.where(ks.count_nonzero(axis=0).astype(np.bool_))[0]
    nb_unique_indices = len(required_indices)
    if nb_unique_indices == 0:  # FIXME : what to do if we have only relational missing ??
        required_indices = np.array([0])  # currently we will taking first element to keep compatibility
        nb_unique_indices = 1
    if nb_unique_indices < ks.shape[1]:
        if ks.format == "csr":
            # apply mapping
            ii = Int64HashTable(nb_unique_indices * 2)
            ii.map_locations(required_indices.astype(np.int, copy=False))
            ks_compact = KarmaSparse((ks.data, ii.lookup(ks.indices.astype(np.int)), ks.indptr),
                                     format='csr', shape=(ks.shape[0], nb_unique_indices), copy=False,
                                     has_canonical_format=True, has_sorted_indices=True)
            return required_indices, ks_compact
        else:
            return required_indices, ks[:, required_indices]
    else:
        return np.arange(ks.shape[1], dtype=np.int), ks

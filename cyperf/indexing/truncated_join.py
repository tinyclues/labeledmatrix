
from cyperf.indexing.indexed_list import is_increasing
from cyperf.matrix.routine import (indices_truncation_sorted, first_indices_sorted, last_indices_sorted,
                                   indices_truncation_lookup, first_indices_lookup, last_indices_lookup)
import numpy as np
from cyperf.matrix.karma_sparse import KarmaSparse
from pandas.core.algorithms import htable
from pandas import to_datetime


def create_truncated_index(user_index, date_values):
    assert user_index.indices.shape[0] == len(date_values)

    date_values = safe_datetime64_cast(date_values)
    mask = np.logical_not(np.isnat(date_values))

    if len(date_values) == 0:
        raise ValueError('Empty data is not supported')

    if not np.any(mask):
        raise ValueError('Date conversion failed')

    if np.all(mask) and is_increasing(date_values):
        return SortedTruncatedIndex(user_index, SortedDateIndex(date_values))
    else:
        return LookUpTruncatedIndex(user_index, LookUpDateIndex(date_values))


def safe_datetime64_cast(date_values):
    """
    >>> d = safe_datetime64_cast(['2014-01-14', np.nan, 'DD', None, '2014-01-14 - bug'])
    >>> d
    array(['2014-01-14', 'NaT', 'NaT', 'NaT', '2014-01-14'], dtype='datetime64[D]')
    >>> np.isnat(d)
    array([False,  True,  True,  True, False], dtype=bool)
    >>> safe_datetime64_cast(['2014/01/14', np.nan, 'DD', None, '2014/01/14 - 5 hahha'])
    array(['2014-01-14', 'NaT', 'NaT', 'NaT', '2014-01-14'], dtype='datetime64[D]')
    """
    if isinstance(date_values, np.ndarray) and date_values.dtype.kind == 'M':
        return np.asarray(date_values, dtype="datetime64[D]")
    else:
        try:
            date_values = np.asarray(date_values, dtype="S10")
        except:
            pass
        try:
            return np.asarray(date_values, dtype="datetime64[D]")
        except ValueError:
            return to_datetime(date_values, errors='coerce', infer_datetime_format=True, box=False)\
                        .astype("datetime64[D]")


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


class LookUpDateIndex(object):
    def __init__(self, date_values):
        self.date_values = safe_datetime64_cast(date_values)
        self.min_date = np.min(self.date_values)  # this will ignore NaT
        self.max_date = np.max(self.date_values)
        self.mask = np.logical_not(np.isnat(self.date_values))

    def decay(self, d, half_life):
        d = safe_datetime64_cast(d)
        d_mask = np.logical_not(np.isnat(d))

        min_date = min(d.min(), self.min_date)
        max_date = max(d.max(), self.max_date)
        delta = (max_date - min_date + 1).astype(int)

        unique_decayed = 2. ** (-np.arange(delta, dtype=np.float) / half_life)

        column_decay = np.zeros(len(self.date_values), dtype=np.float)
        ind_self = (self.date_values[self.mask] - min_date).view(np.int)
        column_decay[self.mask] = (1. / unique_decayed)[ind_self]

        row_decay = np.zeros(len(d), dtype=np.float)
        ind_d = (d[d_mask] - min_date).view(np.int)
        row_decay[d_mask] = unique_decayed[ind_d]

        return row_decay, column_decay


class SortedDateIndex(LookUpDateIndex):

    def __init__(self, date_values):
        super(SortedDateIndex, self).__init__(date_values)
        assert is_increasing(self.date_values)
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


class BaseTruncatedIndex(object):

    def __init__(self, user_index, date_index):
        self.user_index = user_index
        self.date_index = date_index

    def ks_get_batch_window_indices(self, u, d, lower=-1, upper=0, truncation='all', half_life=None):
        indices, indptr = self.get_batch_window_indices(u, d, lower, upper, truncation)

        data = np.ones(len(indices), dtype=np.float64)
        ks = KarmaSparse((data, indices, indptr), format="csr",
                         shape=(len(u), self.user_index.indptr[-1]),
                         copy=False, has_sorted_indices=True, has_canonical_format=True)
        if half_life is not None:
            row_decay, column_decay = self.date_index.decay(d, half_life=half_life)
            ks = ks.scale_along_axis_inplace(row_decay, axis=1)\
                   .scale_along_axis_inplace(column_decay, axis=0)

        return compactify_on_right(ks)

    def get_batch_window_indices_with_intensity(self, u, d, lower=-1, upper=0, half_life=None):
        return self.ks_get_batch_window_indices(u, d, lower, upper,
                                                truncation='all', half_life=half_life)

    def get_first_batch_window_indices_with_intensities(self, u, d, lower=-1, upper=0, half_life=None):
        return self.ks_get_batch_window_indices(u, d, lower, upper,
                                                truncation='first', half_life=half_life)

    def get_last_batch_window_indices_with_intensities(self, u, d, lower=-1, upper=0, half_life=None):
        return self.ks_get_batch_window_indices(u, d, lower, upper,
                                                truncation='last', half_life=half_life)


class LookUpTruncatedIndex(BaseTruncatedIndex):

    _truncation_method_map = {'first': first_indices_lookup,
                              'last': last_indices_lookup,
                              'all': indices_truncation_lookup}

    def get_batch_window_indices(self, u, d, lower=-1, upper=0, truncation='all'):
        assert len(d) == len(u)
        assert lower < upper
        truncation_method = self._truncation_method_map[truncation]

        # This query can be factorized by unique users !
        indices, indptr = self.user_index.get_batch_indices(u)
        local_dates = np.ascontiguousarray(self.date_index.date_values.view(np.int))
        local_dates[np.logical_not(self.date_index.mask)] -= max(lower, upper) + 1  # to avoid overflow

        d = safe_datetime64_cast(d)
        d_mask_dirty = np.isnat(d)
        d = d.astype(np.int)
        d[d_mask_dirty] -= max(lower, upper) + 1  # to avoid overflow

        indices, indptr = truncation_method(indices, indptr, local_dates,
                                            d, lower, upper)

        # # XXX : hack to resize memory in place !
        indices.resize(indptr[-1])
        return indices, indptr


class SortedTruncatedIndex(BaseTruncatedIndex):

    _truncation_method_map = {'first': first_indices_sorted,
                              'last': last_indices_sorted,
                              'all': indices_truncation_sorted}

    def get_batch_window_indices(self, u, d, lower=-1, upper=0, truncation='all'):
        assert len(d) == len(u)
        truncation_method = self._truncation_method_map[truncation]

        # This query can be factorized by unique users !
        indices, indptr = self.user_index.get_batch_indices(u)
        lower_bound, upper_bound = self.date_index.get_window_indices(d, lower, upper)
        indices, indptr = truncation_method(indices, indptr, lower_bound, upper_bound)
        indices.resize(indptr[-1])
        return indices, indptr


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
            ii = htable.Int64HashTable(nb_unique_indices * 2)
            ii.map_locations(required_indices.astype(np.int, copy=False))
            ks_compact = KarmaSparse((ks.data, ii.lookup(ks.indices.astype(np.int)), ks.indptr),
                                     format='csr', shape=(ks.shape[0], nb_unique_indices), copy=False,
                                     has_canonical_format=True, has_sorted_indices=True)
            return required_indices, ks_compact
        else:
            return required_indices, ks[:, required_indices]
    else:
        return np.arange(ks.shape[1], dtype=np.int), ks

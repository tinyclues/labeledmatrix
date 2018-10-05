
from cyperf.indexing.indexed_list import is_increasing
from cyperf.matrix.routine import (indices_truncation_sorted, first_indices_sorted, last_indices_sorted,
                                   indices_truncation_lookup, first_indices_lookup, last_indices_lookup)
import numpy as np
from cyperf.matrix.karma_sparse import KarmaSparse, DTYPE
from pandas.core.algorithms import htable
from pandas import to_datetime
from multiprocessing.pool import ThreadPool

MaxSizeHtable = 10 ** 6
MinCompression = 0.9
NbThreads = 4


def _merge_ks_struct(series):
    if len(series) > 1:
        indices = np.concatenate([x[0] for x in series])
        indptr = np.cumsum(np.concatenate([[0]] + [np.diff(x[1]) for x in series]))
    else:
        indices, indptr = series[0]
    return indices, indptr


def _slice_batches(length, n):
    size = length / n
    if size == 0:
        return [slice(0, length)]
    else:
        return [slice(i * size, length if i + 1 == n else (i + 1) * size) for i in xrange(n)]


def two_integer_array_deduplication(arr1, arr2, shift=1):
    """
    Factorize for zip(arr1, arr2) where arr1, arr2 are both integer arrays
    (see also MultiIndex._build_index method)

    # shift is need, since np.iinfo(np.int).min is ignore by pandas.htable

    returns : reversed_indices, (unique_values1, unique_values2)
    """
    assert len(arr1) == len(arr2)
    arr1 = np.asarray(arr1, dtype=np.int)
    arr2 = np.asarray(arr2, dtype=np.int)

    max_size = min(len(arr1), MaxSizeHtable)
    fz = htable.Int64Factorizer(max_size)
    ind1, u1 = fz.factorize(arr1 + shift), fz.uniques.to_array()
    u1 -= shift

    fz = htable.Int64Factorizer(max_size)
    ind2, u2 = fz.factorize(arr2 + shift), fz.uniques.to_array()
    u2 -= shift

    coef = max(len(u1), len(u2))
    val12 = ind1 * coef  # check overflow ?..
    val12 += ind2

    fz = htable.Int64Factorizer(len(u1) + len(u2))
    val12, u12 = fz.factorize(val12), fz.uniques.to_array()
    del fz

    # doing maximum inplace
    # first factor
    au1 = u12 / coef
    au1 = u1[au1]
    # second factor
    u12 %= coef  # inplace "au2"
    u12 = u2[u12]

    return val12, (au1, u12)


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
    >>> d = safe_datetime64_cast(['2014-01-14', np.nan, 'DD', None, '2014-01-14'])
    >>> d
    array(['2014-01-14',        'NaT',        'NaT',        'NaT',
           '2014-01-14'], dtype='datetime64[D]')
    >>> np.isnat(d)
    array([False,  True,  True,  True, False])
    >>> safe_datetime64_cast(['2014/01/14', np.nan, 'DD', None, '2014/01/14'])
    array(['2014-01-14',        'NaT',        'NaT',        'NaT',
           '2014-01-14'], dtype='datetime64[D]')
    """
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

        column_decay = np.zeros(len(self.date_values), dtype=DTYPE)
        row_decay = np.zeros(len(d), dtype=DTYPE)

        if np.any(d_mask):  # not all d values are dirty

            min_date = min(d.min(), self.min_date)
            max_date = max(d.max(), self.max_date)
            delta = (max_date - min_date + 1).astype(int)

            unique_decayed = 2. ** (-np.arange(delta, dtype=DTYPE) / half_life)

            ind_self = (self.date_values[self.mask] - min_date).view(np.int)
            column_decay[self.mask] = (1. / unique_decayed)[ind_self]

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

        self.interval_indices = np.hstack([self.interval_indices, [len(self.date_values)]])

    def get_window_indices(self, d, lower=-1, upper=0):
        """
        Returns indices for day such that lower <= day - d  < upper
        """
        assert lower < upper
        d = safe_datetime64_cast(d)

        lower_bound = self.get_lower_window_indices(d, lower)
        upper_bound = self.get_upper_window_indices(d, upper)

        return lower_bound, upper_bound

    def get_upper_window_indices(self, d, upper=0):
        d = safe_datetime64_cast(d)

        d_upper = (d + upper).clip(self.min_date, self.max_date + 1)
        d_upper = (d_upper - self.min_date).view(np.int)

        return self.interval_indices[d_upper]

    def get_lower_window_indices(self, d, lower=0):
        d = safe_datetime64_cast(d)

        d_lower = (d + lower).clip(self.min_date, self.max_date + 1)
        d_lower = (d_lower - self.min_date).view(np.int)

        return self.interval_indices[d_lower]


class BaseTruncatedIndex(object):

    def __init__(self, user_index, date_index):
        self.user_index = user_index
        self.date_index = date_index

    def ks_get_batch_window_indices(self, u, d, lower=-1, upper=0, truncation='all', half_life=None, nb=None):
        indices, indptr, repeated_indices, d = self._get_batch_window_indices(u, d, lower, upper, truncation)

        data = np.ones(len(indices), dtype=DTYPE)
        ks = KarmaSparse((data, indices, indptr), format="csr",
                         shape=(len(d), self.user_index.indptr[-1]),
                         copy=False, has_sorted_indices=True, has_canonical_format=True)
        if half_life is not None:
            row_decay, column_decay = self.date_index.decay(d, half_life=half_life)
            ks = ks.scale_along_axis_inplace(row_decay, axis=1)\
                   .scale_along_axis_inplace(column_decay, axis=0)

        if nb is not None:
            # this can be done inside cython routine earlier to be more memory efficient ...
            ks = ks.truncate_by_count(nb, axis=1)
        target_indices, ks_compact = compactify_on_right(ks)

        return target_indices, ks_compact, repeated_indices

    def get_batch_window_indices_with_intensity(self, u, d, lower=-1, upper=0, half_life=None, nb=None):
        return self.ks_get_batch_window_indices(u, d, lower, upper,
                                                truncation='all', half_life=half_life, nb=nb)

    def get_first_batch_window_indices_with_intensities(self, u, d, lower=-1, upper=0, half_life=None, nb=None):
        return self.ks_get_batch_window_indices(u, d, lower, upper,
                                                truncation='first', half_life=half_life, nb=nb)

    def get_last_batch_window_indices_with_intensities(self, u, d, lower=-1, upper=0, half_life=None, nb=None):
        return self.ks_get_batch_window_indices(u, d, lower, upper,
                                                truncation='last', half_life=half_life, nb=nb)


class LookUpTruncatedIndex(BaseTruncatedIndex):

    _truncation_method_map = {'first': first_indices_lookup,
                              'last': last_indices_lookup,
                              'all': indices_truncation_lookup}

    def _get_batch_window_indices(self, u, d, lower=-1, upper=0, truncation='all'):
        assert len(d) == len(u)
        assert lower < upper
        truncation_method = self._truncation_method_map[truncation]

        positions = self.user_index._get_positions(u)
        local_dates = self.date_index.date_values.view(np.int)

        d = safe_datetime64_cast(d).view(np.int)
        repeated_indices, (positions_, d_) = two_integer_array_deduplication(positions, d)
        if len(positions_) < MinCompression * len(positions):
            positions, d = positions_, d_
        else:
            del positions_, d_
            repeated_indices = None

        def partial_ks(slice_):
            return truncation_method(positions[slice_], self.user_index.indices,
                                     self.user_index.indptr, local_dates, d[slice_], lower, upper)

        if len(positions) >= NbThreads:  # Parallel partition + merge
            pp = ThreadPool(NbThreads)
            indices, indptr = _merge_ks_struct(pp.map(partial_ks,
                                                      _slice_batches(len(positions), NbThreads)))
            pp.close()
            pp.terminate()
        else:
            indices, indptr = partial_ks(slice(None))

        # we may want to make a switch based on the compression rate
        return indices, indptr, repeated_indices, d.view('M8[D]')


class SortedTruncatedIndex(BaseTruncatedIndex):

    _truncation_method_map = {'first': first_indices_sorted,
                              'last': last_indices_sorted,
                              'all': indices_truncation_sorted}

    def _get_batch_window_indices(self, u, d, lower=-1, upper=0, truncation='all'):
        assert len(d) == len(u)
        assert lower < upper
        truncation_method = self._truncation_method_map[truncation]

        positions = self.user_index._get_positions(u)

        d = safe_datetime64_cast(d).view(np.int)
        repeated_indices, (positions_, d_) = two_integer_array_deduplication(positions, d)
        if len(positions_) < MinCompression * len(positions):
            positions, d = positions_, d_
        else:
            del positions_, d_
            repeated_indices = None

        lower_bound, upper_bound = self.date_index.get_window_indices(d.view('M8[D]'), lower, upper)

        def partial_ks(slice_):
            return truncation_method(positions[slice_], self.user_index.indices,
                                     self.user_index.indptr, lower_bound[slice_], upper_bound[slice_])

        if len(positions) >= NbThreads:  # Parallel partition + merge
            pp = ThreadPool(NbThreads)
            indices, indptr = _merge_ks_struct(pp.map(partial_ks,
                                                      _slice_batches(len(positions), NbThreads)))
            pp.close()
            pp.terminate()
        else:
            indices, indptr = partial_ks(slice(None))

        # we may want to make a switch based on the compression rate
        return indices, indptr, repeated_indices, d.view('M8[D]')


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
            ii = htable.Int64HashTable(min(nb_unique_indices * 2, MaxSizeHtable))
            ii.map_locations(required_indices.astype(np.int, copy=False))
            ks_compact = KarmaSparse((ks.data, ii.lookup(ks.indices.astype(np.int)), ks.indptr),
                                     format='csr', shape=(ks.shape[0], nb_unique_indices), copy=False,
                                     has_canonical_format=True, has_sorted_indices=True)
            return required_indices, ks_compact
        else:
            return required_indices, ks[:, required_indices]
    else:
        return np.arange(ks.shape[1], dtype=np.int), ks

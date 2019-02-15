from __future__ import absolute_import
from __future__ import division   # see https://www.python.org/dev/peps/pep-0238/#abstract
import numpy as np
from cyperf.tools.types import get_open_mp_num_thread
from cyperf.tools import parallel_unique
from cyperf.indexing.indexed_list import is_increasing

from cyperf.matrix.routine import indices_truncation_lookup
from cyperf.matrix.karma_sparse import KarmaSparse

from pandas.core.algorithms import htable
from pandas import to_datetime
from multiprocessing.pool import ThreadPool
from six.moves import range

MaxSizeHtable = 10 ** 6
MinCompression = 0.9


def _merge_ks_struct(series):
    if len(series) > 1:
        deltas = np.concatenate([x[0] for x in series])
        indices = np.concatenate([x[1] for x in series])
        indptr = np.cumsum(np.concatenate([[0]] + [np.diff(x[2]) for x in series]))
    else:
        deltas, indices, indptr = series[0]
    return deltas, indices, indptr


def _slice_batches(length, n):
    size = length // n
    if size == 0:
        return [slice(0, length)]
    else:
        return [slice(i * size, length if i + 1 == n else (i + 1) * size) for i in range(n)]


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
    au1 = u12 // coef
    au1 = u1[au1]
    # second factor
    u12 %= coef  # inplace "au2"
    u12 = u2[u12]

    return val12, (au1, u12)


def create_truncated_index(user_index, date_values):
    return LookUpTruncatedIndex(user_index, date_values)


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
        return to_datetime(date_values, errors='coerce',
                           infer_datetime_format=True, box=False).astype("datetime64[D]")


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
        self.min_date = np.min(self.date_values)  # this will ignore NaT
        self.max_date = np.max(self.date_values)
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


class LookUpTruncatedIndex(object):
    def __init__(self, user_index, source_dates):
        """
        we're  joining `source` onto `target`. The variables are named accordingly.
        :param user_index: ColumnIndex, index of the user key on the `source` table
        :param source_dates: array, `source` dates
        """
        assert user_index.indices.shape[0] == len(source_dates)
        if len(source_dates) == 0:
            raise ValueError('Empty data is not supported')
        self.user_index = user_index

        source_dates = safe_datetime64_cast(source_dates)
        if np.all(np.isnat(source_dates)):
            raise ValueError('All dates are NaT after conversion: check the input date format!')
        self.source_dates = source_dates

    def get_batch_window_indices_with_intensity(self, target_users, target_dates,
                                                lower=-1, upper=0, half_life=None, nb=None):
        """
        :param target_users: iterable, user keys on the target table
        :param target_dates: iterable, date keys on the target table
        :param lower: int, relative lower bound of the date window
        :param upper: int, relative upper bound of the date window
        :param half_life: float or None, half_life to be applied to ks_intensity
        :param nb: int, number of elements to keep
            if > 0: keeps only the most recent ones (closer to upper)
                if == 1: equivalent to last
            if < 0: keeps only the most ancient ones (closer to lower)
                if == -1: equivalent to first
            if 0 or None: keeps all of them

        :return: (source_indices, ks_intensity, repeated_indices)
            source_indices: array, indices of `source` table corresponding to the columns of ks_intensity
            ks_intensity: KarmaSparse, matrix mapping each unique `target` line
                            to the intensities of the different `source` lines
            repeated_indices: array, line of ks_intensity for each `target` line
        """
        assert len(target_dates) == len(target_users)
        assert lower < upper
        nb = nb or 0
        half_life = half_life or 0

        target_users_position_in_source_index = self.user_index._get_positions(target_users)
        source_dates = self.source_dates.view(np.int)

        target_dates = safe_datetime64_cast(target_dates).view(np.int)

        repeated_indices, (target_users_position_in_source_index_, target_dates_) = two_integer_array_deduplication(
            target_users_position_in_source_index, target_dates)
        if len(target_users_position_in_source_index_) < MinCompression * len(target_users_position_in_source_index):
            target_users_position_in_source_index, target_dates = target_users_position_in_source_index_, target_dates_
        else:
            del target_users_position_in_source_index_, target_dates_
            repeated_indices = None

        def partial_ks(slice_):
            return indices_truncation_lookup(target_users_position_in_source_index[slice_], target_dates[slice_],
                                             self.user_index.indices, self.user_index.indptr, source_dates,
                                             lower, upper, half_life, nb)

        nb_threads = get_open_mp_num_thread()
        if len(target_users_position_in_source_index) >= nb_threads:  # Parallel partition + merge
            pp = ThreadPool(nb_threads)
            data, indices, indptr = _merge_ks_struct(pp.map(partial_ks,
                                                              _slice_batches(len(target_users_position_in_source_index),
                                                                             nb_threads)))
            pp.close()
            pp.terminate()
        else:
            data, indices, indptr = partial_ks(slice(None))

        ks = KarmaSparse((data, indices, indptr), format="csr",
                         shape=(len(indptr) - 1, self.user_index.indptr[-1]),
                         copy=False, has_sorted_indices=False, has_canonical_format=False)
        source_indices, ks_compact = compactify_on_right(ks)

        # we may want to make a switch based on the compression rate
        return source_indices, ks_compact, repeated_indices


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
            required_indices = parallel_unique(ks.indices)
        else:
            required_indices = parallel_unique(ks.nonzero()[1])
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

import numpy as np
from cyperf.tools import take_indices, parallel_sort

from cyperf.indexing.column_index import (factorize_inplace, groupsort_indexer,
                                          get_size_batch, get_batch_indices, get_first_batch, get_positions,
                                          sorted_indices, quantiles_indices, quantiles_indices_with_most_common,
                                          quantiles_indices_with_first, count,
                                          deduplicate_indices, get_keys_indices,
                                          unique_indices_inplace, get_unique_indices)

from cyperf.indexing.column_index import (positions_select_inplace, count_select, deduplicate_indices_select,
                                          reversed_index_select, get_keys_indices_select, compact_select)

from cyperf.indexing.column_index import (get_positions_multiindex, key_indices_multiindex,
                                          compact_multiindex, dispatch_tupled_values)


def get_index_dtype(size):
    if size > np.iinfo(np.int32).max:
        return np.int64
    else:
        return np.int32


def is_iterable(inp):
    try:
        iter(inp)
        return True
    except TypeError:
        return False


def unique_indices(values, reverse=False):
    """
    >>> unique_indices([3, 3, 2, 0, 2, 1, 0], False)
    array([1, 4, 5, 6], dtype=int32)
    >>> unique_indices([3, 3, 2, 0, 2, 1, 0], True)
    array([0, 2, 3, 5], dtype=int32)
    >>> unique_indices(np.array([3, 3, 2, 0, 2, 1, 0]), False)
    array([1, 4, 5, 6], dtype=int32)
    >>> unique_indices(tuple([3, 3, 2, 0, 2, 1, 0]), True)
    array([0, 2, 3, 5], dtype=int32)
    """
    nb = len(values)
    position = np.zeros(nb, dtype=get_index_dtype(nb))
    n_keys = unique_indices_inplace(values, reverse, position)
    return position[:n_keys].copy()


def factorize(values):
    """
    Factorises given list, tuple or np.array
    Args:
        values: ITER fused type with values
    Returns: * dict {key: int} of unique keys from values,
             * numpy array representing values using dict of positions,
             * number of different values
    """
    nb = len(values)
    reversed_indices = np.zeros(nb, dtype=get_index_dtype(nb))
    position = factorize_inplace(values, reversed_indices)
    n_keys = len(position)
    # in this case we don't need to have int64 array for reversed_indices
    reversed_indices = reversed_indices.astype(get_index_dtype(n_keys), copy=False)
    return position, reversed_indices, n_keys


class ColumnIndex(object):
    def __init__(self, *args):
        """
        >>> key_position, reversed_indices, n_keys = factorize(['c', 'b', 'a', 'a', 'b', 'b', 'd'])
        >>> sorted(key_position.keys())
        ['a', 'b', 'c', 'd']
        >>> reversed_indices
        array([0, 1, 2, 2, 1, 1, 3], dtype=int32)
        >>> n_keys
        4
        >>> indptr = np.zeros(n_keys + 1, dtype=np.int32)
        >>> length, indices = groupsort_indexer(indptr, reversed_indices)
        >>> indptr
        array([0, 1, 4, 6, 7], dtype=int32)
        >>> length
        4
        >>> indices
        array([0, 1, 4, 5, 2, 3, 6], dtype=int32)
        """
        if len(args) == 5 and isinstance(args[1], dict):
            # In this case constructor have received tuple of 5 attributes:
            # reference to Backend.__getitem__ method, position, indices, reversed_indices, indptr
            self._values = args[0]
            self.position = args[1]
            self.indices = args[2]
            self.reversed_indices = args[3]
            self.indptr = args[4]
        elif len(args) == 1 and hasattr(args[0], '__call__'):
            # the main purpose of this index is for a given list (numpy array or other data structure)
            # to obtain indices of a given value
            # let [c, b, a, a, b, b, d] be the list of values of a Column which we pass to ColumnIndex constructor
            # factorize routine will construct the auxiliary position dict, where to every value we will assign unique
            # integer: in our example position = {c: 0, b: 1, a: 2, d: 3}
            # then in reversed_indices we will have same values with every value replaced by respective integer
            # reversed_indices = [0, 1, 2, 2, 1, 1, 3]
            # finally we will count number of times every value present in list
            # counts = [1, 3, 2, 1] and we define indptr as cumulated sum of counts
            # indptr = [0, 1, 4, 6, 7]
            # next step is to construct array with indices, where we will stock indices of each value in slices
            # indices = groupsort_indexer(indptr, reversed_indices) = [0, 1, 4, 5, 2, 3, 6]
            # where indices of c are [0], of b: [1, 4, 5], of a: [2, 3] and of d: [6]
            # now to obtain indices of a given value u we firstly find its respective integer as pos = position[u]
            # next we find its slice as indptr[pos] : indptr[pos + 1]
            # and finally we take indices[indptr[pos]: indptr[pos + 1]] (see __getitem__(self, value))

            # we check if the the first argument is a method (Backend._getitem__)
            self._values = args[0]

            self.position, self.reversed_indices, n_keys = factorize(self._values(slice(None, None, None)))
            self.indptr = np.zeros(n_keys + 1, dtype=get_index_dtype(len(self.reversed_indices) + 1))
            _, self.indices = groupsort_indexer(self.indptr, self.reversed_indices)
        elif len(args) == 1 and is_iterable(args[0]):  # raw values in args[0]
            values = args[0]
            ll = len(values)

            def call_method(indices):
                return take_indices(values, indices, ll)

            self.__init__(call_method)
        else:
            raise ValueError(args)

    # TODO we need to serialize ColumnIndex together with Column, as we need a ref to backend.__getitem__() method

    def _get_position(self, value):
        """
        Returns position in index for a given value
        """
        return self._get_positions([value])[0]

    def _get_positions(self, values):
        """
        Returns positions in index for a given iterable of values
        """
        return get_positions(self.position, values, self.reversed_indices)

    def __getitem__(self, value):
        """
        >>> col = ['b', 'a', 'a', 'c', 'b', 'd']
        >>> my_index = ColumnIndex(col)
        >>> my_index['b']
        array([0, 4], dtype=int32)
        >>> my_index['e']
        array([], dtype=int32)
        >>> selection = [2, 4, 5]
        >>> select_col = take_indices(col, selection)
        >>> index_select = my_index.select(selection, select_col)
        >>> index_select['a']
        array([0], dtype=int32)
        >>> index_select['c']
        array([], dtype=int32)
        """
        pos = self._get_position(value)
        if pos == -1:
            return np.array([], dtype=self.indices.dtype)
        return self.indices[self.indptr[pos]:self.indptr[pos + 1]]

    @property
    def compression_rate(self):
        """
        >>> my_index = ColumnIndex(['b', 'a', 'a', 'b'])
        >>> my_index.compression_rate
        0.5
        >>> my_index = ColumnIndex(['b', 'a', 'a', 'a', 'a'])
        >>> round(my_index.compression_rate, 3)
        0.4
        """
        if self.indices.shape[0] == 0:
            # case of empty index
            return 0.0
        return float(len(self)) / self.indices.shape[0]

    def __len__(self):
        """
        >>> len(ColumnIndex(['b', 'a', 'a', 'c', 'b', 'd']))
        4
        """
        return len(self.position)

    def __contains__(self, value):
        """
        >>> my_index = ColumnIndex(['b', 'a', 'a', 'c', 'b', 'd'])
        >>> 'i' in my_index
        False
        >>> 'a' in my_index
        True
        >>> ['a'] in my_index
        False
        """
        try:
            hash(value)
            return self._get_position(value) != -1
        except TypeError:
            return False

    def __repr__(self):
        return '{} for {} values with compression rate of {}'.format(
            self.__class__, self.indices.shape[0], self.compression_rate)

    def get_size(self, value):
        """
        >>> col = ['b', 'a', 'a', 'c', 'b', 'd']
        >>> my_index = ColumnIndex(col)
        >>> my_index.get_size('b')
        2
        >>> my_index.get_size(4)
        0
        >>> selection = [2, 4, 5]
        >>> select_col = take_indices(col, selection)
        >>> index_select = my_index.select(selection, select_col)
        >>> index_select.get_size('c')
        0
        """
        pos = self._get_position(value)
        if pos == -1:
            return 0
        return self.indptr[pos + 1] - self.indptr[pos]

    def get_size_batch(self, values):
        """
        >>> col = ['b', 'a', 'a', 'c', 'b', 'd']
        >>> my_index = ColumnIndex(col)
        >>> my_index.get_size_batch(['b', 'f', 'a', 'e'])
        array([2, 0, 2, 0], dtype=int32)
        >>> selection = [2, 4, 5]
        >>> select_col = take_indices(col, selection)
        >>> index_select = my_index.select(selection, select_col)
        >>> index_select.get_size_batch(['b', 'c'])
        array([1, 0], dtype=int32)
        """
        positions = self._get_positions(values)
        return get_size_batch(positions, self.indptr)

    def get_batch_indices(self, values):
        """
        >>> col = ['b', 'a', 'a', 'c', 'b', 'd']
        >>> my_index = ColumnIndex(col)
        >>> my_index.get_batch_indices(['b', 'f', 'a', 'e'])
        (array([0, 4, 1, 2], dtype=int32), array([0, 2, 2, 4, 4], dtype=int32))
        >>> my_index.get_batch_indices(['b', 'b', 'b'])
        (array([0, 4, 0, 4, 0, 4], dtype=int32), array([0, 2, 4, 6], dtype=int32))
        >>> selection = [2, 4, 5]
        >>> select_col = take_indices(col, selection)
        >>> index_select = my_index.select(selection, select_col)
        >>> index_select.get_batch_indices(['a', 'f', 'b', 'c'])
        (array([0, 1], dtype=int32), array([0, 1, 1, 2, 2], dtype=int32))
        """
        positions = self._get_positions(values)
        result, indptr = get_batch_indices(positions, self.indptr, self.indices)
        return result, indptr.astype(get_index_dtype(indptr[-1]), copy=False)

    def get_unique_indices(self, values):
        """
        return unique sorted indices that map to "values"

        >>> my_index = ColumnIndex(['b', 'a', 'a',  'c', 'b', 'd'])
        >>> my_index.get_unique_indices(['c', 'b'])
        array([0, 3, 4], dtype=int32)
        >>> my_index.get_unique_indices(['d', 'a', 'a', 'a'])
        array([1, 2, 5], dtype=int32)
        >>> my_index.get_unique_indices(['d', 'b', 'c'])
        array([0, 3, 4, 5], dtype=int32)
        """
        positions = self._get_positions(values)
        return get_unique_indices(positions, self.indptr, self.indices)

    def get_first(self, value):
        """
        >>> col = ['b', 'a', 'a', 'c', 'b', 'd']
        >>> my_index = ColumnIndex(col)
        >>> my_index.get_first('b')
        0
        >>> my_index.get_first('e')
        -1
        >>> selection = [2, 4, 5]
        >>> select_col = take_indices(col, selection)
        >>> index_select = my_index.select(selection, select_col)
        >>> index_select.get_first('c')
        -1
        """
        pos = self._get_position(value)
        if pos == -1:
            return -1
        return self.indices[self.indptr[pos]]

    def get_first_batch(self, values):
        """
        >>> col = ['b', 'a', 'a', 'c', 'b', 'd']
        >>> my_index = ColumnIndex(col)
        >>> my_index.get_first_batch(['b', 'f', 'a', 'e'])
        array([ 0, -1,  1, -1], dtype=int32)
        >>> selection = [2, 4, 5]
        >>> select_col = take_indices(col, selection)
        >>> index_select = my_index.select(selection, select_col)
        >>> index_select.get_first_batch(['a', 'c', 'h'])
        array([ 0, -1, -1], dtype=int32)
        """
        positions = self._get_positions(values)
        return get_first_batch(positions, self.indptr, self.indices)

    def count(self):
        """
        >>> my_index = ColumnIndex(['b', 'a', 'a', 'c', 'b', 'd'])
        >>> my_index.count() == {'a': 2, 'c': 1, 'b': 2, 'd': 1}
        True
        """
        keys_indices = get_keys_indices(self.indptr, self.indices)
        return count(self._values(keys_indices), self.indptr, self.indices)

    def deduplicate_indices(self, take='first'):
        """
        >>> my_index = ColumnIndex(['b', 'a', 'a', 'c', 'b', 'd'])
        >>> my_index.deduplicate_indices('first')
        array([0, 1, 3, 5], dtype=int32)
        >>> my_index.deduplicate_indices('last')
        array([2, 3, 4, 5], dtype=int32)
        """
        return deduplicate_indices(self.indptr, self.indices, take)

    def reversed_index(self):
        """
        Get factorised column representation; used as pivot primitive
        Returns: list of keys and np.array of reversed indices

        >>> my_index = ColumnIndex(['b', 'a', 'a', 'c', 'b', 'd'])
        >>> u, ind = my_index.reversed_index()
        >>> u
        ['b', 'a', 'c', 'd']
        >>> ind
        array([0, 1, 1, 2, 0, 3], dtype=int32)
        >>> [u[x] for x in ind]
        ['b', 'a', 'a', 'c', 'b', 'd']
        """
        keys_positions = get_keys_indices(self.indptr, self.indices)
        return self._values(keys_positions), self.reversed_indices

    def __iter__(self):
        return self.keys().__iter__()

    def keys(self):
        """
        >>> sorted(ColumnIndex(['b', 'a', 'a', 'c', 'b', 'd']).keys())
        ['a', 'b', 'c', 'd']
        """
        return list(self.position.keys())

    def sorted_indices(self, reverse=False):
        """
        >>> col = [4, 3, 5, 1, 8, 10, 0, 1, 10, 8, 7, 6, 2, 2, 9, 2, 5, 1, 5, 4]
        >>> index = ColumnIndex(col)
        >>> selection = [0, 2, 3, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        >>> select_col = take_indices(col, selection)
        >>> index_select = index.select(selection, select_col)
        >>> index_select.sorted_indices()
        array([ 2,  3, 10,  5,  6,  8,  0, 12,  1,  9, 11,  4,  7], dtype=int32)
        >>> index_select.sorted_indices(reverse=True)
        array([ 7,  4,  1,  9, 11,  0, 12,  5,  6,  8,  2,  3, 10], dtype=int32)
        >>> take_indices(col, take_indices(selection, index_select.sorted_indices(reverse=True)))
        [9, 8, 5, 5, 5, 4, 4, 2, 2, 2, 1, 1, 1]

        >>> col = list(map(str, [4, 3, 5, 3, 2, 2]))
        >>> index_str = ColumnIndex(col)
        >>> index_str.sorted_indices(reverse=True)
        array([2, 0, 1, 3, 4, 5], dtype=int32)
        >>> index_str.sorted_indices(reverse=False)
        array([4, 5, 1, 3, 0, 2], dtype=int32)
        """
        sorted_keys = parallel_sort(self.keys(), reverse=reverse)
        sorted_keys_positions = self._get_positions(sorted_keys)
        return sorted_indices(sorted_keys_positions, self.indptr, self.indices)

    def _quantiles_indices(self, sorted_keys_positions, nb, label, actual_indptr, actual_indices):
        if len(self) == 0:
            raise ValueError('Impossible to compute quantiles on an empty index')
        if label == 'most_common':
            return quantiles_indices_with_most_common(sorted_keys_positions, actual_indptr, actual_indices,
                                                      self.indptr, nb)
        elif label == 'first':
            return quantiles_indices_with_first(sorted_keys_positions, actual_indptr, actual_indices, self.indptr, nb)
        else:
            return quantiles_indices(sorted_keys_positions, actual_indptr, actual_indices, self.indptr, nb)

    def quantiles(self, nb, label=None):
        """
        >>> index = ColumnIndex([0.78, 0.78, 0.5, 0.87, 0.87, 0.9, 0.6, 0.6, 0.9, 0.9])
        >>> index.quantiles(4, label='most_common')
        ([0.6, 0.78, 0.9], [0.6, 0.78, 0.87, 0.9])
        >>> index.quantiles(4, label='first')
        ([0.6, 0.78, 0.9], [0.5, 0.78, 0.87, 0.9])
        """
        keys = self.keys()
        keys.sort()
        sorted_keys_positions = self._get_positions(keys)
        q_indices = self._quantiles_indices(sorted_keys_positions, nb, label, self.indptr, self.indices)
        if label:
            return self._values(q_indices[0]), self._values(q_indices[1])
        else:
            return self._values(q_indices)

    def select(self, indices, selection_values_method):
        """
        Creates a new SelectIndex for a given iterable of indices or slice
        >>> col = ['b', 'a', 'c', 'a', 'c', 'b', 'b']
        >>> index = ColumnIndex(col)
        >>> selection = [3, 5, 6]
        >>> select_col = take_indices(col, selection)
        >>> index_select = index.select(selection, select_col)
        >>> index_select.parent_values(slice(None, None, None))
        ['b', 'a', 'c', 'a', 'c', 'b', 'b']
        >>> id(index.position) == id(index_select.position)
        True
        >>> index_select.indptr
        array([0, 2, 3, 3], dtype=int32)
        >>> index_select.indices
        array([1, 2, 0], dtype=int32)
        >>> index_select.reversed_indices
        array([1, 0, 0], dtype=int32)
        >>> selection = slice(None, None, 2)
        >>> select_col = take_indices(col, selection)
        >>> slice_select = index.select(selection, select_col)
        >>> slice_select.keys()
        ['b', 'c']
        >>> slice_select.reversed_indices
        array([0, 2, 2, 0], dtype=int32)
        >>> slice_select.indices
        array([0, 3, 1, 2], dtype=int32)
        >>> slice_select.indptr
        array([0, 2, 2, 4], dtype=int32)
        """
        return SelectIndex(self, indices, selection_values_method)

    def compact(self, new_values_method):
        return self


class SelectIndex(ColumnIndex):
    """
    SelectIndex will need properly written column_values_method in parent index (or parent_column_values if parent is
    another SelectIndex)

    It also inherits from parent following auxillary functions: _prepare_values, _prepare_value
    """
    def __init__(self, *args):
        parent_index, selection, selection_values_method = args
        if isinstance(parent_index, SelectIndex):
            self.parent_indices = parent_index.parent_indices
            self.parent_indptr = parent_index.parent_indptr
            self.parent_values = parent_index.parent_values
            self.parent_get_positions = parent_index.parent_get_positions
        elif isinstance(parent_index, ColumnIndex):
            self.parent_indices = parent_index.indices
            self.parent_indptr = parent_index.indptr
            self.parent_values = parent_index._values
            self.parent_get_positions = parent_index._get_positions
        else:
            raise ValueError('First argument must be SelectIndex or ColumnIndex, got {} instead'
                             .format(type(parent_index)))

        reversed_indices = np.ascontiguousarray(take_indices(parent_index.reversed_indices, selection))
        indptr = np.zeros_like(self.parent_indptr)
        self.length, indices = groupsort_indexer(indptr, reversed_indices)
        # as we use old position dict dtype of reversed_indices will be the same
        # however maximal value in indptr corresponds to the size of selection,
        # so it can possibly can be downcasted
        # indices will be constructed with groupsort_indexer function
        # and will have the same dtype as indptr
        indptr = indptr.astype(get_index_dtype(indptr[-1]), copy=False)

        super().__init__(selection_values_method, parent_index.position, indices, reversed_indices, indptr)

    def _get_positions(self, values):
        return positions_select_inplace(self.parent_get_positions(values), self.indptr)

    def __len__(self):
        return self.length

    def count(self):
        """
        >>> col = ['b', 'a', 'c', 'a', 'c', 'b', 'b']
        >>> index = ColumnIndex(col)
        >>> selection = [3, 5, 6]
        >>> select_col = take_indices(col, selection)
        >>> index_select = index.select(selection, select_col)
        >>> index_select.count() == {'a': 1, 'b': 2}
        True
        """
        keys_indices = get_keys_indices_select(self.indptr, self.indices, self.reversed_indices,
                                               self.parent_indices, self.parent_indptr, self.length)
        return count_select(self.parent_values(keys_indices), self.indptr, self.indices, self.reversed_indices)

    def deduplicate_indices(self, take='first'):
        """
        >>> col = ['b', 'a', 'c', 'a', 'c', 'b', 'b']
        >>> index = ColumnIndex(col)
        >>> selection = [3, 5, 6]
        >>> select_col = take_indices(col, selection)
        >>> index_select = index.select(selection, select_col)
        >>> index_select.deduplicate_indices(take='last')
        array([0, 2], dtype=int32)
        """
        return deduplicate_indices_select(self.indptr, self.indices, self.reversed_indices, self.length, take)

    def reversed_index(self):
        """
        Returns: list of keys and np.array of reversed indices.
        Order of keys is that of parent ColumnIndex

        >>> col = np.array(['b', 'a', 'c', 'a', 'c', 'b', 'b'], dtype='|S1')
        >>> index = ColumnIndex(col.__getitem__)
        >>> selection = [3, 5, 6]
        >>> select_col = take_indices(col, selection)
        >>> index_select = index.select(selection, select_col)
        >>> u, ind = index_select.reversed_index()
        >>> np.all(u == np.array(['b', 'a'], dtype='|S1'))
        True
        >>> ind
        array([1, 0, 0], dtype=int32)
        >>> [u[x] for x in ind] == [b'a', b'b', b'b']
        True
        """
        keys_positions, reversed_indices = \
            reversed_index_select(self.indptr, self.indices, self.reversed_indices,
                                  self.parent_indices, self.parent_indptr, self.length)

        return (self.parent_values(keys_positions),
                reversed_indices.astype(get_index_dtype(self.length), copy=False))

    def keys(self):
        """
        Returns: list of keys. Order of keys is that of parent ColumnIndex

        >>> col = ['b', 'a', 'c', 'a', 'c', 'b', 'b']
        >>> index = ColumnIndex(col)
        >>> selection = [3, 5, 6]
        >>> select_col = take_indices(col, selection)
        >>> index_select = index.select(selection, select_col)
        >>> index_select.keys()
        ['b', 'a']
        """
        keys_indices = get_keys_indices_select(self.indptr, self.indices, self.reversed_indices,
                                               self.parent_indices, self.parent_indptr, self.length)
        return self.parent_values(keys_indices)

    def quantiles(self, nb, label=None):
        keys = self.keys()
        keys.sort()
        sorted_keys_positions = self._get_positions(keys)
        q_indices = self._quantiles_indices(sorted_keys_positions, nb, label, self.parent_indptr, self.parent_indices)
        if label:
            return self.parent_values(q_indices[0]), self.parent_values(q_indices[1])
        else:
            return self.parent_values(q_indices)

    def compact(self, new_values_method):
        """
        Creates compacted version of the index, i.e. deletes all keys which are not used in selection
        """
        # since this method is called in unlazy after changing a backend,
        # so a call of new_values_method will give a reference to a list/array/... and no calculations will be made

        position, indices, reversed_indices, indptr = compact_select(new_values_method(slice(None, None, None)),
                                                                     self.indptr, self.indices, self.reversed_indices)

        # we are sure that dtype of indices and indptr should be the same as before compact
        # however since we reconstruct position dict we possibly can downcast reversed_indices:
        reversed_indices = reversed_indices.astype(get_index_dtype(indptr.shape[0] - 1), copy=False)

        return ColumnIndex(new_values_method, position, indices, reversed_indices, indptr)


class MultiIndex(ColumnIndex):
    """
    Class that build an index based on two already built ColumnIndex indices

    Parameters
    ----------
    index_0 : first index
    index_1 : second index

    Examples
    --------
    >>> col_0 = ['c', 'b', 'a', 'a', 'b', 'b', 'd']
    >>> col_1 = ['toto', 'toto', 'tutu', 'toto', 'tutu', 'toto', 'toto']
    >>> index_0 = ColumnIndex(col_0)
    >>> index_1 = ColumnIndex(col_1)
    >>> mi = MultiIndex(index_0, index_1, list(zip(col_0, col_1)).__getitem__)
    >>> mi[('c', 'tutu')]
    array([], dtype=int32)
    >>> mi[('bobo', 'tutuu')]
    array([], dtype=int32)
    >>> mi[('c', 'toto')]
    array([0], dtype=int32)
    >>> mi[('b', 'toto')]
    array([1, 5], dtype=int32)
    """
    def __init__(self, *args):
        """
        We can construct MultiIndex class from two "indexes": by index here we understand a subclass of ColumnIndex
        or a tuple of the signature (position dict, reversed_indices np.ndarray, values_method callable)
        """
        if len(args) != 3:
            raise ValueError('Illegal number of arguments: {}'.format(len(args)))

        self.position_0 = None
        self.reversed_indices_0 = None
        self.get_positions_0 = None

        self.position_1 = None
        self.reversed_indices_1 = None
        self.get_positions_1 = None

        self._init_side_index(0, args[0])
        self._init_side_index(1, args[1])

        int_index = self._build_index()

        super().__init__(args[2], int_index.position, int_index.indices, int_index.reversed_indices, int_index.indptr)

    def _init_side_index(self, i, arg):
        if isinstance(arg, tuple) or isinstance(arg, list):
            if isinstance(arg[0], dict) and isinstance(arg[1], np.ndarray):
                setattr(self, 'position_{}'.format(i), arg[0])
                setattr(self, 'reversed_indices_{}'.format(i), arg[1])
                setattr(self, 'get_positions_{}'.format(i), getattr(self, 'basic_get_positions_{}'.format(i)))
            else:
                raise ValueError('Unknown signature for index {}: {}'.format(i, tuple(type(a) for a in arg)))
        else:
            if isinstance(arg, ColumnIndex):
                setattr(self, 'position_{}'.format(i), arg.position)
                setattr(self, 'reversed_indices_{}'.format(i), arg.reversed_indices)
                setattr(self, 'get_positions_{}'.format(i), arg._get_positions)
            else:
                raise ValueError('Illegal parameter for index {}: {}'.format(i, type(arg)))

    def basic_get_positions_0(self, values):
        return get_positions(self.position_0, values, self.reversed_indices_0)

    def basic_get_positions_1(self, values):
        return get_positions(self.position_1, values, self.reversed_indices_1)

    def _build_index(self):
        self.coeff = max(len(self.position_0), len(self.position_1))
        tuples_to_int = self.coeff * self.reversed_indices_0.astype(get_index_dtype(self.coeff * (self.coeff + 1)))
        tuples_to_int += self.reversed_indices_1

        def _values_method(indices):
            return take_indices(tuples_to_int, indices)

        return ColumnIndex(_values_method)

    # @cached_property
    def _default_missing_value(self):
        # TODO here we try to know which value can be used as missing, i.e. it can't be found in any of parent columns
        for v in [None, False, np.nan, 3.1415, -1, 'a', ('t',), ('t', None)]:
            if self.get_positions_0([v])[0] == -1 and self.get_positions_1([v])[0] == -1:
                return v
        for _ in range(10 ** 7):
            v = np.random.rand()
            if self.get_positions_0([v])[0] == -1 and self.get_positions_1([v])[0] == -1:
                return v
        # Really? if you have this exception that kind of data do you have?!
        raise ValueError('MultiIndex: Could not find any value for _default_missing_value')

    def _get_positions(self, values):
        values_0, values_1 = dispatch_tupled_values(values, self._default_missing_value)
        positions_0 = self.get_positions_0(values_0)
        positions_1 = self.get_positions_1(values_1)

        return get_positions_multiindex(self.position, self.reversed_indices, self.coeff, positions_0, positions_1)

    def keys(self):
        # TODO there were some benchmarks showed that this implementation of keys can be faster than position.keys()
        keys_indices = key_indices_multiindex(self.indptr, self.indices)
        return self._values(keys_indices)

    def compact(self, new_values_method):
        """
        Creates compacted version of the index, i.e. reconstructs new position dict, other primitives are already good
        since we constructed them from equivalent iterable of ints
        """
        # since this method is called in unlazy after changing a backend,
        # so a call of new_values_method will give a reference to a list/array/... and no calculations will be made

        new_position = compact_multiindex(new_values_method(slice(None, None, None)), self.indptr, self.indices,
                                          self.reversed_indices)

        return ColumnIndex(new_values_method, new_position, self.indices, self.reversed_indices, self.indptr)

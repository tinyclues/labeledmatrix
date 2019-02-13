# cython: embedsignature=True
# cython: nonecheck=True
# cython: overflowcheck=True
# cython: unraisable_tracebacks=True
# cython: infer_types=True

import numpy as np
from cyperf.tools.types import ITYPE
from cyperf.tools import parallel_unique

cimport cython
cimport numpy as np
from cpython.dict cimport PyDict_Contains, PyDict_GetItem
from cpython.ref cimport PyObject


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void inplace_reversed_index(ITER values, np.ndarray[ndim=1, dtype=INT1, mode='c'] indices,
                                  dict position, list unique_values) except *:
    """
    >>> values = [4, 4, 'r', 'r', 2]
    >>> indices = np.zeros(len(values), dtype=np.int)
    >>> position = {}
    >>> unique_values = []
    >>> inplace_reversed_index(values, indices, position, unique_values)
    >>> unique_values
    [4, 'r', 2]
    >>> indices
    array([0, 0, 1, 1, 2])
    """
    cdef INT1 nb = len(values), i, ind, count = len(unique_values)
    cdef PyObject *obj
    cdef object val

    assert len(indices) >= nb

    for i in range(nb):
        val = values[i]
        obj = PyDict_GetItem(position, val)
        if obj is not NULL:
            ind = <object>obj
            indices[i] = ind
        else:
            position[val] = count
            indices[i] = count
            unique_values.append(val)
            count += 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple reversed_index(ITER values):
    """
    >>> reversed_index([4, 4, 'r', 'r', 2])
    ([4, 'r', 2], array([0, 0, 1, 1, 2], dtype=int32))
    >>> reversed_index(range(4))
    ([0, 1, 2, 3], array([0, 1, 2, 3], dtype=int32))
    >>> reversed_index(IndexedList(list(range(4))))
    ([0, 1, 2, 3], array([0, 1, 2, 3], dtype=int32))
    """
    if isinstance(values, IndexedList):
        return values, np.arange(len(values), dtype=ITYPE)

    check_values(values)

    cdef ITYPE_t nb = len(values), i, count = 0, ind
    cdef np.ndarray[ndim=1, dtype=ITYPE_t] indices = np.zeros(nb, dtype=ITYPE)
    cdef dict new_values = {}
    cdef list unique_values = []
    cdef PyObject *obj
    cdef object val

    for i in range(nb):
        val = values[i]
        obj = PyDict_GetItem(new_values, val)
        if obj is not NULL:
            ind = <object>obj
            indices[i] = ind
        else:
            new_values[val] = count
            indices[i] = count
            unique_values.append(val)
            count += 1
    return unique_values, indices


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef bool is_strictly_increasing(ITER x) except? 0:
    """
    >>> is_strictly_increasing([4, 4, 5])
    False
    >>> is_strictly_increasing([4, 3, 5])
    False
    >>> is_strictly_increasing([4, 5, 9])
    True
    >>> is_strictly_increasing(['a', 'b', 'c'])
    True
    >>> is_strictly_increasing(np.array(['2017-03-31', '2017-03-31', '2017-04-05'], dtype='datetime64[D]'))
    False
    """
    if isinstance(x, np.ndarray) and x.dtype.kind in ['S', 'U', 'M']:
        return x.size == 0 or np.all(x[1:] > x[:x.size - 1])

    cdef ITYPE_t i, nb = len(x)
    for i in range(nb - 1):
        if x[i] >= x[i+1]:
            return 0
    return 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef bool is_increasing(ITER x) except? 0:
    """
    >>> is_increasing([4, 4, 5])
    True
    >>> is_increasing([4, 3, 5])
    False
    >>> is_increasing([4, 5, 9])
    True
    >>> is_increasing(['a', 'a', 'c'])
    True
    >>> is_increasing(np.array(['2017-03-31', '2017-03-31', '2017-04-05'], dtype='datetime64[D]'))
    True
    """
    if isinstance(x, np.ndarray) and x.dtype.kind in ['S', 'U', 'M']:
        return x.size == 0 or np.all(x[1:] >= x[:x.size - 1])

    cdef ITYPE_t i, nb = len(x)
    for i in range(nb - 1):
        if x[i] > x[i+1]:
            return 0
    return 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dict unique_index(ITER ll):
    cdef dict result = {}
    cdef ITYPE_t i, nb = len(ll)
    cdef object val

    for i in range(nb):
        val = ll[i]
        if PyDict_Contains(result, val) == 1:
            raise ValueError('List should have unique values')
        else:
            result[val] = i
    return result


cpdef IndexedList new_uniquelist(list my_list, index=None):
    """
    For pickling protocol 2
    """
    return IndexedList(my_list, index)


cdef class IndexedList:

    def __cinit__(self, list my_list, index=None):
        check_values(my_list)
        if index is None:
            self._index = unique_index(my_list)
        else:
            self._index = index
        self.list = my_list

    def __len__(self):
        return self.list.__len__()

    def __iter__(self):
        return iter(self.list)

    def __repr__(self):
        """
        >>> a = ['a', 'b', 'd', 'c']
        >>> aa = IndexedList(a)
        >>> aa._index == {'a': 0, 'c': 3, 'b': 1, 'd': 2}
        True
        >>> aa.list
        ['a', 'b', 'd', 'c']
        >>> aa
        ['a', 'b', 'd', 'c']
        >>> len(aa)
        4
        >>> 'e' in aa
        False
        >>> 'd' in aa
        True
        >>> [x for x in aa]
        ['a', 'b', 'd', 'c']
        >>> aa[2]
        'd'
        >>> aa.index('c')
        3
        >>> IndexedList(['R', 'R']) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: List should have unique values
        """
        if len(self.list) > 100:
            return self.list[:20].__repr__()[:-1] + ", ... ," + self.list[-20:].__repr__()[1:]
        else:
            return self.list.__repr__()

    def __getitem__(self, ind):
        return self.list.__getitem__(ind)

    def __contains__(self, value):
        return value in self._index

    def __richcmp__(self, other, int op):
        if op == 2:  # __eq__
            if isinstance(self, IndexedList) and isinstance(other, IndexedList):
                return self.list.__eq__(other.list)
            elif isinstance(self, IndexedList):
                return self.list.__eq__(other)
            elif isinstance(other, IndexedList):
                return other.list.__eq__(self)
            else:
                raise NotImplementedError()
        elif op == 3:  # !=
            return not (self == other)
        else:
            raise NotImplementedError()

    def __reduce__(self):
        return new_uniquelist, (self.list, self._index)

    cpdef ITYPE_t index(self, value) except -1:
        return self._index.__getitem__(value)

    cpdef bool is_sorted(self):
        return is_strictly_increasing(self.list)

    cpdef tuple sorted(self):
        """
        >>> aa = IndexedList(['a', 'b', 'd', 'c'])
        >>> aa.is_sorted()
        False
        >>> bb, arg = aa.sorted()
        >>> bb
        ['a', 'b', 'c', 'd']
        >>> arg
        [0, 1, 3, 2]
        >>> bb.is_sorted()
        True
        """
        argsort = sorted(np.arange(len(self), dtype=ITYPE), key=self.__getitem__)
        return self.select(argsort), argsort

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef IndexedList select(self, indices):
        """
        >>> a = ['a', 'b', 'd', 'c']
        >>> aa = IndexedList(a)
        >>> aa.select([0, 3])
        ['a', 'c']
        >>> aa.select([0, 5]) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        IndexError: Indices out of bound
        >>> aa.select([-3, 5]) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        IndexError: Indices out of bound
        >>> aa.select([0, 1, 1]) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Indices should be unique
        """
        cdef np.ndarray[ndim=1, dtype=ITYPE_t] ind = np.asarray(indices, dtype=ITYPE)
        cdef ITYPE_t i, nb = ind.shape[0]
        cdef list my_list = self.list
        cdef list result = []
        cdef dict index = {}
        cdef object val
        # check increasing and bounds
        if nb == 0:
            return IndexedList([], {})

        if len(parallel_unique(ind)) < len(ind):
            raise ValueError('Indices should be unique')
        if np.min(ind) < 0 or np.max(ind) >= len(self):
            raise IndexError('Indices out of bound')
        # feeding new variables
        for i in range(nb):
            val = my_list[ind[i]]
            result.append(val)
            index[val] = i
        return IndexedList(result, index)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef tuple union(self, other, bool short_form=True):
        """
        Returns a tuple containing:
            * the resulting IndexedList
            * an array containing the indexes of result's elements inside
              the given IndexedList
            * on array containing the indexes of result's elements inside other
        If the element was not contained in the initial list, its index is len(list)

        :param short_form: instead of returning an array of result's elements
                           indexes for the shortest input list (or the first one
                           if they have the same length), the method returns
                           the length added to this list to get the result
        This avoid copy of list when they are useless.

        >>> a = ['a', 'b', 'd', 'c']
        >>> b = ['b', 'c', 'd', 'e', 'f']
        >>> aa = IndexedList(a)
        >>> bb = IndexedList(b)
        >>> aa.union(bb, False)
        (['a', 'b', 'd', 'c', 'e', 'f'], array([0, 1, 2, 3, 4, 4], dtype=int32), array([5, 0, 2, 1, 3, 4], dtype=int32))
        >>> bb.union(aa, False)
        (['a', 'b', 'd', 'c', 'e', 'f'], array([5, 0, 2, 1, 3, 4], dtype=int32), array([0, 1, 2, 3, 4, 4], dtype=int32))
        >>> aa.union(bb, True)
        (['a', 'b', 'd', 'c', 'e', 'f'], 2, array([5, 0, 2, 1, 3, 4], dtype=int32))
        >>> bb.union(aa, True)
        (['a', 'b', 'd', 'c', 'e', 'f'], array([5, 0, 2, 1, 3, 4], dtype=int32), 2)
        >>> u, arga, argb = aa.union(bb, False)
        >>> [a[i] if i < len(aa) else None for i in arga]
        ['a', 'b', 'd', 'c', None, None]
        >>> [b[i] if i < len(bb) else None for i in argb]
        [None, 'b', 'd', 'c', 'e', 'f']
        >>> aa.union(aa)[0] == aa
        True
        >>> type(aa.union(aa)[0]) == type(aa.union(bb)[0])
        True
        >>> aa.union(aa, short_form=False)
        (['a', 'b', 'd', 'c'], array([0, 1, 2, 3], dtype=int32), array([0, 1, 2, 3], dtype=int32))
        >>> aa.union(aa, short_form=True)
        (['a', 'b', 'd', 'c'], 0, array([0, 1, 2, 3], dtype=int32))
        """
        if self == other:
            if short_form:
                return self, 0, np.arange(len(self.list), dtype=ITYPE)
            return self, np.arange(len(self.list), dtype=ITYPE), np.arange(len(self.list), dtype=ITYPE)
        if not isinstance(other, IndexedList):
            other = IndexedList(other)

        if len(self) > len(other):
            common, arg1, arg2 = other.union(self, short_form)
            return common, arg2, arg1

        cdef list ll = other.list
        cdef ITYPE_t i = 0, k, n1 = len(self), n2 = len(ll), it = n1
        cdef dict self_index = self._index
        cdef list result_list = list(self.list)
        cdef dict result_index = self._index.copy()
        cdef PyObject *obj
        cdef np.ndarray[ndim=1, dtype=ITYPE_t] arg = np.hstack([np.full((n1,), n2, dtype=ITYPE),
                                                                np.zeros(n2, dtype=ITYPE)])
        cdef object val

        for i in range(n2):
            val = ll[i]
            obj = PyDict_GetItem(self_index, val)
            if obj is not NULL:
                k = <object>obj
                arg[k] = i
            else:
                result_list.append(val)
                arg[it] = i
                result_index[val] = it
                it += 1
        if short_form:
            return (IndexedList(result_list, result_index), it - n1, arg[:it])
        else:
            return (IndexedList(result_list, result_index),
                    np.hstack([np.arange(n1, dtype=ITYPE),
                               np.full((it - n1,), n1, dtype=ITYPE)]),
                    arg[:it])


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef tuple intersection(self, other):
        """
        Returns a tuple containing:
            * the resulting IndexedList
            * an array containing the indexes of result's elements inside
              the given IndexedList
            * on array containing the indexes of result's elements inside other

        >>> a = ['a', 'b', 'd', 'c']
        >>> b = ['b', 'c', 'd', 'e', 'f']
        >>> aa = IndexedList(a)
        >>> bb = IndexedList(b)
        >>> aa.intersection(bb)
        (['b', 'd', 'c'], array([1, 2, 3], dtype=int32), array([0, 2, 1], dtype=int32))
        >>> bb.intersection(aa)
        (['b', 'd', 'c'], array([0, 2, 1], dtype=int32), array([1, 2, 3], dtype=int32))
        >>> u, arga, argb = aa.intersection(bb)
        >>> [a[i] for i in arga] == u
        True
        >>> [b[i] for i in argb] == u
        True
        >>> aa.intersection(aa)[0] == aa
        True
        >>> aa.intersection(aa)
        (['a', 'b', 'd', 'c'], array([0, 1, 2, 3], dtype=int32), array([0, 1, 2, 3], dtype=int32))
        >>> type(aa.intersection(aa)[0]) == type(aa.intersection(bb)[0])
        True
        """
        if self == other:
            return self, np.arange(len(self.list), dtype=ITYPE), np.arange(len(self.list), dtype=ITYPE)
        if not isinstance(other, IndexedList):
            other = IndexedList(other)

        if len(self) > len(other):
            common, arg1, arg2 = other.intersection(self)
            return common, arg2, arg1

        cdef ITYPE_t i, k, it = 0, size = min(len(self), len(other)), nb = len(self)
        cdef np.ndarray[ndim=1, dtype=ITYPE_t] original_position_self = np.zeros(size, dtype=ITYPE)
        cdef np.ndarray[ndim=1, dtype=ITYPE_t] original_position_other = np.zeros(size, dtype=ITYPE)
        cdef list self_list = self.list, inter = []
        cdef dict other_index = other._index, index = {}
        cdef PyObject *obj
        cdef object val

        for i in range(nb):
            val = self_list[i]
            obj = PyDict_GetItem(other_index, val)
            if obj is not NULL:
                inter.append(val)
                index[val] = it
                original_position_self[it] = i
                k = <object>obj
                original_position_other[it] = k
                it += 1
        return IndexedList(inter, index), original_position_self[:it], original_position_other[:it]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef tuple align(self, other):
        """
        >>> a = ['a', 'b', 'd', 'c']
        >>> b = ['b', 'c', 'd', 'e', 'f']
        >>> aa = IndexedList(a)
        >>> bb = IndexedList(b)
        >>> aa.align(bb)
        (['b', 'c', 'd', 'e', 'f'], array([1, 3, 2, 4, 4], dtype=int32), array([0, 1, 2, 3, 4], dtype=int32))
        >>> bb.align(aa)
        (['a', 'b', 'd', 'c'], array([5, 0, 2, 1], dtype=int32), array([0, 1, 2, 3], dtype=int32))
        """
        if not isinstance(other, IndexedList):
            other = IndexedList(other)

        cdef ITYPE_t i, k, nb2 = len(other), nb1 = len(self)
        cdef np.ndarray[ndim=1, dtype=ITYPE_t] original_position_self = np.full((nb2,), nb1, dtype=ITYPE)
        cdef PyObject *obj
        cdef object val
        cdef dict tmp_index
        cdef list tmp_list

        if nb2 <= nb1:
            tmp_index = self._index
            tmp_list = other.list
            for i in range(nb2):
                val = tmp_list[i]
                obj = PyDict_GetItem(tmp_index, val)
                if obj is not NULL:
                    k = <object>obj
                    original_position_self[i] = k
        else:
            tmp_index = other._index
            tmp_list = self.list
            for i in range(nb1):
                val = tmp_list[i]
                obj = PyDict_GetItem(tmp_index, val)
                if obj is not NULL:
                    k = <object>obj
                    original_position_self[k] = i
        return other, original_position_self, np.arange(nb2, dtype=ITYPE)

    cpdef tuple difference(self, other):
        """
        Returns a tuple containing:
            * the resulting IndexedList
            * an array containing the indexes of result's elements inside
              the given IndexedList

        >>> a = ['a', 'b', 'd', 'c']
        >>> b = ['b', 'c', 'd', 'e', 'f']
        >>> aa = IndexedList(a)
        >>> bb = IndexedList(b)
        >>> aa.difference(bb)
        (['a'], [0])
        >>> bb.difference(aa)
        (['e', 'f'], [3, 4])
        >>> bb.difference(bb)
        ([], [])
        >>> type(bb.difference(bb)[0]) == type(aa.difference(bb)[0])
        True
        >>> len(bb.difference(bb)) == len(aa.difference(bb))
        True
        """
        if self == other:
            return IndexedList([]), []
        if not isinstance(other, IndexedList):
            other = IndexedList(other)
        diff = IndexedList([x for x in self.list if x not in other._index])
        return diff, list(map(self._index.__getitem__, diff))

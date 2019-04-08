from __future__ import absolute_import
from __future__ import division   # see https://www.python.org/dev/peps/pep-0238/#abstract
from six.moves import map, range

from warnings import warn

import numpy as np
from itertools import islice
from scipy.special import expit

from cyperf.tools.sort_tools import cython_argsort
from cyperf.tools import parallel_sort_routine
from cyperf.tools.parallel_sort_routine import (inplace_string_parallel_sort,
                                                inplace_numerical_parallel_sort as _inplace_numerical_parallel_sort,
                                                inplace_numerical_parallel_sort_nan)
from cyperf.tools.getter import (take_indices_on_numpy, take_indices_on_iterable,
                                 apply_python_dict, cast_to_float_array)


def _direct_dtype_converter(arr):
    if not isinstance(arr, np.ndarray):
        return arr

    kind = arr.dtype.kind
    if kind in ['b', 'B']:
        return arr.view(np.uint8)
    elif kind in ['M', 'm']:
        return arr.view(np.int64)
    else:
        return arr


def _check_as_one_dim(arr):
    assert arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)
    return arr.ravel()


def inplace_numerical_parallel_sort(a, reverse=False):
    if a.dtype.kind == 'f' and np.any(np.isnan(a)):
        inplace_numerical_parallel_sort_nan(a, reverse=reverse)
    else:
        _inplace_numerical_parallel_sort(_direct_dtype_converter(a), reverse=reverse)


def cy_parallel_sort(a, reverse=False):
    """
    it does not support np.dtype == "S" or "U"
    """
    if isinstance(a, (list, tuple)):
        assert len(a) == 0 or isinstance(a[0], str)
        b = list(a)
        inplace_string_parallel_sort(b, reverse)
    elif isinstance(a, np.ndarray):
        a = _check_as_one_dim(a)
        kind = a.dtype.kind

        if kind in ['S', 'U']:
            raise TypeError('unsupported kind of numpy array {}'.format(kind))
        b = a.copy()
        if kind == 'O':  # we expect to have a string
            inplace_string_parallel_sort(b, reverse)
        else:
            inplace_numerical_parallel_sort(b, reverse)
    else:
        raise TypeError('unsupported type: {}'.format(type(a)))
    return b


def sort_fallback(a, reverse=False):
    if isinstance(a, np.ndarray):
        a = _check_as_one_dim(a)
        return np.sort(a)[::-1] if reverse else np.sort(a)
    else:
        return sorted(a, reverse=reverse)


def parallel_sort(a, reverse=False):
    """
    Parallel sort for numpy arrays based on "gnu_parallel" gcc library
    """
    try:
        return cy_parallel_sort(a, reverse=reverse)
    except (TypeError, AssertionError):
        warn('Unsupported type {} for ParallelSort, fallback on slow implem.'.format(type(a)))
        return sort_fallback(a, reverse=reverse)


def parallel_unique(a):
    """
    equivalent np.unique(a)
    """
    try:
        a_sorted = parallel_sort(np.asarray(a))
    except (TypeError, AssertionError):
        return np.unique(a)

    if len(a_sorted) < 2:
        return a_sorted
    else:
        uindex = np.concatenate([[True], a_sorted[1:] > a_sorted[:-1]])
        return a_sorted[uindex]


def argsort_fallback(arr, reverse=False):
    from cyperf.indexing import get_index_dtype
    out_dtype = get_index_dtype(len(arr))
    # warn if fallback
    if isinstance(arr, np.ndarray) and not reverse:
        arr = _check_as_one_dim(arr)
        result = np.argsort(arr, kind="mergesort").astype(out_dtype, copy=False)
    else:
        # FIXME : when reverse is True and arr is np.ndarray it will be slow but true formula
        # FIXME: test on Nan, results will not be correct on np.nan
        result = np.asarray(sorted(range(len(arr)), key=arr.__getitem__, reverse=reverse), dtype=out_dtype)
    return result


def cy_parallel_argsort(arr, reverse=False):
    output_dtype_str = 'long' if len(arr) > np.iinfo(np.int32).max else 'int'
    if isinstance(arr, (list, tuple)):
        name = 'object'
    else:
        arr = np.asarray(arr)
        arr = _direct_dtype_converter(_check_as_one_dim(arr))
        dtype = arr.dtype

        name = 'numpy_strings' if dtype.kind == 'S' else dtype.name
        if dtype.kind == 'f' and np.any(np.isnan(arr)):
            output_dtype_str += "_nan"

    return getattr(parallel_sort_routine,
                   'parallel_argsort_{}_{}'.format(name, output_dtype_str))(arr, reverse=reverse)


def parallel_argsort(arr, reverse=False):
    """
    Parallel MergeSort

    it support the most of numerical types and string containers (like np.array or list)
    it support np.string_ dtype but not np.unicode_ (but it could)
    """
    try:
        return cy_parallel_argsort(arr, reverse=reverse)
    except (AttributeError, TypeError):
        warn('Unsupported argument {} for ParallelArgSort, fallback on slow implem.'.format(type(arr)))
        return argsort_fallback(arr, reverse=reverse)


def argsort(arr, nb=-1, reverse=False):
    """
       Parallel and QuickSort cython fast version of argsort

       args:
            * arr - input array
            * nb - number of elements to sort (for partial sort); default=-1 meaning full argsort
            * reverse - False/True, to use decreasing/increasing order
    """
    if nb < 0:
        nb += len(arr) + 1

    if nb <= len(arr) // 5:  # arbitrary constant
        try:
            return cython_argsort(arr, nb, reverse)
        except TypeError:
            pass
    return parallel_argsort(arr, reverse=reverse)


def logit(x, shift=0., width=1.):
    """
    >>> import numpy as np
    >>> np.allclose(logit(1, 0, 1), logistic(1, 0, 1))
    True
    >>> np.allclose(logit(3, -3, 2), logistic(3, -3, 2))
    True
    """
    return expit((x - shift) / width)


def logit_inplace(x, shift=0., width=1.):
    """
    TODO we should use precomputed logit table!!!
    """
    if shift != 0:
        x -= shift
    x /= 2 * width
    np.tanh(x, out=x)
    x += 1
    x *= 0.5
    return x


def take_indices(iterable, indices, length=None):
    """
    >>> x = np.array([3, 4, 5.])
    >>> ind = np.array([1, 0, 1, 2])
    >>> bool_ind = np.array([True, False, True], dtype=np.bool)
    >>> take_indices(x, 1)
    4.0
    >>> take_indices(slice(0, 7, 2), 3, 15)
    6
    >>> take_indices(x, ind)
    array([ 4.,  3.,  4.,  5.])
    >>> take_indices(x.tolist(), ind)
    [4.0, 3.0, 4.0, 5.0]
    >>> take_indices(tuple(x), ind)
    [4.0, 3.0, 4.0, 5.0]
    >>> take_indices(tuple(x), tuple(ind))
    [4.0, 3.0, 4.0, 5.0]
    >>> take_indices(list(x), tuple(ind))
    [4.0, 3.0, 4.0, 5.0]
    >>> take_indices(list(x), list(ind))
    [4.0, 3.0, 4.0, 5.0]
    >>> take_indices(list(x), bool_ind)
    [3.0, 5.0]
    >>> take_indices(x, bool_ind)
    array([ 3.,  5.])
    >>> take_indices(slice(0, 3), bool_ind, 3)
    array([0, 2])
    >>> take_indices('reree', [0,3])
    ['r', 'e']
    >>> take_indices(range(5), slice(0, 3))
    [0, 1, 2]
    >>> take_indices(np.arange(5), slice(0, 3))
    array([0, 1, 2])
    >>> take_indices({0: 5}, [0])
    [5]
    >>> take_indices({0: 5}, None)
    {0: 5}
    >>> y = np.arange(10).reshape((5, 2))
    >>> take_indices(y, [2, 0])
    array([[4, 5],
           [0, 1]])
    >>> take_indices(y, np.arange(2, 4))
    array([[4, 5],
           [6, 7]])

    >>> ind = np.array([1,8])
    >>> take_indices(x, ind) #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    IndexError: index 8 is out of bounds for size 3

    >>> take_indices(set('reree'), [0,3])
    Traceback (most recent call last):
    ...
    AttributeError: 'set' object has no attribute '__getitem__'
    >>> take_indices(x, ind.tolist()) #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    IndexError: Out of bounds on buffer access (axis 0)
    >>> take_indices(x.tolist(), ind.tolist()) #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    IndexError: list index out of range
    >>> take_indices(x.tolist(), 'reree')
    Traceback (most recent call last):
    ...
    TypeError: list indices must be integers, not str
    >>> take_indices({0: 5}, [5])
    Traceback (most recent call last):
    ...
    KeyError: 5
    """
    from cyperf.matrix.karma_sparse import is_karmasparse

    def fall_back_take_indices(iterable, indices):
        """
            Pure python analogue
        """
        return list(map(iterable.__getitem__, indices))

    if indices is None:
        return iterable  # without copy
    elif np.isscalar(indices):
        if isinstance(iterable, slice):
            return take_on_slice(iterable, [indices], length)[0]
        else:
            return iterable[indices]
    elif isinstance(indices, slice):
        if isinstance(iterable, slice):
            return compose_slices(iterable, indices, length)
        try:
            return iterable[indices]
        except Exception:
            return islice(iterable, indices.start, indices.stop, indices.step)
    elif isinstance(iterable, slice):
        return take_on_slice(iterable, indices, length)
    elif isinstance(iterable, np.ndarray) and isinstance(indices, np.ndarray):
        if np.issubdtype(indices.dtype, np.bool_):
            return iterable[indices]
        else:
            return iterable.take(indices, axis=0)
    elif isinstance(iterable, np.ndarray):
        try:
            return take_indices_on_numpy(iterable, indices)
        except IndexError as e:
            raise e
        except Exception:
            return iterable.take(indices, axis=0)
    elif is_karmasparse(iterable):
        return iterable[indices]
    else:
        # boolean slicing
        if isinstance(indices, np.ndarray) and np.issubdtype(indices.dtype, np.bool_):
            assert len(indices) == len(iterable)
            if isinstance(iterable, np.ndarray):
                return iterable[indices]  # shorter way for numpy.array
            indices = np.nonzero(indices)[0]
        try:
            return take_indices_on_iterable(iterable, indices)
        except IndexError as e:
            raise e
        except Exception:
            return fall_back_take_indices(iterable, indices)


def take_on_slice(slice_, indices, length):
    inner_length = slice_length(slice_, length)
    if inner_length == 0:
        return np.array([], dtype=np.int32)
    if isinstance(indices, np.ndarray) and np.issubdtype(indices.dtype, np.bool_):
        assert len(indices) == inner_length
        indices = np.nonzero(indices)[0]
    else:
        indices = np.array(indices, copy=True, dtype=np.int32 if length < 2**32 else None)
    slice_ = slice(*slice_.indices(length))
    indices[indices < 0] += inner_length
    indices *= slice_.step
    indices += slice_.start
    return indices


def is_trivial_slice(slice_, length):
    """
    Checks whether x[slice_] is the same as x for length=len(x)
    """
    return isinstance(slice_, slice) and \
        (slice_.start == 0 or slice_.start is None)\
        and (slice_.stop >= length or slice_.stop is None)\
        and (slice_.step == 1 or slice_.step is None)


def slice_length(slice_, length):
    """
    Return len(x[slice_]) for length=len(x)
    """
    start, stop, step = slice_.indices(length)
    if (stop - start) * step <= 0:
        return 0  # empty slice
    elif start == 0 and stop == length and step == 1:
        return length  # whole iterable
    else:
        return (abs(stop - start) - 1) // abs(step) + 1


def compose_slices(slice1, slice2, length):
    """
    returns a slice that is a combination of the two slices.
    As in
      x[slice1][slice2]
    becomes
      combined_slice = compose_slices(slice1, slice2, len(x))
      x[combined_slice]

    Taken from http://stackoverflow.com/questions/19257498/combining-two-slicing-operations

    :param slice1: The first slice
    :param slice2: The second slice
    :param length: The length of the first dimension of data being sliced. (eg len(x))

    >>> x = np.arange(100)
    >>> slice1 = slice(3, 20, 2)
    >>> slice2 = slice(2, 8, None)
    >>> sc = compose_slices(slice1, slice2, len(x))
    >>> np.all(x[slice1][slice2] == x[sc])
    True
    """
    slice1_step = 1 if slice1.step is None else slice1.step
    slice2_step = 1 if slice2.step is None else slice2.step
    step = slice1_step * slice2_step
    slice1_length = slice_length(slice1, length)
    if slice1_length == 0 or slice_length(slice2, slice1_length) == 0:
        return slice(0, 0, step)

    slice1 = slice(*slice1.indices(length))
    slice2 = slice(*slice2.indices(slice1_length))

    start = slice1.start + slice2.start * slice1_step
    stop = slice1.start + slice2.stop * slice1_step
    if start > stop and stop < 0:
        stop = None
    return slice(start, stop, step)

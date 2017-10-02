
import numpy as np
from itertools import islice
from scipy.special import expit

from cyperf.tools.sort_tools import cython_argsort
from cyperf.tools.getter import (take_indices_on_numpy, take_indices_on_iterable,
                                 apply_python_dict, cast_to_float_array, python_feature_hasher)


def argsort(xx, nb=-1, reverse=False):
    """
       QuickSort cython fast version of argsort

       args:
            * xx - input array
            * nb - number of elements to sort (for partial sort); default=-1 meaning full argsort
            * reverse - False/True, to use decreasing/increasing order
    """
    try:
        return cython_argsort(xx, nb, reverse)
    except TypeError:  # fallback if non-accepted dtype (like string)
        x = np.argsort(xx, kind='quicksort')
        return x[::-1] if reverse else x


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
    >>> bool_ind = np.array([True, False, True], dtype='bool')
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
        return map(iterable.__getitem__, indices)
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
        except:
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
        except:
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
        except:
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

from numbers import Integral

import numpy as np
from toolz import merge as dict_merge


def aeq(matrix1, matrix2):
    """
    shortcut for np.allclose(matrix1, matrix2, rtol=1e-7)
    """
    return np.allclose(matrix1, matrix2, rtol=1e-7)


def co_axis(axis):
    """
    Given an axis, return the other one.

    :param axis: an axis, in range [0, 1]
    :return: the co-axis in range [0, 1]

    Exemples: ::

        >>> co_axis(0)
        1
        >>> co_axis(1)
        0
        >>> co_axis(3)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        RuntimeError: axis 3 is out of range [0,1]

    """
    # TODO: check usage of this
    if axis == 0:
        return 1
    elif axis == 1:
        return 0
    else:
        raise RuntimeError(f"axis {axis} is out of range [0,1]")


def zipmerge(dictionary1, dictionary2):
    """
    >>> aa = {'a': 3, 'c': 4}
    >>> bb = {'a': 5, 'b': 4}
    >>> zipmerge((aa, bb), (bb, aa))
    ({'a': 5, 'c': 4, 'b': 4}, {'a': 3, 'c': 4, 'b': 4})
    """
    return tuple(dict_merge(x, y) for x, y in zip(dictionary1, dictionary2))


def is_integer(arg):
    return isinstance(arg, Integral)


def lm_compute_volume_at_cutoff(lm, potential_cutoff):
    """
    Compute the number of lines required to reach a share of the total potential

    :param lm: labeled_matrix. scores_as_labeled_matrix
    :param potential_cutoff: float.
    :return: dict. Keys are topics (columns in the input lm), values are the volume of users (rows in the input lm)
                   required to reach a given share of the total potential (sum of the lm values for each column).
    """
    scores_array = lm.to_dense().matrix
    # Get total potential
    total_potential = scores_array.sum(axis=0)

    # Sort scores_array per topic
    scores_ordered_array = np.sort(scores_array, axis=0)[::-1]

    # Compute array of users part of the potential cutoff
    in_potential_cutoff_array = (scores_ordered_array.cumsum(axis=0) / total_potential) < potential_cutoff
    # Compute users-in-potential-cutoff volume share
    vol_at_cutoff = in_potential_cutoff_array.sum(axis=0).astype('float32') / scores_array.shape[0]

    return dict(zip(lm.column, vol_at_cutoff))

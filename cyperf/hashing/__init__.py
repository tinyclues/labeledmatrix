import numpy as np
from cyperf.tools import parallel_unique

from .hash_tools import hash_numpy_string, hash_generic_string, hasher_numpy_string
from .hash_tools import hash_numpy_string_with_many_seeds as _hash_numpy_string_with_many_seeds
from .hash_tools import randomizer_numpy_string as _randomizer_numpy_string
from .hash_tools import randomizer_python_string as _randomizer_python_string
from .hash_tools import increment_over_numpy_string as _increment_over_numpy_string


def hash_string(arr, seed):
    if isinstance(arr, np.ndarray) and arr.dtype.kind == 'S':
        return hash_numpy_string(arr, seed)
    else:
        return hash_generic_string(arr, seed)


def hash_numpy_string_with_many_seeds(keys, seeds):
    return _hash_numpy_string_with_many_seeds(np.asarray(keys, dtype='S'),
                                              np.asarray(seeds, dtype='uint32'))


def randomizer_string(keys, composition, seed):
    composition = np.asarray(composition, dtype='uint32')
    if isinstance(keys, list):
        return _randomizer_python_string(keys, composition, seed)
    return _randomizer_numpy_string(np.asarray(keys, dtype='S'), composition, seed)


def increment_over_numpy_string(keys, segments, values, seeds, composition, nb_segments=None):
    segments = np.asarray(segments, dtype=np.int64)
    if nb_segments is None:
        nb_segments = parallel_unique(segments).shape[0]
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False

    increment = _increment_over_numpy_string(np.ascontiguousarray(keys, dtype='S'),
                                             segments,
                                             np.ascontiguousarray(values),
                                             np.ascontiguousarray(seeds, dtype='uint32'),
                                             np.ascontiguousarray(composition, dtype='uint32'),
                                             nb_segments)

    return increment[:, :, 0] if squeeze else increment

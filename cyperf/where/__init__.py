import numpy as np
from cytoolz import curry

from cyperf.indexing import get_index_dtype
import indices_where_long
import indices_where_int
import numpy_fallback


def _get_module(size):
    if size > np.iinfo(np.int32).max:
        return indices_where_long
    else:
        return indices_where_int


@curry
def _fast_filter_proto(name, column, *args):
    full_name = 'indices_where'
    if name:
        full_name += '_' + name

    size = len(column)
    dt = get_index_dtype(size)
    if hasattr(numpy_fallback, full_name) and isinstance(column, np.ndarray) and \
       ('same' not in full_name or isinstance(args[0], np.ndarray)):
        return getattr(numpy_fallback, full_name)(column, *args).astype(dt, copy=False)
    else:
        # list or python filter like where_in that does not exist in numpy
        return getattr(_get_module(size), full_name)(column, *args)


def fast_filter(name, *args):
    function = _fast_filter_proto(name)

    def filter_decorator(column):
        return function(column, *args)

    return filter_decorator


# numpy supported
indices_where = _fast_filter_proto('')
indices_where_not = _fast_filter_proto('not')
indices_where_eq = _fast_filter_proto('eq')
indices_where_ne = _fast_filter_proto('ne')
indices_where_lt = _fast_filter_proto('lt')
indices_where_le = _fast_filter_proto('le')
indices_where_gt = _fast_filter_proto('gt')
indices_where_ge = _fast_filter_proto('ge')
indices_where_between = _fast_filter_proto('between')
indices_where_same = _fast_filter_proto('same')
indices_where_not_same = _fast_filter_proto('not_same')

# # general
indices_where_in = _fast_filter_proto('in')
indices_where_not_in = _fast_filter_proto('not_in')
indices_where_between = _fast_filter_proto('between')
indices_where_contains = _fast_filter_proto('contains')
indices_where_not_contains = _fast_filter_proto('not_contains')
indices_where_operator = _fast_filter_proto('operator')
indices_where_callable = _fast_filter_proto('callable')

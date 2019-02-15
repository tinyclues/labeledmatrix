from __future__ import absolute_import
from functools import partial
import numpy as np

from cyperf.indexing import get_index_dtype
from cyperf.where import indices_where_long, indices_where_int, numpy_fallback


def _switch_module(size):
    if size > np.iinfo(np.int32).max:
        return indices_where_long
    else:
        return indices_where_int


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
        return getattr(_switch_module(size), full_name)(column, *args)


def fast_filter(name, *args):
    function = partial(_fast_filter_proto, name)

    def filter_decorator(column):
        return function(column, *args)

    return filter_decorator


# numpy supported
indices_where = partial(_fast_filter_proto, '')
indices_where_not = partial(_fast_filter_proto, 'not')
indices_where_eq = partial(_fast_filter_proto, 'eq')
indices_where_ne = partial(_fast_filter_proto, 'ne')
indices_where_lt = partial(_fast_filter_proto, 'lt')
indices_where_le = partial(_fast_filter_proto, 'le')
indices_where_gt = partial(_fast_filter_proto, 'gt')
indices_where_ge = partial(_fast_filter_proto, 'ge')
indices_where_between = partial(_fast_filter_proto, 'between')
indices_where_same = partial(_fast_filter_proto, 'same')
indices_where_not_same = partial(_fast_filter_proto, 'not_same')

# # general
indices_where_in = partial(_fast_filter_proto, 'in')
indices_where_not_in = partial(_fast_filter_proto, 'not_in')
indices_where_contains = partial(_fast_filter_proto, 'contains')
indices_where_not_contains = partial(_fast_filter_proto, 'not_contains')
indices_where_operator = partial(_fast_filter_proto, 'operator')
indices_where_callable = partial(_fast_filter_proto, 'callable')

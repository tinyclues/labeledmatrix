from __future__ import absolute_import

import numpy as np
from functools import partial

from cyperf.indexing import get_index_dtype
from cyperf.tools.getter import check_values
from cyperf.where import cy_filter, numpy_fallback
from cyperf.tools.vector import int32Vector, int64Vector


__all__ = ['fast_filter', 'indices_where_same', 'indices_where_not_same']


def fast_filter(name, *args):
    '''
    wrapper expose in karma-core Column.find_indices
    '''
    function = partial(fast_filter_proto, name)

    def filter_decorator(column):
        return function(column, *args)

    return filter_decorator


def _get_container(size):
    if size > np.iinfo(np.int32).max:
        return int64Vector()
    else:
        return int32Vector()


def fast_filter_proto(name, column, *args):
    full_name = 'indices_where'
    if name:
        full_name += '_' + name

    size = len(column)

    if hasattr(numpy_fallback, full_name) and isinstance(column, np.ndarray) and \
       ('same' not in full_name or isinstance(args[0], np.ndarray)):
        return getattr(numpy_fallback, full_name)(column, *args).astype(get_index_dtype(size), copy=False)
    else:
        # list or python filter like where_in that does not exist in numpy
        check_values(column)
        return getattr(cy_filter, full_name)(_get_container(size), column, *args)


# numpy supported
indices_where = partial(fast_filter_proto, '')
indices_where_not = partial(fast_filter_proto, 'not')
indices_where_eq = partial(fast_filter_proto, 'eq')
indices_where_ne = partial(fast_filter_proto, 'ne')
indices_where_lt = partial(fast_filter_proto, 'lt')
indices_where_le = partial(fast_filter_proto, 'le')
indices_where_gt = partial(fast_filter_proto, 'gt')
indices_where_ge = partial(fast_filter_proto, 'ge')
indices_where_between = partial(fast_filter_proto, 'between')
indices_where_same = partial(fast_filter_proto, 'same')
indices_where_not_same = partial(fast_filter_proto, 'not_same')

# general
indices_where_in = partial(fast_filter_proto, 'in')
indices_where_not_in = partial(fast_filter_proto, 'not_in')
indices_where_contains = partial(fast_filter_proto, 'contains')
indices_where_not_contains = partial(fast_filter_proto, 'not_contains')
indices_where_operator = partial(fast_filter_proto, 'operator')
indices_where_callable = partial(fast_filter_proto, 'callable')

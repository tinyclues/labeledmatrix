import indices_where_long
import indices_where_int
import numpy as np


def get_module(size):
    if size > np.iinfo(np.int32).max:
        return indices_where_long
    else:
        return indices_where_int

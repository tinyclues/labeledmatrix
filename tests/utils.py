#
# Copyright tinyclues, All rights reserved
#
import random
import math

import pandas as pd

from labeledmatrix.core.random import UseSeed


def gen_values(length):
    """
    Returns a list of *length* values in an almost random fashion.

    Examples: ::

        >>> length = random.randint(100, 1000)
        >>> values = gen_values(length)
        >>> len(values) == length
        True
        >>> from collections import Counter
        >>> counts = Counter(values)
        >>> list(range(len(counts))) == sorted(counts.keys())
        True

    """
    coefs = []
    bound = int(math.sqrt(length))
    while sum(coefs) < length - bound:
        coefs.append(random.randint(1, bound))
    coefs.append(length - sum(coefs))
    values = []
    for i in ([x] * coef for x, coef in zip(range(len(coefs)), coefs)):
        values.extend(i)
    random.shuffle(values)
    return values


@UseSeed()
def basic_dataframe(length, column_names=None):
    """
    Returns a dataframe with *length* rows and column names defined by *column_names*.
    All columns share the same values so they can be used for relations.

    Parameters:
        - length: int
          number of rows in the dataframe
        - column_names: iterable of strings
          names of the columns in the dataframe (default: ['a', 'b', 'c'])

    Examples: ::

        >>> length = random.randint(100, 1000)
        >>> data = basic_dataframe(length)
        >>> list(data.columns)
        ['a', 'b', 'c']
        >>> len(data) == length
        True
        >>> column_names = ['x', 'y', 'z']
        >>> data = basic_dataframe(length, column_names)
        >>> list(data.columns) == column_names
        True
        >>> len(data) == length
        True

    """
    if not column_names:
        column_names = ['a', 'b', 'c']
    data = pd.DataFrame()
    for name in column_names:
        data[name] = gen_values(length)
    return data

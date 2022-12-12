import string
from typing import Union, List, Any

import numpy as np
import scipy.sparse as sp

from cyperf.indexing.indexed_list import reversed_index
from cyperf.matrix.karma_sparse import KarmaSparse, ks_diag

from labeledmatrix.learning.matrix_utils import keep_sparse
from labeledmatrix.learning.utils import use_seed


# TODO challenge reversed_index through this file vs pd.factorize


def from_zip_occurrence(input1: Union[List[Any], np.ndarray],
                        input2: Union[List[Any], np.ndarray]):
    """
    TODO doc
    TODO move to unit tests
    >>> from karma.synthetic.basic import basic_dataframe  # FIXME
    >>> from labeledmatrix.core.labeledmatrix import LabeledMatrix
    >>> df = basic_dataframe(100)
    >>> lm = LabeledMatrix(*simple_counter((df['a'].values, df['b'].values)))
    >>> lm.matrix.sum() == 100
    True
    >>> lm.row == df.drop_duplicates('a')['a'].values
    True
    >>> lm.column == df.drop_duplicates('b')['b'].values
    True
    >>> df1 = df.shuffle()
    >>> lm1 = LabeledMatrix(*simple_counter((df1['a'].values, df1['b'].values)))
    >>> lm1.sort() == lm.sort()
    True
    >>> for i in range(len(df)):
    ...     a, b = df['a'][i], df['b'][i]
    ...     assert lm1[a, b] == df.counts(('a', 'b'))[(a,b)]
    """
    if len(input1) != len(input2):
        raise ValueError(f'Inputs must have same length, got {len(input1)} and {len(input2)} instead')
    unique_values1, indices1 = reversed_index(input1)
    unique_values2, indices2 = reversed_index(input2)
    matrix = KarmaSparse((indices1, indices2), shape=(len(unique_values1), len(unique_values2)), format="csr")
    if not keep_sparse(matrix):
        matrix = matrix.toarray()
    return (unique_values1, unique_values2), matrix


def from_random(shape=(4, 3), density=0.5, seed=None, square=False):
    """
    :param shape: the shape of the `LabeledMatrix` (rows, columns)
    :param density: ratio of non-zero elements
    :param seed: seed that could be used to seed the random generator
    :param square: same labels for rows and columns
    """
    alphabet = list(string.ascii_lowercase)

    def _random_word(length):
        while True:
            yield ''.join(np.random.choice(alphabet, size=length))

    def _generate_words(nb, length):
        random_word = _random_word(length)
        words = []
        words_set = set()
        while len(words) < nb:
            word = next(random_word)
            if word not in words_set:
                words_set.add(word)
                words.append(word)
        return words

    with use_seed(seed):
        row = _generate_words(shape[0], 5)
        if square:
            column = row
        else:
            column = _generate_words(shape[1], 5)
        matrix = KarmaSparse(sp.rand(len(row), len(column), density, format="csr", random_state=seed))

    return (row, column), matrix


def from_diagonal(keys, values, keys_deco=None, sparse=True):
    matrix = ks_diag(np.asarray(values))
    if not sparse:
        matrix = matrix.toarray()

    return (keys, keys), matrix, (keys_deco, keys_deco)


def from_pivot(dataframe, index=None, columns=None, values=None, aggregator="sum", sparse=True):
    pass

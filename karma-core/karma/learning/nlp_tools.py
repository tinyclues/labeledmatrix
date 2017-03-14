#
# Copyright tinyclues, All rights reserved
#

import string

DEFAULT_DEPTH = 3
DEFAULT_EXCLUDE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def ngrams(phrase, depth=DEFAULT_DEPTH):
    """
    returns a list of grams from the list of words
    >>> ngrams(range(6), depth=2)
    [(0,), (1,), (0, 1), (2,), (1, 2), (3,), (2, 3), (4,), (3, 4), (5,), (4, 5)]
    >>> len(ngrams(range(100), depth=2))
    199
    >>> ngrams(range(4), depth=6)
    [(0,), (1,), (0, 1), (2,), (1, 2), (0, 1, 2), (3,), (2, 3), (1, 2, 3), (0, 1, 2, 3)]
    """
    res = []
    for i in xrange(len(phrase)):
        for j in xrange(depth):
            if i >= j:
                res.append(tuple(phrase[i - j: i + 1]))
    return res


def prepare(text, exclude=DEFAULT_EXCLUDE):
    """
    in string text.lower(), replace all symbols of "exclude" list by a space,
    then split it by a space and drop out the empty strings in obtained array.
    >>> prepare('I I   fdf , ! !!! eE     re')
    ['i', 'i', 'fdf', 'ee', 're']
    """
    clean = text.lower().translate(string.maketrans(exclude, " " * len(exclude)))
    return [x for x in clean.split(' ') if x]


def raw_ngrams(text, depth=DEFAULT_DEPTH, exclude=DEFAULT_EXCLUDE):
    return ngrams(prepare(text, exclude=exclude), depth=depth)

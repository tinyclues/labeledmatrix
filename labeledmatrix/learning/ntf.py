#
# Copyright tinyclues, All rights reserved
#
import numpy as np

__all__ = ['ntf']


def ntf(tensor, rank=2, max_iter=150):
    """
    tensor should be non-negative numpy.array (dim = 3) with non-zero columns.
    if n,m,k = tensor.shape then nmf(tensor, rank) returns tree non-negative matrices of shapes (n,rank), (m,rank), (k,rank) respectively
    Here in factorization, we minimize "alpha divergence" with alpha=1 corresponding to Kullback-Leibler divergence

    Examples: ::

        >>> from labeledmatrix.learning.matrix_utils import kl_div
        >>> import numpy as np
        >>> tensor = np.random.rand(15, 10, 6)
        >>> a, b, c = ntf(tensor, rank=4, max_iter=400)
        >>> a.shape
        (15, 4)
        >>> b.shape
        (10, 4)
        >>> c.shape
        (6, 4)
        >>> tensor = scalar_tensor_dot(np.random.rand(6, 4), np.random.rand(6, 4), np.random.rand(6, 4))
        >>> a, b, c = ntf(tensor, rank=4, max_iter=300)
        >>> kl_div(tensor, scalar_tensor_dot(a, b, c)) < 1
        True
    """
    # print("NMF : Tensor dimensions are {}".format(tensor.shape))
    # print("NMF : Used rank is equal to {}".format(rank))
    result = NTF(tensor, rank)
    result.initial(rank)
    return result.iterate(max_iter)


def scalar_tensor_dot(a, b, c):
    """
    >>> a = np.random.rand(4,2)
    >>> b = np.random.rand(5,2)
    >>> c = np.random.rand(3,2)
    >>> scalar_tensor_dot(a, b, c).shape
    (4, 5, 3)
    """
    return np.einsum('ir,jr,kr->ijk', a, b, c)


def kl_single_update(a, tensor, b, c):
    """    realize one step of update for kullback divergence
    >>> y = np.random.rand(15, 10, 6)
    >>> a = np.random.rand(15, 4)
    >>> b = np.random.rand(10, 4)
    >>> c = np.random.rand(6, 4)
    >>> kl_single_update(a, y, b, c).shape == a.shape
    True
    >>> np.max(kl_single_update(a, scalar_tensor_dot(a, b, c), b, c) - a) < 10 ** (-7)
    True
    """
    return a * np.einsum('ij,j->ij',
                         np.einsum('qj,tj,itq->ij', c, b, tensor / scalar_tensor_dot(a, b, c)),
                         1.0 / (b.sum(axis=0) * c.sum(axis=0)))


class NTF(object):
    def __init__(self, tensor, rank, metric="KL"):
        self._tensor = tensor
        if np.min(self._tensor) < 0:
            raise ValueError("in Tensor, entries should be positives")
        if np.min(self._tensor.sum(axis=0).sum(axis=1)) == 0:
            raise ValueError("in Tensor, not all entries in a row can be zero")
        self.n, self.m, self.k = self._tensor.shape
        self._epsilon = 10 ** (-7)
        self._tensor = self._tensor
        self.metric = metric
        self.initial(rank)

    def initial(self, rank, a=None, b=None, c=None):
        self.rank = rank
        self.a = np.random.rand(self.n, self.rank) if a is None else a
        self.b = np.random.rand(self.m, self.rank) if b is None else b
        self.c = np.random.rand(self.k, self.rank) if c is None else b
        self._clip()

    def _clip(self):
        self.a.clip(min=self._epsilon, out=self.a)
        self.b.clip(min=self._epsilon, out=self.b)
        self.c.clip(min=self._epsilon, out=self.c)

    def kl_update(self):
        self.a = kl_single_update(self.a, self._tensor, self.b, self.c)
        self.b = kl_single_update(self.b, self._tensor.transpose(1, 0, 2), self.a, self.c)
        self.c = kl_single_update(self.c, self._tensor.transpose(2, 1, 0), self.b, self.a)
        self._clip()

    def iterate(self, maxiter):
        if self.metric == 'KL':
            for i in range(maxiter):
                self.kl_update()
        return self.a, self.b, self.c

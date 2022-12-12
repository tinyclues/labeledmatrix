from collections import Hashable
from functools import wraps, reduce

from numpy.random.mtrand import seed as np_seed
from numpy.random.mtrand import get_state as np_get_state
from numpy.random.mtrand import set_state as np_set_state
from random import seed as py_seed
from random import getstate as py_get_state
from random import setstate as py_set_state

from itertools import imap, product
from multiprocessing.pool import ThreadPool

import torch  # TODO keep or drop ?
import numpy as np
from cyperf.tools import parallel_unique

from cyperf.tools import take_indices

# FIXME VirtualHStack keep or drop ?
from karma.core.utils import is_iterable, quantile_boundaries, coerce_to_tuple_and_check_all_strings
from labeledmatrix.learning.matrix_utils import (safe_hstack, number_nonzero, cast_float32,
                                                 direct_product, direct_product_dot,
                                                 direct_product_dot_transpose, direct_product_second_moment,
                                                 cast_2dim_float32_transpose, safe_min, safe_max)
from labeledmatrix.learning.thread_setter import blas_level_threads

NB_THREADS_MAX = 16
NB_CV_GROUPS_MAX = 10 ** 5
KNOWN_LOGISTIC_METHODS = ['logistic_coefficients', 'logistic_coefficients_and_posteriori']


def safe_dot_torch(a, b):
    """
    we use torch.matmul instead of numpy.dot because torch uses thread-safe mkl
    meanwhile openblas used by numpy.dot is not:
    https://github.com/numpy/numpy/issues/11046
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    else:
        return a.dot(b)


class VirtualDirectProduct(object):
    """
    We may also want to support scalar case of x or y ?


        X should accept list like this [x, y, z, (x,y), (z, x)]
        tuples stay for direct_product

    """

    def __init__(self, left, right, is_transposed=False):
        assert left.shape[0] == right.shape[0]
        assert left.ndim == 2
        assert right.ndim == 2
        self.is_transposed = is_transposed
        self.left, self.right = cast_float32(left), cast_float32(right)

    @property
    def T(self):
        return VirtualDirectProduct(self.left, self.right, is_transposed=not self.is_transposed)

    @property
    def shape(self):
        shape = (self.left.shape[0], self.left.shape[1] * self.right.shape[1])
        return shape[::-1] if self.is_transposed else shape

    @property
    def dtype(self):
        return np.promote_types(self.left.dtype, self.right.dtype)

    @property
    def nnz(self):
        # it does not work at all case (by sparse x dense should be Ok)
        return number_nonzero(self.left) * number_nonzero(self.right) / max(self.right.shape[0], 1)

    def _extreme_values(self):
        min_left = safe_min(self.left, axis=1)
        max_left = safe_max(self.left, axis=1)
        min_right = safe_min(self.right, axis=1)
        max_right = safe_max(self.right, axis=1)

        return list(product([min_left, max_left], [min_right, max_right]))

    def min(self, axis=None):
        if axis is None:
            return min(map(lambda *args: (args[0] * args[1]).min(), self._extreme_values()))
        else:
            raise NotImplementedError('VirtualDirectProduct.min supports only axis=None')

    def max(self, axis=None):
        if axis is None:
            return max(map(lambda *args: (args[0] * args[1]).max(), self._extreme_values()))
        else:
            raise NotImplementedError('VirtualDirectProduct.max supports only axis=None')

    @property
    def ndim(self):
        return 2

    def __getitem__(self, indices):
        if self.is_transposed:
            raise NotImplementedError('VirtualDirectProduct.__getitem__ when is_transposed')
        else:
            return VirtualDirectProduct(self.left[indices], self.right[indices])

    def sum(self, axis=None):
        if axis is None:
            return self.sum(axis=1).sum()

        if axis == 1:
            return self.dot(np.ones(self.shape[1], dtype=self.dtype))
        elif axis == 0:
            return self.transpose_dot(np.ones(self.shape[0], dtype=self.dtype))
        else:
            raise NotImplementedError(axis)

    def dot(self, w, power=1):
        if self.is_transposed:
            return direct_product_dot_transpose(self.left, self.right, w, power)
        else:
            return direct_product_dot(self.left, self.right, w, power)

    def transpose_dot(self, w, power=1):
        if self.is_transposed:
            return direct_product_dot(self.left, self.right, w, power)
        else:
            return direct_product_dot_transpose(self.left, self.right, w, power)

    def second_moment(self):
        if self.is_transposed:
            raise NotImplementedError()
        else:
            return direct_product_second_moment(self.left, self.right)

    def materialize(self):
        res = direct_product(self.left, self.right)
        return res.T if self.is_transposed else res

    def __array__(self):
        # to simplify equality tests with numpy
        return np.asarray(self.materialize())


def safe_vdp_cast(x):
    return VirtualDirectProduct(*x) if isinstance(x, tuple) else x


def vdp_materilaze(x):
    return x.materialize() if isinstance(x, VirtualDirectProduct) else x


class BasicVirtualHStack(object):
    def __init__(self, X):
        if isinstance(X, BasicVirtualHStack):
            self.X = X.X
            self.is_block = X.is_block
            if self.is_block:
                self.dims = X.dims
            return

        self.is_block = isinstance(X, (list, tuple))
        if self.is_block:
            if len(X) == 0:
                raise ValueError('Cannot create a VirtualHStack of an empty list')

            X = map(safe_vdp_cast, X)

            integer_entries = []
            common_shape0 = None
            for i in range(len(X)):
                if not hasattr(X[i], 'shape') or X[i].shape is None or X[i].shape == ():
                    try:
                        X[i] = int(X[i])
                    except (ValueError, TypeError):
                        raise ValueError('Cannot create a VirtualHStack with {}'.format(type(X[i])))
                    integer_entries.append(i)
                else:
                    shape0 = X[i].shape[0]
                    if common_shape0 is None:
                        common_shape0 = shape0
                    elif shape0 != common_shape0:
                        raise ValueError('Cannot create a VirtualHStack for an array of length {} while other array '
                                         'has length {}'.format(shape0, common_shape0))
            for i in integer_entries:
                X[i] = np.zeros((common_shape0 or 1, X[i]))

            X = [xx if isinstance(xx, VirtualDirectProduct) else cast_2dim_float32_transpose(xx) for xx in X]
            self.dims = np.cumsum([0] + [x.shape[1] for x in X])
        else:
            X = X if isinstance(X, VirtualDirectProduct) else cast_float32(X)

        self.X = X

    def __getitem__(self, indices):
        if indices is None:
            return self
        if self.is_block:
            return BasicVirtualHStack([x[indices] for x in self.X])
        else:
            return BasicVirtualHStack(self.X[indices])

    @property
    def shape(self):
        if self.is_block:
            return self.X[0].shape[0], self.dims[-1]
        else:
            return self.X.shape

    def split_by_dims(self, w):
        assert w.shape[0] == self.shape[1]
        if self.is_block:
            return [w[self.dims[i]:self.dims[i + 1]] for i in range(len(self.X))]
        else:
            return w

    def adjust_array_to_total_dimension(self, arr, param_name=''):
        try:
            if not is_iterable(arr):
                return np.full(self.shape[1], arr)
            elif len(arr) == self.shape[1]:
                return np.asarray(arr)
            elif self.is_block and len(arr) == len(self.X):
                exp_arr = [np.full(x.shape[1], elt) if not is_iterable(elt) else elt for elt, x in zip(arr, self.X)]
                res = np.concatenate(exp_arr)
                assert (len(res) == self.shape[1])
                return res
            else:
                assert False
        except AssertionError:
            additional_message = ' or {}'.format(len(self.X)) if self.is_block else ''
            param_message = "parameter '{}'".format(param_name) if param_name != '' else 'parameter'
            raise ValueError('{} is invalid: expected float or an array-like of length {}{}, '
                             'got array-like of length {}'.format(param_message, self.shape[1], additional_message,
                                                                  len(arr)))

    def adjust_to_block_dimensions(self, arr, param_name=''):
        return self.split_by_dims(self.adjust_array_to_total_dimension(arr, param_name))

    def materialize(self):
        if self.is_block:
            if len(self.X) == 1:
                return vdp_materilaze(self.X[0])
            else:
                return safe_hstack(map(vdp_materilaze, self.X))
        else:
            return vdp_materilaze(self.X)


class VirtualHStack(BasicVirtualHStack):
    def __init__(self, X, nb_threads=1, nb_inner_threads=None):
        super(VirtualHStack, self).__init__(X)
        self.pool = None
        self.nb_threads = nb_threads
        self.nb_inner_threads = nb_inner_threads
        self._set_parallelism()

    def __repr__(self):
        repr_ = "<VirtualHStack :"
        repr_ += '\n * shape {}'.format(self.shape)
        if self.is_block:
            repr_ += "\n * containing {} blocks".format(len(self.X))
            repr_ += "\n * with dimensions {}".format(np.diff(self.dims))
        else:
            repr_ += "\n * containing single matrix"
        repr_ += "\n *  nb_threads={}, nb_inner_threads={}".format(self.nb_threads, self.nb_inner_threads)
        return unicode(repr_ + ">", encoding="utf-8", errors='replace').encode("utf-8")

    def __del__(self):
        self._close_pool()  # I hope that it works as I expect
        self.X = None

    def _set_parallelism(self):
        if self.is_block:
            self.nb_threads = min(self.nb_threads, len(self.X), NB_THREADS_MAX)
            if self.nb_threads > 1:
                self.pool = ThreadPool(self.nb_threads)

            self.order = np.argsort(map(self._block_priority, self.X))[::-1]
            self.reverse_order = np.argsort(self.order)
        else:
            self.nb_threads = 1

        if self.nb_inner_threads is None:
            if self.nb_threads > 0:
                self.nb_inner_threads = min(NB_THREADS_MAX, max(1, int(2 * 32. / self.nb_threads)))
            else:
                self.nb_inner_threads = NB_THREADS_MAX

    def __getitem__(self, indices):
        if indices is None:
            return self
        return VirtualHStack(super(VirtualHStack, self).__getitem__(indices),
                             self.nb_threads,
                             self.nb_inner_threads)

    def _close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.terminate()
            self.pool = None

    @staticmethod
    def _block_priority(x):
        if isinstance(x, np.ndarray):
            return x.shape[1]
        else:  # KarmaSparse dot is 2 times slow than numpy.dot
            return int(2. * x.nnz / max(x.shape[0], 1))

    def dot(self, w, row_indices=None):
        with blas_level_threads(self.nb_inner_threads):
            if self.is_block:
                def _dot(i):
                    x = self.X[i]
                    if row_indices is not None:
                        x = x[row_indices]
                    return safe_dot_torch(x, w[self.dims[i]:self.dims[i + 1]].astype(x.dtype, copy=False))

                if self.pool is not None:
                    return reduce(np.add, self.pool.imap_unordered(_dot, self.order))
                else:
                    return reduce(np.add, imap(_dot, self.order))
            else:
                return safe_dot_torch(self.X, w.astype(self.X.dtype, copy=False))

    def transpose_dot(self, w):
        with blas_level_threads(self.nb_inner_threads):
            if self.is_block:
                def _dot(i):
                    x = self.X[i]
                    return safe_dot_torch(x.T, w.astype(x.dtype, copy=False))

                if self.pool is not None:
                    return np.hstack(take_indices(self.pool.map(_dot, self.order), self.reverse_order))
                else:
                    return np.hstack(map(_dot, range(len(self.X))))
            else:
                return safe_dot_torch(self.X.T, w.astype(self.X.dtype, copy=False))

    @property
    def row_nnz(self):
        """This function computes the mean row density of the virtual hstack, defined as the sum of the row densities,
        the mean number of non zero terms by row, of each block among the virtual hstack.
        """
        with blas_level_threads(self.nb_inner_threads):
            if self.is_block:
                def _density(i):
                    x = self.X[i]
                    return number_nonzero(x) * 1. / x.shape[0]

                if self.pool is not None:
                    return np.sum(self.pool.map(_density, self.order))
                else:
                    return np.sum(map(_density, self.order))
            else:
                return number_nonzero(self.X) * 1. / self.X.shape[0]


def check_axis_values(y):
    axis_unique_values = parallel_unique(y)
    if axis_unique_values.tolist() not in ([0.], [1.], [0., 1.]):
        raise ValueError('Set of values taken by axis, {}, is not a subset of [0, 1]'
                         .format(axis_unique_values))


def create_basic_virtual_hstack(dataframe, inputs):
    def find_dp(inp):
        # a bit dirty hack
        backend = dataframe[inp]._backend
        if hasattr(backend, 'dependencies') and hasattr(backend, 'instruction') and \
                backend.instruction.name == "direct_product" and len(backend.dependencies) == 2:
            return tuple([x[0] for x in backend.dependencies])  # this returns names
        else:
            return inp

    # local cache to take all columns only once
    local_column_cache, features = {}, []

    for col in map(find_dp, inputs):
        if isinstance(col, tuple):
            lname, rname = col
            left = local_column_cache.get(lname, dataframe[lname][:])
            local_column_cache[lname] = left
            right = local_column_cache.get(rname, dataframe[rname][:])
            local_column_cache[rname] = right
            features.append((left, right))
        else:
            features.append(local_column_cache.get(col, dataframe[col][:]))
            local_column_cache[col] = features[-1]
    return BasicVirtualHStack(features)


class use_seed():
    def __init__(self, seed=None):
        if seed is not None:
            if not isinstance(seed, int) and not isinstance(seed, np.int32):
                if isinstance(seed, Hashable):
                    self.seed = abs(np.int32(hash(seed)))  # hash returns int64, np.seed needs to be int32
                else:
                    raise ValueError("Invalid seed value `{}`, It should be an integer.".format(seed))
            elif seed < 0:
                raise ValueError("Invalid seed value `{}`, It should be positive.".format(seed))
            else:
                self.seed = seed
        else:
            self.seed = None

    def __enter__(self):
        self.np_state = np_get_state()
        self.py_state = py_get_state()
        if self.seed is not None:
            np_seed(self.seed)
            py_seed(self.seed)
        # Note: Returning self means that in "with ... as x", x will be self
        return self

    def __exit__(self, typ, val, _traceback):
        if self.seed is not None:
            np_set_state(self.np_state)
            py_set_state(self.py_state)

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kw):
            seed = self.seed if self.seed is not None else kw.pop('seed', None)
            with use_seed(seed):
                return f(*args, **kw)

        return wrapper

#
# Copyright tinyclues, All rights reserved
#

from __future__ import absolute_import
from contextlib import contextmanager
import ctypes
from ctypes.util import find_library

import os
from cyperf.tools.types import set_open_mp_num_thread, get_open_mp_num_thread
from numpy.linalg import _umath_linalg
from multiprocessing import cpu_count

__all__ = ['blas_threads', 'set_open_blas_num_thread', 'open_mp_threads', 'set_open_mp_num_thread']


def find_openblas_lib():
    """
    do not use it for now as it introduces SIGSEVs when using VirtualHStack, cf tinyclues/datascience#115
    >>> find_openblas_lib()
    'libopenblasp-r0-382c8f3a.3.5.dev.so'
    """
    try:
        p = os.popen('ldd {}'.format(_umath_linalg.__file__), 'r')
        deps = [x.split(' ')[0][1:] for x in p.readlines() if 'openblas' in x]
        assert len(deps) == 1
        return deps[0]
    except:
        return find_library('openblas')


openblas_lib = ctypes.cdll.LoadLibrary(find_openblas_lib())
mkl_lib = ctypes.cdll.LoadLibrary(find_library('mkl_rt'))


def mkl_get_num_thread():
    try:
        return mkl_lib.mkl_get_max_threads()
    except AttributeError:
        return 1


def mkl_set_num_thread(n_threads):
    """
    see https://stackoverflow.com/questions/28283112/using-mkl-set-num-threads-with-numpy

    >>> mkl_set_num_thread(7)
    >>> mkl_get_num_thread()
    7
    """
    try:
        mkl_lib.mkl_set_num_threads(ctypes.byref(ctypes.c_int(n_threads)))
    except (AttributeError, TypeError):
        pass


def get_open_blas_num_thread(safe=False):
    """
    see http://stackoverflow.com/questions/29559338/set-max-number-of-threads-at-runtime-on-numpy-openblas
    """
    try:
        return openblas_lib.openblas_get_num_threads()
    except AttributeError:
        try:
            return mkl_get_num_thread()
        except ImportError:
            pass
        if safe:
            from .runtime import KarmaSetup  # circular import
            return KarmaSetup.open_blas_nb_thread
        else:
            return -1


def set_open_blas_num_thread(n_threads):
    n_threads = max(1, min(n_threads, cpu_count()))
    try:
        openblas_lib.openblas_set_num_threads(n_threads)
    except AttributeError:
        pass
    mkl_set_num_thread(n_threads)


@contextmanager
def blas_threads(n_threads):
    """
    Sets number of threads in MKL to a given number

    Args:
        n_threads: int or None, in case of None this context has no effect

    >>> old = get_open_blas_num_thread()
    >>> expected_n_threads = 1
    >>> with blas_threads(1): get_open_blas_num_thread() == 1
    True
    >>> old == get_open_blas_num_thread()
    True
    >>> with blas_threads(None): old == get_open_blas_num_thread()
    True
    """
    old_num_threads = get_open_blas_num_thread()
    if n_threads is not None:
        set_open_blas_num_thread(n_threads)
    yield
    if n_threads is not None:
        set_open_blas_num_thread(old_num_threads)


@contextmanager
def open_mp_threads(n_threads):
    """
    Sets number of threads in OpenMP to a given number

    Args:
        n_threads: int or None, in case of None this context has no effect

    >>> with open_mp_threads(3): get_open_mp_num_thread() == 3
    True
    >>> with open_mp_threads(2): get_open_mp_num_thread() == 2
    True
    """
    if n_threads is not None:
        initial_thread_value = get_open_mp_num_thread()
        set_open_mp_num_thread(n_threads)
    yield
    if n_threads is not None:
        set_open_mp_num_thread(initial_thread_value)

@contextmanager
def numexpr_threads(n_threads):
    """
    Sets number of threads that numexpr can use

    Args:
        n_threads: int or None, in case of None this context has no effect

    >>> with numexpr_threads(1): get_numexpr_num_thread() == 1
    True
    >>> with numexpr_threads(2): get_numexpr_num_thread() == 2
    True
    """
    if n_threads is not None:
        n_threads = min(n_threads, cpu_count())
        initial_thread_value = get_numexpr_num_thread()
        set_numexpr_num_thread(n_threads)
    yield
    if n_threads is not None:
        set_numexpr_num_thread(initial_thread_value)


@contextmanager
def blas_level_threads(n_threads):
    """
    Sets number of threads to use in parallelism for blas level calls. Sets blas, openmp
    and numexpr threads at some common value.

    Args:
        n_threads: int or None, in case of None this context has no effect

    >>> with blas_level_threads(1): get_numexpr_num_thread() == 1
    True
    >>> with blas_level_threads(3): get_open_blas_num_thread() == 3
    True
    """
    with blas_threads(n_threads), open_mp_threads(n_threads), numexpr_threads(n_threads), torch_threads(n_threads):
        yield

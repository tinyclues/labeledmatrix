#
# Copyright tinyclues, All rights reserved
#
import sys
from glob import glob
from pip._internal.req import parse_requirements

from setuptools import find_packages, setup
from Cython.Build import cythonize
from Cython.Distutils.extension import Extension

import multiprocessing
from numpy.distutils.misc_util import get_info
import numpy as np


root_path = 'cyperf'

SOURCE_FILE = [
    'clustering/sparse_affinity_propagation.pyx',
    'clustering/hierarchical.pyx',
    'clustering/space_tools.pyx',
    'clustering/heap.pyx',
    'matrix/karma_sparse.pyx',
    'matrix/routine.pyx',
    'matrix/rank_dispatch.pyx',
    'matrix/argmax_dispatch.pyx',
    'tools/getter.pyx',
    'tools/sort_tools.pyx',
    'tools/types.pyx',
    'tools/curve.pyx',
    'indexing/column_index.pyx',
    'indexing/indexed_list.pyx',
    'where/indices_where_int.pyx',
    'where/indices_where_long.pyx',
    'hashing/hash_tools.pyx']


cargs = ['-O3', '-Wno-unused-function', '-Wno-maybe-uninitialized', '-Wno-unused-variable',
         '-std=c++11', '-ffast-math', '-fopenmp', '-msse4.2']
largs = ['-fopenmp', '-msse4.2']

info = get_info('npymath')


EXTENSIONS = [Extension(root_path + '.' + f.replace('.pyx', '').replace('/', '.'), glob(root_path + '/' + f),
                        language="c++",
                        include_dirs=info['include_dirs'],
                        library_dirs=info['library_dirs'],
                        libraries=info['libraries'],
                        extra_compile_args=cargs,
                        extra_link_args=largs,
                        cython_directives={'language_level': 2, 'embedsignature': True}) for f in SOURCE_FILE]


requirements = [str(i.req) for i in parse_requirements("requirements.txt", session=False)]
test_requirements = [str(i.req) for i in parse_requirements("test_requirements.txt", session=False)]


VERSION = "1"
NB_COMPILE_JOBS = 4


def setup_given_extensions(extensions):
    setup(name='karma-perf',
          version=str(VERSION),
          packages=find_packages(exclude=['tests*']),
          ext_modules=cythonize(extensions),
          include_dirs=[np.get_include()],
          install_requires=requirements,
          tests_require=test_requirements,
          url='https://github.com/tinyclues/karma-perf')


def setup_extensions_in_sequential():
    setup_given_extensions(EXTENSIONS)


def setup_extensions_in_parallel():
    cythonize(EXTENSIONS, nthreads=NB_COMPILE_JOBS)
    pool = multiprocessing.Pool(processes=NB_COMPILE_JOBS)
    pool.map(setup_given_extensions, EXTENSIONS)
    pool.close()
    pool.join()


if "build_ext" in sys.argv:
    setup_extensions_in_parallel()
else:
    setup_extensions_in_sequential()

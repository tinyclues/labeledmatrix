#
# Copyright tinyclues, All rights reserved
#
import sys
from glob import glob
import multiprocessing

from setuptools import find_packages, setup
from Cython.Build import cythonize
from Cython.Distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import get_info

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

cargs = ['-O3', '-std=c++11', '-ffast-math', '-fopenmp', '-lgomp', '-msse4.2',
         '-Wno-unused-function', '-Wno-maybe-uninitialized', '-Wno-unused-variable']
largs = ['-fopenmp', '-lgomp']

info = get_info('npymath')

compiler_directives = {'language_level': sys.version_info[0], 'embedsignature': True}

EXTENSIONS = [Extension(root_path + '.' + f.replace('.pyx', '').replace('/', '.'),
                        glob(root_path + '/' + f),
                        language="c++",
                        include_dirs=info['include_dirs'],
                        library_dirs=info['library_dirs'],
                        libraries=info['libraries'],
                        extra_compile_args=cargs,
                        extra_link_args=largs) for f in SOURCE_FILE]


#
# Requirements here declare to the _users_ of this lib, what are the dependencies to run it properly.
# /!\ This must be kept in sync with requirements.txt
#    Unfortunately, a `parse_requirements("requirements.txt")` will not work, as this setup.py is ran by tox
#    within a virtualenv in a separate directory
#
requirements = ["numpy>=1.16",
                "scipy>=1.2",
                "cython>=0.29",
                "pandas>=0.22"]

VERSION = "1"
NB_COMPILE_JOBS = multiprocessing.cpu_count()


def setup_given_extensions(extensions):
    setup(name='karma-perf',
          version=str(VERSION),
          packages=find_packages(exclude=['tests*']),
          ext_modules=cythonize(extensions, compiler_directives=compiler_directives, nthreads=NB_COMPILE_JOBS),
          include_dirs=[np.get_include()],
          install_requires=requirements,
          url='https://github.com/tinyclues/odyssey/karma-perf')


def setup_extensions_in_sequential():
    setup_given_extensions(EXTENSIONS)


def setup_extensions_in_parallel():
    pool = multiprocessing.Pool(processes=NB_COMPILE_JOBS)
    pool.map(setup_given_extensions, EXTENSIONS)
    pool.close()
    pool.terminate()


if "build_ext" in sys.argv:
    setup_extensions_in_parallel()
else:
    setup_extensions_in_sequential()

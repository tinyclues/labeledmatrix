import os
import sys
from glob import glob
import multiprocessing

from setuptools import find_packages, setup
from setuptools.extern import packaging
from pip._internal.req import parse_requirements
from Cython.Build import cythonize
from Cython import Tempita as tempita
from Cython.Distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import get_info

root_path = 'cyperf'


# trying to sort in increasing compilation time to get more win from parallel build
SOURCE_FILE = [
    'clustering/sparse_affinity_propagation.pyx',
    'clustering/hierarchical.pyx',
    'clustering/space_tools.pyx',
    'clustering/heap.pyx',
    'hashing/hash_tools.pyx',
    'matrix/karma_sparse.pyx',
    'matrix/routine.pyx',
    'matrix/rank_dispatch.pyx',
    'matrix/argmax_dispatch.pyx',
    'tools/types.pyx',
    'tools/getter.pyx',
    'tools/sort_tools.pyx',
    'tools/curve.pyx',
    'indexing/column_index.pyx',
    'indexing/indexed_list.pyx',
    'where/cy_filter.pyx']

TEMPLATE_SOURCE = ['tools/vector.pyx.in',
                   'tools/parallel_sort_routine.pyx.in']

basic_cargs = ['-O3', '-std=c++11', '-fopenmp', '-lgomp', '-msse4.2',
               '-Wno-unused-function', '-Wno-maybe-uninitialized', '-Wno-unused-variable']
largs = ['-fopenmp', '-lgomp']


def cargs(f):
    if 'parallel_sort_routine' in f or 'karma_sparse' in f:
        # we need to remove flag '-ffast-math' flag to deal with nan
        # see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=25975 and https://github.com/cython/cython/issues/550
        return basic_cargs
    else:
        return basic_cargs + ['-ffast-math']


info = get_info('npymath')

compiler_directives = {'language_level': sys.version_info[0], 'embedsignature': True}


def render_tempita_pyx(file_path):
    source = [root_path + '/' + file_path]

    pxd_source = root_path + '/' + file_path.replace('.pyx.in', '.pxd.in')
    if os.path.exists(pxd_source):
        source.append(pxd_source)

    for f_name in source:
        destination = f_name.rstrip('.in')
        if os.path.exists(destination) and (os.stat(f_name).st_mtime < os.stat(destination).st_mtime):
            continue
        with open(f_name, "r") as f:
            tmpl = f.read()
        pyxcontent = tempita.sub(tmpl)
        with open(destination, "w") as f:
            f.write(pyxcontent)

    return file_path.rstrip('.in')


EXTENSIONS = [Extension(root_path + '.' + f.replace('.pyx', '').replace('/', '.'),
                        glob(root_path + '/' + f),
                        language="c++",
                        include_dirs=info['include_dirs'],
                        library_dirs=info['library_dirs'],
                        libraries=info['libraries'],
                        extra_compile_args=cargs(f),
                        extra_link_args=largs)
              for f in [render_tempita_pyx(f) for f in TEMPLATE_SOURCE] + SOURCE_FILE]

requirements = [str(i.req) for i in parse_requirements("requirements.txt", session=False)]

def get_version():
    tag = os.getenv('CIRCLE_TAG', None)
    if tag is None:
        return "0.local"

    ver = packaging.version.Version(tag)  # To force to be a valid tag version
    normalized_version = str(ver)
    if normalized_version != tag:
        raise packaging.version.InvalidVersion("Should probably be {}".format(normalized_version))
    return normalized_version

NB_COMPILE_JOBS = 2 if os.getenv('CIRCLECI', False) else multiprocessing.cpu_count()

def setup_given_extensions(extensions):
    setup(name='karma-perf',
          version=get_version(),
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

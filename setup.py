#!/usr/bin/env python

from glob import glob
import os
import sys

from pip._internal.req import parse_requirements
from setuptools import setup
from distutils.dist import Distribution


def render_tempita():
    from Cython import Tempita

    for tpl in glob("cyperf/**/*.pyx.in") + glob("cyperf/**/*.pxd.in"):
        dest = tpl[:-3]
        if not os.path.exists(dest) or os.stat(tpl).st_mtime > os.stat(dest).st_mtime:
            with open(tpl, 'r') as fd_in, open(dest, 'w') as fd_out:
                fd_out.write(Tempita.sub(fd_in.read()))


def create_extension(template, kwds):
    info = get_info('npymath')

    kwds['include_dirs'] = kwds.get('include_dirs', []) + info['include_dirs']
    kwds['library_dirs'] = info['library_dirs']
    kwds['libraries'] = info['libraries']
    kwds['extra_compile_args'] = ['-O3', '-std=c++11', '-fopenmp', '-lgomp', '-msse4.2', '-Wno-unused-function',
                                  '-Wno-maybe-uninitialized', '-Wno-unused-variable']
    kwds['extra_link_args'] = ['-fopenmp', '-lgomp']

    # we need to remove flag '-ffast-math' flag to deal with nan
    # see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=25975 and https://github.com/cython/cython/issues/550
    if kwds['name'] not in ('cyperf.tools.parallel_sort_routine', 'cyperf.matrix.karma_sparse'):
        kwds['extra_compile_args'].append('-ffast-math')

    return default_create_extension(template, kwds)


if "build_ext" in sys.argv:
    render_tempita()

try:
    from numpy.distutils.misc_util import get_info
    from Cython.Build import cythonize
    from Cython.Build.Dependencies import default_create_extension
except ImportError:  # for conda _load_setup_py_data jinja template
    ext_modules = []
else:
    try:
        dist = Distribution()
        dist.parse_command_line()
        nthreads = int(dist.command_options['build_ext']['parallel'][1])
    except Exception:
        nthreads = 1

    ext_modules = cythonize(
        "cyperf/**/*.pyx",
        create_extension=create_extension,
        compiler_directives={'language_level': sys.version_info[0], 'embedsignature': True},
        language='c++',
        nthreads=nthreads,
    )

setup(
    ext_modules=ext_modules,
    install_requires=[str(i.req) for i in parse_requirements("requirements.txt", session=False)]
)
#!/usr/bin/env python

from glob import glob
import os
import sys

from setuptools import setup

from pipenv.utils.dependencies import convert_deps_to_pip  # TODO consider pyproject.toml
from pipenv.project import Project

from numpy.distutils.misc_util import get_info
from Cython.Build import cythonize
from Cython.Build.Dependencies import default_create_extension


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


ext_modules = cythonize(
    "cyperf/**/*.pyx",
    create_extension=create_extension,
    compiler_directives={'language_level': sys.version_info[0], 'embedsignature': True},
)

pfile = Project(chdir=False).parsed_pipfile
setup(
    ext_modules=ext_modules,
    install_requires=list(convert_deps_to_pip(pfile['packages']).values())
)

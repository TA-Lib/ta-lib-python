#!/usr/bin/env python

import sys
import os
import os.path
import warnings

from setuptools import setup, Extension

import numpy

platform_supported = False

lib_talib_name = 'ta-lib'               # the name as of TA-Lib 0.6.1

if any(s in sys.platform for s in ['darwin', 'linux', 'bsd', 'sunos']):
    platform_supported = True
    include_dirs = [
        '/usr/include',
        '/usr/local/include',
        '/opt/include',
        '/opt/local/include',
        '/opt/homebrew/include',
        '/opt/homebrew/opt/ta-lib/include',
    ]
    library_dirs = [
        '/usr/lib',
        '/usr/local/lib',
        '/usr/lib64',
        '/usr/local/lib64',
        '/opt/lib',
        '/opt/local/lib',
        '/opt/homebrew/lib',
        '/opt/homebrew/opt/ta-lib/lib',
    ]

elif sys.platform == "win32":
    platform_supported = True
    lib_talib_name = 'ta-lib-static'
    include_dirs = [
        r"c:\ta-lib\c\include",
        r"c:\Program Files\TA-Lib\include",
        r"c:\Program Files (x86)\TA-Lib\include",
    ]
    library_dirs = [
        r"c:\ta-lib\c\lib",
        r"c:\Program Files\TA-Lib\lib",
        r"c:\Program Files (x86)\TA-Lib\lib",
    ]

if 'TA_INCLUDE_PATH' in os.environ:
    include_dirs = os.environ['TA_INCLUDE_PATH'].split(os.pathsep)

if 'TA_LIBRARY_PATH' in os.environ:
    library_dirs = os.environ['TA_LIBRARY_PATH'].split(os.pathsep)

if not platform_supported:
    raise NotImplementedError(sys.platform)

for path in library_dirs:
    try:
        files = os.listdir(path)
        if any(lib_talib_name in f for f in files):
            break
    except OSError:
        pass
else:
    warnings.warn('Cannot find ta-lib library, installation may fail.')

# Get the Cython build_ext or fall back to setuptools build_ext
try:
    from Cython.Distutils import build_ext
    has_cython = True
except ImportError:
    from setuptools.command.build_ext import build_ext
    has_cython = False

class NumpyBuildExt(build_ext):
    """
    Custom build_ext command that adds numpy's include_dir to extensions.
    """

    def build_extensions(self):
        """
        Add numpy's include directory to Extension includes.
        """
        numpy_incl = numpy.get_include()
        for ext in self.extensions:
            ext.include_dirs.append(numpy_incl)

        super().build_extensions()

cmdclass = {'build_ext': NumpyBuildExt}

ext_modules = [
    Extension(
        'talib._ta_lib',
        ['talib/_ta_lib.pyx' if has_cython else 'talib/_ta_lib.c'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[lib_talib_name],
        runtime_library_dirs=[] if sys.platform == 'win32' else library_dirs)
]

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

#!/usr/bin/env python

import sys
import os
import warnings

from distutils.dist import Distribution
from distutils.version import LooseVersion

display_option_names = Distribution.display_option_names + ['help', 'help-commands']
query_only = any('--' + opt in sys.argv for opt in display_option_names) or len(sys.argv) < 2 or sys.argv[1] == 'egg_info'

# Use setuptools for querying the package, normal builds use distutils
if query_only:
    try:
        from setuptools import setup
    except ImportError:
        from distutils.core import setup
else:
    from distutils.core import setup

from distutils.extension import Extension

lib_talib_name = 'ta_lib'  # the underlying C library's name

platform_supported = False
for prefix in ['darwin', 'linux', 'bsd', 'sunos']:
    if prefix in sys.platform:
        platform_supported = True
        include_dirs = [
            '/usr/include',
            '/usr/local/include',
            '/opt/include',
            '/opt/local/include',
        ]
        if 'TA_INCLUDE_PATH' in os.environ:
            include_dirs.append(os.environ['TA_INCLUDE_PATH'])
        lib_talib_dirs = [
            '/usr/lib',
            '/usr/local/lib',
            '/usr/lib64',
            '/usr/local/lib64',
            '/opt/lib',
            '/opt/local/lib',
        ]
        if 'TA_LIBRARY_PATH' in os.environ:
            lib_talib_dirs.append(os.environ['TA_LIBRARY_PATH'])
        break

if sys.platform == "win32":
    platform_supported = True
    lib_talib_name = 'ta_libc_cdr'
    include_dirs = [r"c:\ta-lib\c\include"]
    lib_talib_dirs = [r"c:\ta-lib\c\lib"]

if not platform_supported:
    raise NotImplementedError(sys.platform)

# Do not require numpy or cython for just querying the package
if not query_only:
    import numpy
    include_dirs.insert(0, numpy.get_include())

try:
    import Cython
    if LooseVersion(Cython.__version__) >= LooseVersion('0.25'):
        from Cython.Distutils.old_build_ext import old_build_ext as build_ext
    else:
        from Cython.Distutils import build_ext
    has_cython = True
except ImportError:
    has_cython = False

for lib_talib_dir in lib_talib_dirs:
    try:
        files = os.listdir(lib_talib_dir)
        if any(lib_talib_name in f for f in files):
            break
    except OSError:
        pass
else:
    warnings.warn('Cannot find ta-lib library, installation may fail.')

cmdclass = {}
if has_cython:
    cmdclass['build_ext'] = build_ext

ext_modules = []
for name in ['common', 'func', 'abstract', 'stream']:
    ext = Extension(
        'talib.%s' % name,
        [('talib/%s.pyx' if has_cython else 'talib/%s.c') % name],
        include_dirs = include_dirs,
        library_dirs = lib_talib_dirs,
        libraries = [lib_talib_name]
    )
    ext_modules.append(ext)

setup(
    name = 'TA-Lib',
    version = '0.4.10',
    description = 'Python wrapper for TA-Lib',
    author = 'John Benediktsson',
    author_email = 'mrjbq7@gmail.com',
    url = 'http://github.com/mrjbq7/ta-lib',
    download_url = 'https://github.com/mrjbq7/ta-lib/releases',
    classifiers = [
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Cython",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    packages = ['talib'],
    ext_modules = ext_modules,
    cmdclass = cmdclass,
    requires = ['numpy'],
)

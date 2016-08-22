#!/usr/bin/env python

import os
import sys
import glob
from os.path import join as join_path

from distutils.dist import Distribution

PRJ_DIR = os.path.dirname(os.path.abspath(__file__))
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
    from Cython.Distutils import build_ext
    has_cython = True
except ImportError:
    has_cython = False

vendor_include_dir = join_path(PRJ_DIR, "vendor", "include")
vendor_talib_dir = join_path(PRJ_DIR, "vendor", "ta-lib")

libraries = [lib_talib_name]
for lib_talib_dir in lib_talib_dirs:
    try:
        files = os.listdir(lib_talib_dir)
        if any(lib_talib_name in f for f in files):
            break
    except OSError:
        pass
else:
    include_dirs.append(vendor_include_dir)
    include_dirs.append(join_path(vendor_talib_dir, "include"))
    libraries = []


cmdclass = {}
if has_cython:
    cmdclass['build_ext'] = build_ext

ext_modules = []
for name in ['common', 'func', 'abstract', 'stream']:
    local_include_dirs = include_dirs[:]

    sources = [('talib/%s.pyx' if has_cython else 'talib/%s.c') % name]
    if not libraries:
        ta_common_dir = join_path(vendor_talib_dir, "src", "ta_common")
        ta_abstract_dir = join_path(vendor_talib_dir, "src", "ta_abstract")
        ta_func_dir = join_path(vendor_talib_dir, "src", "ta_func")
        if name == "common":
            local_include_dirs.append(ta_common_dir)
            sources.extend(glob.glob(join_path(ta_common_dir, "*.c")))
            sources.extend(glob.glob(join_path(ta_func_dir, "*.c")))
        elif name == "abstract":
            local_include_dirs.append(ta_common_dir)
            local_include_dirs.append(ta_abstract_dir)
            local_include_dirs.append(join_path(ta_abstract_dir, "frames"))
            sources.extend(glob.glob(join_path(ta_abstract_dir, "*.c")))
            sources.remove(join_path(ta_abstract_dir, "excel_glue.c"))
            sources.extend(glob.glob(join_path(ta_abstract_dir, "*", "*.c")))
        elif name == "func":
            local_include_dirs.append(ta_common_dir)
            sources.extend(glob.glob(join_path(ta_func_dir, "*.c")))

    ext = Extension(
        'talib.%s' % name,
        sources,
        include_dirs=local_include_dirs,
        library_dirs=lib_talib_dirs,
        libraries=libraries,
    )
    ext_modules.append(ext)

setup(
    name='TA-Lib',
    version='0.4.10',
    description='Python wrapper for TA-Lib',
    author='John Benediktsson',
    author_email='mrjbq7@gmail.com',
    url='http://github.com/mrjbq7/ta-lib',
    download_url='https://github.com/mrjbq7/ta-lib/releases',
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    packages=['talib'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    requires=['numpy'],
)

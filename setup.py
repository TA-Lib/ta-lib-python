#!/usr/bin/env python

import sys
import glob
import os
import warnings

from distutils.dist import Distribution
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['tests']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


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
sources = []
libraries = []
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

for lib_talib_dir in lib_talib_dirs:
    try:
        files = os.listdir(lib_talib_dir)
        if any(lib_talib_name in f for f in files):
            break
    except OSError:
        pass
    libraries = [lib_talib_name]
else:
    warnings.warn(
        "Cannot find ta-lib library, Try to build from source. "
        "Installation may fail."
    )
    # find vendor/ta-lib -name "*.h" -exec dirname {} \; | sort | uniq
    vendor_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
    )
    vendor_include_dir = os.path.join(
        vendor_dir,
        "include",
    )
    vendor_talib_dir = os.path.join(
        vendor_dir,
        "ta-lib",
    )
    talib_include_dirs = [
        ("include", ),
        ("src", "ta_abstract"),
        ("src", "ta_abstract", "frames"),
        ("src", "ta_common"),
        ("src", "ta_func"),
    ]
    include_dirs.append(os.path.join(vendor_include_dir))
    include_dirs.extend((
        os.path.join(vendor_talib_dir, *path_args)
        for path_args in talib_include_dirs
    ))

    talib_source_dirs = [
        ("ta_abstract", ),
        ("ta_abstract", "frames"),
        ("ta_abstract", "tables"),
        ("ta_common", ),
        ("ta_func", )
    ]
    for path_args in talib_source_dirs:
        source_dir = os.path.join(vendor_talib_dir, "src", *path_args)
        sources.extend(glob.glob(os.path.join(source_dir, "*.c")))
    sources.remove(
        os.path.join(vendor_talib_dir, "src", "ta_abstract", "excel_glue.c")
    )
    libraries = []
    lib_talib_dirs = []

cmdclass = {"test": PyTest}
if has_cython:
    cmdclass['build_ext'] = build_ext

ext_modules = [
    Extension(
        'talib.c_ta_lib',
        ["talib/c_ta_lib.pyx" if has_cython else "talib/c_ta_lib.c"] + sources,
        include_dirs=include_dirs,
        library_dirs=lib_talib_dirs,
        libraries=libraries
    )
]


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

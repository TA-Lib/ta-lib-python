#!/usr/bin/env python

import sys
import os
import os.path
import warnings

try:
    from setuptools import setup, Extension
    from setuptools.dist import Distribution
    requires = {
        "install_requires": ["numpy"],
        "setup_requires": ["numpy"]
    }
except ImportError:
    from distutils.core import setup
    from distutils.dist import Distribution
    from distutils.extension import Extension
    requires = {"requires": ["numpy"]}

lib_talib_name = 'ta_lib'  # the underlying C library's name

platform_supported = False

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
    lib_talib_name = 'ta_libc_cdr'
    include_dirs = [r"c:\ta-lib\c\include"]
    library_dirs = [r"c:\ta-lib\c\lib"]

if 'TA_INCLUDE_PATH' in os.environ:
    paths = os.environ['TA_INCLUDE_PATH'].split(os.pathsep)
    include_dirs.extend(path for path in paths if path)

if 'TA_LIBRARY_PATH' in os.environ:
    paths = os.environ['TA_LIBRARY_PATH'].split(os.pathsep)
    library_dirs.extend(path for path in paths if path)

if not platform_supported:
    raise NotImplementedError(sys.platform)

try:
    from Cython.Distutils import build_ext as cython_build_ext
    has_cython = True
except ImportError:
    has_cython = False

for path in library_dirs:
    try:
        files = os.listdir(path)
        if any(lib_talib_name in f for f in files):
            break
    except OSError:
        pass
else:
    warnings.warn('Cannot find ta-lib library, installation may fail.')


class LazyBuildExtCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """

    def __contains__(self, key):
        return (key == 'build_ext' or
                super(LazyBuildExtCommandClass, self).__contains__(key))

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super(LazyBuildExtCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        global include_dirs
        if key != 'build_ext':
            return super(LazyBuildExtCommandClass, self).__getitem__(key)

        import numpy
        if has_cython:
            org_build_ext = cython_build_ext
        else:
            from setuptools.command.build_ext import build_ext as org_build_ext

        # Cython_build_ext isn't a new-style class in Py2.
        class build_ext(org_build_ext, object):
            """
            Custom build_ext command that lazily adds numpy's include_dir to
            extensions.
            """

            def build_extensions(self):
                """
                Lazily append numpy's include directory to Extension includes.
                This is done here rather than at module scope because setup.py
                may be run before numpy has been installed, in which case
                importing numpy and calling `numpy.get_include()` will fail.
                """
                numpy_incl = numpy.get_include()
                for ext in self.extensions:
                    ext.include_dirs.append(numpy_incl)

                super(build_ext, self).build_extensions()

        return build_ext


cmdclass = LazyBuildExtCommandClass()

ext_modules = [
    Extension(
        'talib._ta_lib',
        ['talib/_ta_lib.pyx' if has_cython else 'talib/_ta_lib.c'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[lib_talib_name],
        runtime_library_dirs=[] if sys.platform == 'win32' else library_dirs)
]

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='TA-Lib',
    version='0.4.28',
    description='Python wrapper for TA-Lib',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='John Benediktsson',
    author_email='mrjbq7@gmail.com',
    url='http://github.com/ta-lib/ta-lib-python',
    download_url='https://github.com/ta-lib/ta-lib-python/releases',
    license='BSD',
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
    **requires)

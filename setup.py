#!/usr/bin/env python

import sys
import os
import warnings

from distutils.dist import Distribution

try:
    from setuptools import setup, Extension
    requires = {"install_requires": ["numpy"], "setup_requires": ["numpy"]}
except:
    from distutils.core import setup
    from distutils.extension import Extension
    requires = {"requires": ["numpy"]}

lib_talib_name = 'ta_lib'  # the underlying C library's name

runtime_lib_dirs = []

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
            runtime_lib_dirs = os.environ['TA_LIBRARY_PATH']
            if runtime_lib_dirs:
                runtime_lib_dirs = runtime_lib_dirs.split(os.pathsep)
                lib_talib_dirs.extend(runtime_lib_dirs)
        break

if sys.platform == "win32":
    platform_supported = True
    lib_talib_name = 'ta_libc_cdr'
    include_dirs = [r"c:\ta-lib\c\include"]
    lib_talib_dirs = [r"c:\ta-lib\c\lib"]

if not platform_supported:
    raise NotImplementedError(sys.platform)

try:
    from Cython.Distutils import build_ext as cython_build_ext
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
        library_dirs=lib_talib_dirs,
        libraries=[lib_talib_name],
        runtime_library_dirs=runtime_lib_dirs)
]
setup(
    name='TA-Lib',
    version='0.4.19',
    description='Python wrapper for TA-Lib',
    author='John Benediktsson',
    author_email='mrjbq7@gmail.com',
    url='http://github.com/mrjbq7/ta-lib',
    download_url='https://github.com/mrjbq7/ta-lib/releases',
    license='BSD',
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
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

#!/usr/bin/env python

import os
import sys
import warnings
import setuptools_scm  # noqa: F401

try:
    from setuptools import setup, Extension
    from setuptools.dist import Distribution

    requires = {
        'install_requires': ['numpy <1.23'],
        'setup_requires': ['numpy <1.23']
    }
except ImportError:
    from distutils.core import setup
    from distutils.dist import Distribution
    from distutils.extension import Extension

    requires = {'requires': ['numpy <1.23']}

lib_talib_name = 'ta_lib'  # the underlying C library's name

platform_supported = False
print('Platform: ', sys.platform)
print('Environment: ', [e for e in os.environ if not e.startswith('_')])

if any(s in sys.platform for s in ['darwin', 'linux', 'bsd', 'sunos']):
    print('Platform supported')
    platform_supported = True
    include_dirs = [
        '/usr/include/ta-lib',
        '/usr/local/include/ta-lib',
        '/opt/include/ta-lib',
        '/opt/local/include/ta-lib',
        '/opt/homebrew/include/ta-lib',
        '/opt/homebrew/opt/ta-lib/include/ta-lib',
    ]
    library_dirs = [
        '/usr/lib/ta-lib',
        '/usr/local/lib/ta-lib',
        '/usr/lib64/ta-lib',
        '/usr/local/lib64/ta-lib',
        '/opt/lib/ta-lib',
        '/opt/local/lib/ta-lib',
        '/opt/homebrew/lib/ta-lib',
        '/opt/homebrew/opt/ta-lib/lib/ta-lib',
    ]

elif sys.platform == 'win32':
    platform_supported = True
    lib_talib_name = 'ta_libc_cdr'
    include_dirs = [r'c:\ta-lib\c\include']
    library_dirs = [r'c:\ta-lib\c\lib']

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
        print('Path not found: ', path)
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
            raise AssertionError('build_ext overridden!')
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


ext_modules = [
    Extension(
        name='talib._ta_lib',
        sources=['src/talib/_ta_lib.pyx'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[lib_talib_name],
        runtime_library_dirs=[] if sys.platform == 'win32' else library_dirs,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass=LazyBuildExtCommandClass(),
    package_data={'talib': ['*.pxd', '*.pyx', '*.c', '*.h'],
                  'src/talib': ['*.pxd', '*.pyx', '*.c', '*.h']},
    **requires)
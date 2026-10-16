#!/usr/bin/env python

import sys
import os
import os.path
import hashlib
import shlex
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
import sysconfig

from setuptools import setup, Extension

import numpy

TALIB_C_VERSION = '0.8.1'
TALIB_C_SHA256 = 'ec59ccd88c0c77f618587d858787c8f9d06c40460a09d66751926f6fd670f985'


def build_talib(directory, plat_name):
    """Build the pinned C release locally, without installing it system-wide."""
    archive = directory / 'ta-lib.tar.gz'
    urllib.request.urlretrieve(
        f'https://github.com/TA-Lib/ta-lib/releases/download/v{TALIB_C_VERSION}/'
        f'ta-lib-{TALIB_C_VERSION}-src.tar.gz', archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != TALIB_C_SHA256:
        raise RuntimeError('TA-Lib C source checksum mismatch')
    with tarfile.open(archive) as source:
        source.extractall(directory)

    prefix = directory / 'install'
    build = directory / 'build'
    args = [
        'cmake', '-S', str(directory / f'ta-lib-{TALIB_C_VERSION}'),
        '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
        f'-DCMAKE_INSTALL_PREFIX={prefix}',
        '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_STATIC_LIBS=ON',
        '-DBUILD_DEV_TOOLS=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
    ]
    if sys.platform == 'win32':
        args += ['-A', {'win32': 'Win32', 'win-amd64': 'x64',
                        'win-arm64': 'ARM64'}[plat_name],
                 '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL']
    elif sys.platform == 'darwin':
        flags = shlex.split(os.environ.get('ARCHFLAGS', ''))
        archs = [flags[i + 1] for i, flag in enumerate(flags) if flag == '-arch']
        if archs:
            args.append('-DCMAKE_OSX_ARCHITECTURES=' + ';'.join(archs))
        target = (os.environ.get('MACOSX_DEPLOYMENT_TARGET') or
                  sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if target:
            args.append('-DCMAKE_OSX_DEPLOYMENT_TARGET=' + target)
    subprocess.check_call(args)
    subprocess.check_call(['cmake', '--build', str(build), '--config', 'Release'])
    subprocess.check_call(['cmake', '--install', str(build), '--config', 'Release'])
    return prefix


# Get the Cython build_ext or fall back to setuptools build_ext
try:
    from Cython.Distutils import build_ext
    has_cython = True
except ImportError:
    from setuptools.command.build_ext import build_ext
    has_cython = False

class NumpyBuildExt(build_ext):
    """
    Build TA-Lib C unless explicit library or header paths are supplied.
    """

    def build_extensions(self):
        """
        Respect explicit paths, otherwise build and statically link TA-Lib C.
        """
        if 'TA_LIBRARY_PATH' in os.environ or 'TA_INCLUDE_PATH' in os.environ:
            if sys.platform == 'win32':
                include_dirs = [r'c:\ta-lib\c\include',
                                r'c:\Program Files\TA-Lib\include',
                                r'c:\Program Files (x86)\TA-Lib\include']
                library_dirs = [r'c:\ta-lib\c\lib',
                                r'c:\Program Files\TA-Lib\lib',
                                r'c:\Program Files (x86)\TA-Lib\lib']
                library = 'ta-lib-static'
            else:
                include_dirs = ['/usr/include', '/usr/local/include', '/opt/include',
                                '/opt/local/include', '/opt/homebrew/include',
                                '/opt/homebrew/opt/ta-lib/include']
                library_dirs = ['/usr/lib', '/usr/local/lib', '/usr/lib64',
                                '/usr/local/lib64', '/opt/lib', '/opt/local/lib',
                                '/opt/homebrew/lib', '/opt/homebrew/opt/ta-lib/lib']
                library = 'ta-lib'
            if 'TA_INCLUDE_PATH' in os.environ:
                include_dirs = os.environ['TA_INCLUDE_PATH'].split(os.pathsep)
            if 'TA_LIBRARY_PATH' in os.environ:
                library_dirs = os.environ['TA_LIBRARY_PATH'].split(os.pathsep)
            for ext in self.extensions:
                ext.include_dirs += include_dirs + [numpy.get_include()]
                ext.library_dirs = library_dirs
                ext.libraries.append(library)
                ext.runtime_library_dirs = [] if sys.platform == 'win32' else library_dirs
            super().build_extensions()
            return

        with tempfile.TemporaryDirectory(prefix='ta-lib-') as temporary:
            prefix = build_talib(Path(temporary), self.plat_name)
            library = 'ta-lib-static.lib' if sys.platform == 'win32' else 'libta-lib.a'
            for ext in self.extensions:
                ext.include_dirs += [str(prefix / 'include'), numpy.get_include()]
                ext.extra_objects = [str(prefix / 'lib' / library)]
            super().build_extensions()

cmdclass = {'build_ext': NumpyBuildExt}

ext_modules = [
    Extension(
        'talib._ta_lib',
        ['talib/_ta_lib.pyx' if has_cython else 'talib/_ta_lib.c'],
        define_macros=[('Py_GIL_DISABLED', '1')]
            if sys.platform == 'win32' and sysconfig.get_config_var('Py_GIL_DISABLED') else [],
        libraries=[] if sys.platform == 'win32' else ['m'])
]

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

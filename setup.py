#!/usr/bin/env python

import sys
import os
import os.path
import re
import warnings

from setuptools import setup, Extension

import numpy


def sync_versions(root):
    """
    Copy the versions in pyproject.toml into the files that repeat them.

    Only in a git checkout: a local build updates the copies, a CI build
    (GITHUB_ACTIONS) fails on a stale one, and a build from an sdist or a
    PyPI install does nothing.
    """
    if not os.path.exists(os.path.join(root, '.git')):
        return
    try:
        import tomllib
    except ImportError:     # Python < 3.11
        return

    with open(os.path.join(root, 'pyproject.toml'), 'rb') as f:
        pyproject = tomllib.load(f)
    version = pyproject['project']['version']
    c_version = pyproject['tool']['ta-lib']['c-version']
    if not re.fullmatch(r'\d+\.\d+\.\d+', c_version):
        sys.exit('setup.py: [tool.ta-lib] c-version must look like 0.8.1, got %r'
                 % c_version)

    # (file, pattern whose group 1 is the version, value, match count; 0 = any)
    build_script = r'^TALIB_C_VER="\$\{TALIB_C_VER:=([^}]*)\}"'
    copies = [
        ('talib/__init__.py', r"^__version__ = '([^']*)'", version, 1),
        ('talib/__init__.py', r"^TA_LIB_C_REQUIRED = '([^']*)'", c_version, 1),
        ('.github/workflows/tests.yml', r'^ +version: "([^"]*)"', c_version, 1),
        ('.github/workflows/wheels.yml', r'^  TALIB_C_VER: (\S+)', c_version, 1),
        ('tools/build_talib_linux.sh', build_script, c_version, 1),
        ('tools/build_talib_macos.sh', build_script, c_version, 1),
        ('tools/build_talib_windows.cmd',
         r'^if not defined TALIB_C_VER set TALIB_C_VER=(\S+)', c_version, 1),
        ('README.md', r'\bta-lib-(\d+\.\d+\.\d+)', c_version, 0),
        ('README.md', r'/download/v(\d+\.\d+\.\d+)/', c_version, 0),
    ]

    stale = []
    for name in dict.fromkeys(path for path, _, _, _ in copies):
        path = os.path.join(root, name)
        with open(path, encoding='utf-8', newline='') as f:
            text = f.read()
        new = text
        for _, pattern, value, count in (c for c in copies if c[0] == name):
            parts, pos, found = [], 0, 0
            for m in re.finditer(pattern, new, re.MULTILINE):
                parts += [new[pos:m.start(1)], value]
                pos = m.end(1)
                found += 1
            if found == 0 or (count and found != count):
                sys.exit('setup.py: expected %s match(es) of %r in %s, found %d'
                         % (count or 'some', pattern, name, found))
            new = ''.join(parts) + new[pos:]
        if new == text:
            continue
        stale.append(name)
        if os.environ.get('GITHUB_ACTIONS') != 'true':
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(new)
            print('setup.py: updated %s from pyproject.toml' % name)

    if stale and os.environ.get('GITHUB_ACTIONS') == 'true':
        sys.exit('setup.py: out of date with pyproject.toml: %s\n'
                 'Run a local build with Python 3.11+ (e.g. "make build") '
                 'and commit the result.' % ', '.join(stale))


sync_versions(os.path.dirname(os.path.abspath(__file__)))

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

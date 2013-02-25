from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import os
import sys


lib_talib_name = 'ta_lib' # the underlying C library's name

platform_supported = False
for prefix in ['darwin', 'linux', 'bsd']:
    if prefix in sys.platform:
        platform_supported = True
        include_dirs = [
            numpy.get_include(),
            '/usr/include',
            '/usr/local/include',
            '/opt/include',
            '/opt/local/include',
            ]
        lib_talib_dirs = [
            '/usr/lib',
            '/usr/local/lib',
            '/opt/lib',
            '/opt/local/lib',
            ]
        break

if sys.platform == "win32":
    platform_supported = True
    lib_talib_name = 'ta_libc_cdr'
    include_dirs = [numpy.get_include(), r"c:\ta-lib\c\include"]
    lib_talib_dirs = [r"c:\ta-lib\c\lib"]

if not platform_supported:
    raise NotImplementedError(sys.platform)

ext_modules = []
for name in ['common', 'func', 'abstract']:
    ext = Extension(
        'talib.%s' % name,
        ['talib/%s.pyx' % name],
        include_dirs=include_dirs,
        library_dirs=lib_talib_dirs,
        libraries=[lib_talib_name]
    )
    ext_modules.append(ext)

setup(
    name = 'TA-Lib',
    version = '0.4.6-git',
    description = 'Python wrapper for TA-Lib',
    author = 'John Benediktsson',
    author_email = 'mrjbq7@gmail.com',
    url = 'http://github.com/mrjbq7/ta-lib',
    download_url = 'https://github.com/mrjbq7/ta-lib/archive/TA_Lib-0.4.5.zip',
    classifiers = [
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
    cmdclass = {'build_ext': build_ext}
)

from distutils.core import setup
from distutils.dist import Distribution
from distutils.extension import Extension

import os
import sys
import warnings


lib_talib_name = 'ta_lib'  # the underlying C library's name
ext_modules = []
cmdclass = {}

platform_supported = False
for prefix in ['darwin', 'linux', 'bsd']:
    if prefix in sys.platform:
        platform_supported = True
        include_dirs = [
            '/usr/include',
            '/usr/local/include',
            '/opt/include',
            '/opt/local/include',
        ]
        lib_talib_dirs = [
            '/usr/lib',
            '/usr/local/lib',
            '/usr/lib64',
            '/usr/local/lib64',
            '/opt/lib',
            '/opt/local/lib',
        ]
        break

if sys.platform == "win32":
    platform_supported = True
    lib_talib_name = 'ta_libc_cdr'
    include_dirs = [r"c:\ta-lib\c\include"]
    lib_talib_dirs = [r"c:\ta-lib\c\lib"]

# Do not require numpy or cython for just querying the package
if any('--' + opt in sys.argv for opt in Distribution.display_option_names +
       ['help-commands', 'help']) or sys.argv[1] == 'egg_info':
    pass
else:
    import numpy
    include_dirs.insert(0, numpy.get_include())

    from Cython.Distutils import build_ext
    cmdclass['build_ext'] = build_ext

if not platform_supported:
    raise NotImplementedError(sys.platform)

for lib_talib_dir in lib_talib_dirs:
    try:
        files = os.listdir(lib_talib_dir)
        if any(lib_talib_name in f for f in files):
            break
    except OSError:
        pass
else:
    warnings.warn('Cannot find ta-lib library, installation may fail.')

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
    version = '0.4.8',
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

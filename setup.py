from distutils.core import setup
from distutils.command.install import install
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import os
import sys
import warnings


lib_talib_name = 'ta_lib'  # the underlying C library's name

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
            os.getenv('VIRTUAL_ENV', './venv') + '/include',
        ]
        lib_talib_dirs = [
            '/usr/lib',
            '/usr/local/lib',
            '/usr/lib64',
            '/usr/local/lib64',
            '/opt/lib',
            '/opt/local/lib',
            os.getenv('VIRTUAL_ENV', './venv') + '/lib',
        ]
        break

if sys.platform == "win32":
    platform_supported = True
    lib_talib_name = 'ta_libc_cdr'
    include_dirs = [numpy.get_include(), r"c:\ta-lib\c\include"]
    lib_talib_dirs = [r"c:\ta-lib\c\lib"]

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

# Hack to install ta-lib library loading into virtualenv
def install_virtualenv_lib_loader():
    with open(os.getenv('VIRTUAL_ENV', './venv') + '/bin/activate', 'a') as f:
        f.write("\nexport LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH\n")
install_virtualenv_lib_loader()

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
    cmdclass = {'build_ext': build_ext}
)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import os
import sys

if sys.platform == "darwin":
    if os.path.exists("/opt/local/include/ta-lib"):
        include_talib_dir = "/opt/local/include"
        lib_talib_dir = "/opt/local/lib"
    else:
        include_talib_dir = "/usr/local/include/"
        lib_talib_dir = "/usr/local/lib/"

elif sys.platform == "linux2" or "freebsd" in sys.platform:
    include_talib_dir = "/usr/local/include/"
    lib_talib_dir = "/usr/local/lib/"

elif sys.platform == "win32":
    include_talib_dir = r"c:\msys\1.0\local\include"
    lib_talib_dir = r"c:\msys\1.0\local\lib"

else:
    raise NotImplementedError(sys.platform)

common_ext = Extension('talib.common_c', ['talib/common_c.pyx'],
    include_dirs=[numpy.get_include(), include_talib_dir],
    library_dirs=[lib_talib_dir],
    libraries=["ta_lib"]
)

func_ext = Extension("talib.func", ["talib/func.pyx"],
    include_dirs=[numpy.get_include(), include_talib_dir],
    library_dirs=[lib_talib_dir],
    libraries=["ta_lib"]
)

abstract_ext = Extension('talib.abstract', ['talib/abstract.pyx'],
    include_dirs=[numpy.get_include(), include_talib_dir],
    library_dirs=[lib_talib_dir],
    libraries=['ta_lib']
)

setup(
    name = 'TA-Lib',
    version = '0.4.2',
    description = 'Python wrapper for TA-Lib',
    author = 'John Benediktsson',
    author_email = 'mrjbq7@gmail.com',
    url = 'http://github.com/mrjbq7/ta-lib',
    download_url = 'https://github.com/mrjbq7/ta-lib/archive/TA_Lib-0.4.2.zip',
    classifiers = [
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD License",
    ],
    packages=['talib'],
    ext_modules=[common_ext, func_ext, abstract_ext],
    cmdclass = {'build_ext': build_ext}
)

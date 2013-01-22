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

ext = Extension("talib", ["talib.pyx"],
    include_dirs=[numpy.get_include(), include_talib_dir],
    library_dirs=[lib_talib_dir],
    libraries=["ta_lib"]
)

setup(
    name = 'TA-Lib',
    version = '0.4.1',
    description = 'Python wrapper for TA-Lib',
    author = 'John Benediktsson',
    author_email = 'mrjbq7@gmail.com',
    url = 'http://github.com/mrjbq7/ta-lib',
    download_url = 'https://github.com/mrjbq7/ta-lib/archive/0.4.1.zip',
    classifiers = [
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD License",
    ],
    ext_modules=[ext],
    cmdclass = {'build_ext': build_ext}
)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import sys

if sys.platform in ("linux2", "darwin"):
    include_talib_dir = "/usr/local/include/ta-lib/"
    lib_talib_dir = "/usr/local/lib/"
elif sys.platform == "win32":
    include_talib_dir = r"c:\msys\1.0\local\include\ta-lib"
    lib_talib_dir = r"c:\msys\1.0\local\lib"    
    
ext = Extension("talib", ["talib.pyx"],
    include_dirs=[numpy.get_include(), 
                  include_talib_dir],
    library_dirs=[lib_talib_dir],
    libraries=["ta_lib"]
)

setup(ext_modules=[ext],
    cmdclass = {'build_ext': build_ext})

#!/usr/bin/env python
import glob
from os import path

try:
    from setuptools import setup, Extension
    requires = {
        "install_requires": ["numpy"],
        "setup_requires": ["numpy"]
    }
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    requires = {"requires": ["numpy"]}

try:
    from Cython.Distutils import build_ext as cython_build_ext
    has_cython = True
except ImportError:
    has_cython = False


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


ext_modules = [
    Extension(
        'talib._ta_lib',
        [
            *glob.glob('talib/upstream/src/ta_common/*.c', recursive=True),
            *glob.glob('talib/upstream/src/ta_func/*.c', recursive=True),
            # We can't just glob ta_abstract, as ta_abstract includes things we
            # don't want like the Excel integration.
            'talib/upstream/src/ta_abstract/ta_group_idx.c',
            'talib/upstream/src/ta_abstract/ta_def_ui.c',
            'talib/upstream/src/ta_abstract/ta_abstract.c',
            'talib/upstream/src/ta_abstract/ta_func_api.c',
            'talib/upstream/src/ta_abstract/frames/ta_frame.c',
            *glob.glob('talib/upstream/src/ta_abstract/tables/*.c'),
            # This is our actual Python extension, everything else above is
            # just the upstream ta-lib dependency.
            'talib/_ta_lib.pyx' if has_cython else 'talib/_ta_lib.c'
        ],
        include_dirs=[
            'talib/upstream/include',
            'talib/upstream/src/ta_abstract',
            'talib/upstream/src/ta_abstract/frames',
            'talib/upstream/src/ta_common',
            'talib/upstream/src/ta_func'
        ]
    )
]

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='TA-Lib',
    version='0.4.24',
    description='Python wrapper for TA-Lib',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='John Benediktsson',
    author_email='mrjbq7@gmail.com',
    url='http://github.com/mrjbq7/ta-lib',
    download_url='https://github.com/mrjbq7/ta-lib/releases',
    license='BSD',
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Cython",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    packages=['talib'],
    ext_modules=ext_modules,
    cmdclass=LazyBuildExtCommandClass(),
    **requires
)

#!/usr/bin/env python

import ctypes
import ctypes.util
import functools
import hashlib
import multiprocessing
import os
import os.path
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
TALIB_C_VER = os.environ.get("TALIB_C_VER", "0.8.1")
FORCE_BUNDLED = os.environ.get("TA_FORCE_BUNDLED") == "1"
_CACHE_ROOT = Path(os.environ.get("TA_BUNDLE_CACHE") or ROOT / "build")
_CUSTOM_SRC = "TALIB_C_URL" in os.environ
TALIB_C_URL = os.environ.get(
    "TALIB_C_URL",
    f"https://github.com/TA-Lib/ta-lib/archive/refs/tags/v{TALIB_C_VER}.tar.gz",
)
_URL_TAG = hashlib.sha1(TALIB_C_URL.encode()).hexdigest()[:8]
BUNDLE_DIR = _CACHE_ROOT / f"ta-lib-c-{TALIB_C_VER}-{_URL_TAG}"


def _find_static(prefix):
    pat = "ta-lib-static.lib" if sys.platform == "win32" else "libta-lib.a"
    hits = sorted(prefix.rglob(pat))
    return hits[0] if hits else None


def build_bundled_talib():
    """Download, build and install TA-Lib C as a static lib. Returns (include_dir, static_lib)."""
    prefix = BUNDLE_DIR / "install"
    lib = _find_static(prefix) if prefix.exists() else None
    if lib:
        return str(prefix / "include"), str(lib)

    work = BUNDLE_DIR / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "talib.tar.gz"
        urllib.request.urlretrieve(TALIB_C_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(work)
    src = next(p for p in work.iterdir() if p.is_dir())
    bdir = work / "_build"

    args = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(bdir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DBUILD_DEV_TOOLS=OFF",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ]
    if sys.platform == "win32":
        args.append("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL")
        if sys.maxsize <= 2**32:
            args += ["-A", "Win32"]
    elif sys.platform == "darwin":
        archs = os.environ.get("ARCHFLAGS", "").replace("-arch", "").split()
        if archs:
            args.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}")
        dt = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if dt:
            args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={dt}")
    subprocess.check_call(args)
    subprocess.check_call(
        ["cmake", "--build", str(bdir), "--config", "Release", "--parallel"]
    )
    subprocess.check_call(["cmake", "--install", str(bdir), "--config", "Release"])

    lib = _find_static(prefix)
    if not lib:
        raise RuntimeError(f"static TA-Lib not found under {prefix}")
    return str(prefix / "include"), str(lib)


platform_supported = False

lib_talib_name = "ta-lib"

if any(s in sys.platform for s in ["darwin", "linux", "bsd", "sunos"]):
    platform_supported = True
    include_dirs = [
        "/usr/include",
        "/usr/local/include",
        "/opt/include",
        "/opt/local/include",
        "/opt/homebrew/include",
        "/opt/homebrew/opt/ta-lib/include",
    ]
    library_dirs = [
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib64",
        "/usr/local/lib64",
        "/opt/lib",
        "/opt/local/lib",
        "/opt/homebrew/lib",
        "/opt/homebrew/opt/ta-lib/lib",
    ]

elif sys.platform == "win32":
    platform_supported = True
    lib_talib_name = "ta-lib-static"
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

if "TA_INCLUDE_PATH" in os.environ:
    include_dirs = os.environ["TA_INCLUDE_PATH"].split(os.pathsep)

if "TA_LIBRARY_PATH" in os.environ:
    library_dirs = os.environ["TA_LIBRARY_PATH"].split(os.pathsep)

if not platform_supported:
    raise NotImplementedError(sys.platform)

# for path in library_dirs:
#    try:
#        files = os.listdir(path)
#        if any(lib_talib_name in f for f in files):
#            break
#    except OSError:
#        pass
# else:
#    warnings.warn('Cannot find ta-lib library, installation may fail.')


_EXPLICIT = "TA_LIBRARY_PATH" in os.environ or "TA_INCLUDE_PATH" in os.environ


def _parse_version(text):
    m = re.match(r"\s*(\d+(?:\.\d+)*)", text or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _system_talib_version():
    if sys.platform == "win32":
        return None
    pattern = "libta-lib*.dylib" if sys.platform == "darwin" else "libta-lib.so*"
    candidates = []
    for d in library_dirs:
        candidates += [str(p) for p in sorted(Path(d).glob(pattern))]
    found = ctypes.util.find_library("ta-lib")
    if found:
        candidates.append(found)
    for cand in candidates:
        try:
            fn = ctypes.CDLL(cand).TA_GetVersionString
            fn.restype = ctypes.c_char_p
            version = _parse_version(fn().decode("ascii", "ignore"))
        except (OSError, AttributeError):
            continue
        if version:
            return version
    return None


def _fmt(version):
    return ".".join(map(str, version))


@functools.cache
def resolve_talib():
    if FORCE_BUNDLED or _CUSTOM_SRC:
        return build_bundled_talib()

    required = _parse_version(TALIB_C_VER)
    found = _system_talib_version()

    if found is None and _EXPLICIT:
        return None
    if found is not None and found >= required:
        print(f"[talib] using system TA-Lib {_fmt(found)}", file=sys.stderr)
        return None

    why = "not found" if found is None else f"{_fmt(found)} < {TALIB_C_VER}"
    print(
        f"[talib] system TA-Lib {why}; building bundled {TALIB_C_VER} (static)",
        file=sys.stderr,
    )
    return build_bundled_talib()


# Get the Cython build_ext or fall back to setuptools build_ext
try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options as CythonOptions
    from setuptools.command.build_ext import build_ext

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

        if has_cython:
            CythonOptions.emit_code_comments = False
            cythonized = cythonize(
                self.extensions,
                nthreads=multiprocessing.cpu_count(),
                compiler_directives={
                    "embedsignature": True,
                    "freethreading_compatible": True,
                },
            )

            for ext, new in zip(self.extensions, cythonized):
                ext.sources = new.sources
                ext.depends = new.depends

        bundled = resolve_talib()

        for ext in self.extensions:
            if bundled:
                ext.include_dirs = [bundled[0]]
                ext.library_dirs = []
                ext.libraries = []
                ext.runtime_library_dirs = []
                ext.extra_objects = list(ext.extra_objects or []) + [bundled[1]]
            ext.include_dirs.append(numpy_incl)

        super().build_extensions()


cmdclass = {"build_ext": NumpyBuildExt}

ext_modules = [
    Extension(
        "talib._ta_lib",
        ["talib/_ta_lib.pyx" if has_cython else "talib/_ta_lib.c"],
        define_macros=[("Py_GIL_DISABLED", "1")]
        if sys.platform == "win32" and sysconfig.get_config_var("Py_GIL_DISABLED")
        else [],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[lib_talib_name],
        runtime_library_dirs=[] if sys.platform == "win32" else library_dirs,
    )
]


setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

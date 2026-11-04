# Installation

You can install from PyPI:

```
$ pip install TA-Lib
```

Binary wheels include the TA-Lib C library and do not require a compiler or a
separate TA-Lib C installation.

Or checkout the sources and build them yourself:

```
$ python -m pip install .
```

Source builds download and statically link the pinned TA-Lib C release. To use
an existing TA-Lib C installation instead, set ``TA_INCLUDE_PATH`` or
``TA_LIBRARY_PATH`` before building. To install the TA-Lib C library yourself,
follow its [installation directions](https://ta-lib.org/install/c/).

See the [README](https://github.com/ta-lib/ta-lib-python) for troubleshooting
install errors.

[Documentation Index](doc_index.md)
[FLOAT_RIGHTNext: Using the Function API](func.md)

# Installation

You can install from PyPI:

```
$ easy_install TA-Lib
```

Or checkout the sources and run ``setup.py`` yourself:

```
$ python setup.py install
```

Note: this requires that you have already installed the ``TA-Lib`` library
on your computer:
#### Mac OS X
```
$ brew install ta-lib
```

#### Windows
```
Download [ta-lib-0.4.0-msvc.zip](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip)
and unzip to ``C:\ta-lib``.
```

#### Linux
```
Download [ta-lib-0.4.0-src.tar.gz](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz)
$ untar and cd
$ ./configure --prefix=/usr
$ make -j5 && make -j5 # the first make fails when building in parallel
$ sudo make install
```

All supported download/install options are listed [here](http://ta-lib.org/hdr_dw.html).

## Troubleshooting

If you get build errors like this, it typically means that it can't find the
underlying ``TA-Lib`` library and needs to be installed:

```
func.c:256:28: fatal error: ta-lib/ta_libc.h: No such file or directory
compilation terminated.
```

If you install ``TA-Lib`` manually using ``make -jX``, the build will fail but
it's OK! Simply rerun ``make -jX`` followed by ``[sudo] make install`` and
everything will work as expected.

[Documentation Index](doc_index.html)
[FLOAT_RIGHTNext: Using the Function API](func.html)

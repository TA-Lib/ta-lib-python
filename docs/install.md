# Installation

You can install from PyPI:

```
$ pip install TA-Lib
```

Or checkout the sources and run ``setup.py`` yourself:

```
$ python setup.py install
```

### Troubleshooting Install Errors

```
func.c:256:28: fatal error: ta-lib/ta_libc.h: No such file or directory
compilation terminated.
```

If you get build errors like this, it typically means that it can't find the
underlying ``TA-Lib`` library and needs to be installed:

# Dependencies
To use TA-Lib for python, you need to have the [TA-Lib](http://ta-lib.org/hdr_dw.html)
already installed:

#### Mac OS X
```
$ brew install ta-lib
```

#### Windows
Download [ta-lib-0.4.0-msvc.zip](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip)
and unzip to ``C:\ta-lib``

#### Linux
Download [ta-lib-0.4.0-src.tar.gz](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz) and:
```
$ untar and cd
$ ./configure --prefix=/usr
$ make
$ sudo make install
```

> If you build ``TA-Lib`` using ``make -jX`` it will fail but that's OK!
> Simply rerun ``make -jX`` followed by ``[sudo] make install``.

[Documentation Index](doc_index.md)
[FLOAT_RIGHTNext: Using the Function API](func.md)

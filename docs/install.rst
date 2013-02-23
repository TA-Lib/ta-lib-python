
Installation
------------

You can install from PyPI:

::

    $ easy_install TA-Lib

Or checkout the sources and run ``setup.py`` yourself:

::

    $ python setup.py install

Note: this requires that you have already installed the ``TA-Lib``
library on your computer (you can `download
it <http://ta-lib.org/hdr_dw.html>`_ or use your computer's package
manager to install it, e.g., ``brew install ta-lib`` on Mac OS X). On
Windows, you can download the
`ta-lib-0.4.0-msvc.zip <http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip>`_
and unzip to ``C:\ta-lib``.

Troubleshooting
---------------

If you get build errors like this, it typically means that it can't find
the underlying ``TA-Lib`` library and needs to be installed:

::

    func.c:256:28: fatal error: ta-lib/ta_libc.h: No such file or directory
    compilation terminated.

If you install ``TA-Lib`` manually using ``make -jX``, the build will
fail but it's OK! Simply rerun ``make -jX`` followed by
``[sudo] make install`` and everything will work as expected.

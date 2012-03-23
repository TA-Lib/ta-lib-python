# TA-Lib

TA-Lib is widely used by trading software developers requiring to perform
technical analysis of financial market data.

The library is available in C/C++, Java, Perl, Python, and .NET.

    http://ta-lib.org/

Unfortunately, the included Python bindings use SWIG and are a little
difficult to install.  This project uses Cython and Numpy to efficiently and
cleanly bind to TA-Lib.

## Example

All of the following examples require:

```python
import numpy
import talib

input = numpy.random.random(100)
```

Calculate a moving average:

```python
i, n, output = talib.MA(input)
```

Calculating bollinger bands:

```python
i, n, upper, middle, lower = talib.BBANDS(input)
```

# TA-Lib

![Tests](https://github.com/mrjbq7/ta-lib/actions/workflows/tests.yml/badge.svg)

This is a Python wrapper for [TA-LIB](http://ta-lib.org) based on Cython
instead of SWIG. From the homepage:

> TA-Lib is widely used by trading software developers requiring to perform
> technical analysis of financial market data.
>
> * Includes 150+ indicators such as ADX, MACD, RSI, Stochastic, Bollinger
>   Bands, etc.
> * Candlestick pattern recognition
> * Open-source API for C/C++, Java, Perl, Python and 100% Managed .NET

The original Python bindings included with TA-Lib use
[SWIG](http://swig.org) which unfortunately are difficult to install and
aren't as efficient as they could be. Therefore this project uses
[Cython](https://cython.org) and [Numpy](https://numpy.org) to efficiently
and cleanly bind to TA-Lib -- producing results 2-4 times faster than the
SWIG interface.

In addition, this project also supports the use of the
[Polars](https://www.pola.rs) and [Pandas](https://pandas.pydata.org)
libraries.

## Installation

You can install from PyPI:

```
$ python3 -m pip install TA-Lib
```

Or checkout the sources and run ``setup.py`` yourself:

```
$ python setup.py install
```

It also appears possible to install via 
[Conda Forge](https://anaconda.org/conda-forge/ta-lib):

```
$ conda install -c conda-forge ta-lib
```

### Dependencies

To use TA-Lib for python, you need to have the
[TA-Lib](http://ta-lib.org/hdr_dw.html) already installed. You should
probably follow their installation directions for your platform, but some
suggestions are included below for reference.

##### Mac OS X

You can simply install using Homebrew:

```
$ brew install ta-lib
```

If you are using Apple Silicon, such as the M1 processors, and building mixed
architecture Homebrew projects, you might want to make sure it's being built
for your architecture:

```
$ arch -arm64 brew install ta-lib
```

And perhaps you can set these before installing with ``pip``:

```
$ export TA_INCLUDE_PATH="$(brew --prefix ta-lib)/include"
$ export TA_LIBRARY_PATH="$(brew --prefix ta-lib)/lib"
```

You might also find this helpful, particularly if you have tried several
different installations without success:

```
$ your-arm64-python -m pip install --no-cache-dir ta-lib
```

##### Windows

Download [ta-lib-0.4.0-msvc.zip](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip)
and unzip to ``C:\ta-lib``.

> This is a 32-bit binary release.  If you want to use 64-bit Python, you will
> need to build a 64-bit version of the library. Some unofficial (**and
> unsupported**) instructions for building on 64-bit Windows 10, here for
> reference:
>
> 1. Download and Unzip ``ta-lib-0.4.0-msvc.zip``
> 2. Move the Unzipped Folder ``ta-lib`` to ``C:\``
> 3. Download and Install Visual Studio Community 2015
>    * Remember to Select ``[Visual C++]`` Feature
> 4. Build TA-Lib Library
>    * From Windows Start Menu, Start ``[VS2015 x64 Native Tools Command
>      Prompt]``
>    * Move to ``C:\ta-lib\c\make\cdr\win32\msvc``
>    * Build the Library ``nmake``

You might also try these unofficial windows binaries for both 32-bit and
64-bit:

https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

##### Linux

Download [ta-lib-0.4.0-src.tar.gz](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz) and:

```
$ tar -xzf ta-lib-0.4.0-src.tar.gz
$ cd ta-lib/
$ ./configure --prefix=/usr
$ make
$ sudo make install
```

> If you build ``TA-Lib`` using ``make -jX`` it will fail but that's OK!
> Simply rerun ``make -jX`` followed by ``[sudo] make install``.

Note: if your directory path includes spaces, the installation will probably
fail with ``No such file or directory`` errors.

### Troubleshooting

If you get a warning that looks like this:

```
setup.py:79: UserWarning: Cannot find ta-lib library, installation may fail.
warnings.warn('Cannot find ta-lib library, installation may fail.')
```

This typically means ``setup.py`` can't find the underlying ``TA-Lib``
library, a dependency which needs to be installed.

---

If you installed the underlying ``TA-Lib`` library with a custom prefix
(e.g., with ``./configure --prefix=$PREFIX``), then when you go to install
this python wrapper you can specify additional search paths to find the
library and include files for the underlying ``TA-Lib`` library using the
``TA_LIBRARY_PATH`` and ``TA_INCLUDE_PATH`` environment variables:

```sh
$ export TA_LIBRARY_PATH=$PREFIX/lib
$ export TA_INCLUDE_PATH=$PREFIX/include
$ python setup.py install # or pip install ta-lib
```

---

Sometimes installation will produce build errors like this:

```
talib/_ta_lib.c:601:10: fatal error: ta-lib/ta_defs.h: No such file or directory
  601 | #include "ta-lib/ta_defs.h"
      |          ^~~~~~~~~~~~~~~~~~
compilation terminated.
```

or:

```
common.obj : error LNK2001: unresolved external symbol TA_SetUnstablePeriod
common.obj : error LNK2001: unresolved external symbol TA_Shutdown
common.obj : error LNK2001: unresolved external symbol TA_Initialize
common.obj : error LNK2001: unresolved external symbol TA_GetUnstablePeriod
common.obj : error LNK2001: unresolved external symbol TA_GetVersionString
```

This typically means that it can't find the underlying ``TA-Lib`` library, a
dependency which needs to be installed.  On Windows, this could be caused by
installing the 32-bit binary distribution of the underlying ``TA-Lib`` library,
but trying to use it with 64-bit Python.

---

Sometimes installation will fail with errors like this:

```
talib/common.c:8:22: fatal error: pyconfig.h: No such file or directory
 #include "pyconfig.h"
                      ^
compilation terminated.
error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```

This typically means that you need the Python headers, and should run
something like:

```
$ sudo apt-get install python3-dev
```

---

Sometimes building the underlying ``TA-Lib`` library has errors running
``make`` that look like this:

```
../libtool: line 1717: cd: .libs/libta_lib.lax/libta_abstract.a: No such file or directory
make[2]: *** [libta_lib.la] Error 1
make[1]: *** [all-recursive] Error 1
make: *** [all-recursive] Error 1
```

This might mean that the directory path to the underlying ``TA-Lib`` library
has spaces in the directory names.  Try putting it in a path that does not have
any spaces and trying again.

---

Sometimes you might get this error running ``setup.py``:

```
/usr/include/limits.h:26:10: fatal error: bits/libc-header-start.h: No such file or directory
#include <bits/libc-header-start.h>
         ^~~~~~~~~~~~~~~~~~~~~~~~~~
```

This is likely an issue with trying to compile for 32-bit platform but
without the appropriate headers.  You might find some success looking at the
first answer to [this question](https://stackoverflow.com/questions/54082459/fatal-error-bits-libc-header-start-h-no-such-file-or-directory-while-compili).

---

If you get an error on macOS like this:

```
code signature in <141BC883-189B-322C-AE90-CBF6B5206F67>
'python3.9/site-packages/talib/_ta_lib.cpython-39-darwin.so' not valid for
use in process: Trying to load an unsigned library)
```

You might look at [this question](https://stackoverflow.com/questions/69610572/how-can-i-solve-the-below-error-while-importing-nltk-package)
and use ``xcrun codesign`` to fix it.

---

If you wonder why ``STOCHRSI`` gives you different results than you expect,
probably you want ``STOCH`` applied to ``RSI``, which is a little different
than the ``STOCHRSI`` which is ``STOCHF`` applied to ``RSI``:

```python
>> >
from src import talib
>> > import numpy
>> > c = numpy.random.randn(100)

# this is the library function
>> > k, d = talib.STOCHRSI(c)

# this produces the same result, calling STOCHF
>> > rsi = talib.RSI(c)
>> > k, d = talib.STOCHF(rsi, rsi, rsi)

# you might want this instead, calling STOCH
>> > rsi = talib.RSI(c)
>> > k, d = talib.STOCH(rsi, rsi, rsi)
```

---

If the build appears to hang, you might be running on a VM with not enough
memory -- try 1 GB or 2 GB.

## Function API

Similar to TA-Lib, the Function API provides a lightweight wrapper of the
exposed TA-Lib indicators.

Each function returns an output array and have default values for their
parameters, unless specified as keyword arguments. Typically, these functions
will have an initial "lookback" period (a required number of observations
before an output is generated) set to ``NaN``.

For convenience, the Function API supports both ``numpy.ndarray`` and
``pandas.Series`` and ``polars.Series`` inputs.

All of the following examples use the Function API:

```python
import numpy
import talib

close = numpy.random.random(100)
```

Calculate a simple moving average of the close prices:

```python
output = talib.SMA(close)
```

Calculating bollinger bands, with triple exponential moving average:

```python
from talib import MA_Type

upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)
```

Calculating momentum of the close prices, with a time period of 5:

```python
output = talib.MOM(close, timeperiod=5)
```

##### NaN's

The underlying TA-Lib C library handles NaN's in a sometimes surprising manner
by typically propagating NaN's to the end of the output, for example:

```python
>>> c = numpy.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0])

>>> talib.SMA(c, 3)
array([nan, nan,  2., nan, nan, nan, nan])
```

You can compare that to a Pandas rolling mean, where their approach is to
output NaN until enough "lookback" values are observed to generate new outputs:

```python
>>> c = pandas.Series([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0])

>>> c.rolling(3).mean()
0    NaN
1    NaN
2    2.0
3    NaN
4    NaN
5    NaN
6    5.0
dtype: float64
```

## Abstract API

If you're already familiar with using the function API, you should feel right
at home using the Abstract API.

Every function takes a collection of named inputs, either a ``dict`` of
``numpy.ndarray`` or ``pandas.Series`` or ``polars.Series``, or a
``pandas.DataFrame`` or ``polars.DataFrame``. If a ``pandas.DataFrame`` or
``polars.DataFrame`` is provided, the output is returned as the same type
with named output columns.

For example, inputs could be provided for the typical "OHLCV" data:

```python
import numpy as np

# note that all ndarrays must be the same length!
inputs = {
    'open': np.random.random(100),
    'high': np.random.random(100),
    'low': np.random.random(100),
    'close': np.random.random(100),
    'volume': np.random.random(100)
}
```

Functions can either be imported directly or instantiated by name:

```python
from talib import abstract

# directly
SMA = abstract.SMA

# or by name
SMA = abstract.Function('sma')
```

From there, calling functions is basically the same as the function API:

```python
from talib.abstract import *

# uses close prices (default)
output = SMA(inputs, timeperiod=25)

# uses open prices
output = SMA(inputs, timeperiod=25, price='open')

# uses close prices (default)
upper, middle, lower = BBANDS(inputs, 20, 2, 2)

# uses high, low, close (default)
slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0) # uses high, low, close by default

# uses high, low, open instead
slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0, prices=['high', 'low', 'open'])
```

## Streaming API

An experimental Streaming API was added that allows users to compute the latest
value of an indicator.  This can be faster than using the Function API, for
example in an application that receives streaming data, and wants to know just
the most recent updated indicator value.

```python
import talib
from talib import stream

close = np.random.random(100)

# the Function API
output = talib.SMA(close)

# the Streaming API
latest = stream.SMA(close)

# the latest value is the same as the last output value
assert (output[-1] - latest) < 0.00001
```

## Supported Indicators and Functions

We can show all the TA functions supported by TA-Lib, either as a ``list`` or
as a ``dict`` sorted by group (e.g. "Overlap Studies", "Momentum Indicators",
etc):

```python
import talib

# list of functions
print talib.get_functions()

# dict of functions by group
print talib.get_function_groups()
```

### Indicator Groups

* Overlap Studies
* Momentum Indicators
* Volume Indicators
* Volatility Indicators
* Price Transform
* Cycle Indicators
* Pattern Recognition

##### Overlap Studies
```
BBANDS               Bollinger Bands
DEMA                 Double Exponential Moving Average
EMA                  Exponential Moving Average
HT_TRENDLINE         Hilbert Transform - Instantaneous Trendline
KAMA                 Kaufman Adaptive Moving Average
MA                   Moving average
MAMA                 MESA Adaptive Moving Average
MAVP                 Moving average with variable period
MIDPOINT             MidPoint over period
MIDPRICE             Midpoint Price over period
SAR                  Parabolic SAR
SAREXT               Parabolic SAR - Extended
SMA                  Simple Moving Average
T3                   Triple Exponential Moving Average (T3)
TEMA                 Triple Exponential Moving Average
TRIMA                Triangular Moving Average
WMA                  Weighted Moving Average
```

##### Momentum Indicators
```
ADX                  Average Directional Movement Index
ADXR                 Average Directional Movement Index Rating
APO                  Absolute Price Oscillator
AROON                Aroon
AROONOSC             Aroon Oscillator
BOP                  Balance Of Power
CCI                  Commodity Channel Index
CMO                  Chande Momentum Oscillator
DX                   Directional Movement Index
MACD                 Moving Average Convergence/Divergence
MACDEXT              MACD with controllable MA type
MACDFIX              Moving Average Convergence/Divergence Fix 12/26
MFI                  Money Flow Index
MINUS_DI             Minus Directional Indicator
MINUS_DM             Minus Directional Movement
MOM                  Momentum
PLUS_DI              Plus Directional Indicator
PLUS_DM              Plus Directional Movement
PPO                  Percentage Price Oscillator
ROC                  Rate of change : ((price/prevPrice)-1)*100
ROCP                 Rate of change Percentage: (price-prevPrice)/prevPrice
ROCR                 Rate of change ratio: (price/prevPrice)
ROCR100              Rate of change ratio 100 scale: (price/prevPrice)*100
RSI                  Relative Strength Index
STOCH                Stochastic
STOCHF               Stochastic Fast
STOCHRSI             Stochastic Relative Strength Index
TRIX                 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
ULTOSC               Ultimate Oscillator
WILLR                Williams' %R
```

##### Volume Indicators
```
AD                   Chaikin A/D Line
ADOSC                Chaikin A/D Oscillator
OBV                  On Balance Volume
```

##### Cycle Indicators
```
HT_DCPERIOD          Hilbert Transform - Dominant Cycle Period
HT_DCPHASE           Hilbert Transform - Dominant Cycle Phase
HT_PHASOR            Hilbert Transform - Phasor Components
HT_SINE              Hilbert Transform - SineWave
HT_TRENDMODE         Hilbert Transform - Trend vs Cycle Mode
```

##### Price Transform
```
AVGPRICE             Average Price
MEDPRICE             Median Price
TYPPRICE             Typical Price
WCLPRICE             Weighted Close Price
```

##### Volatility Indicators
```
ATR                  Average True Range
NATR                 Normalized Average True Range
TRANGE               True Range
```

##### Pattern Recognition
```
CDL2CROWS            Two Crows
CDL3BLACKCROWS       Three Black Crows
CDL3INSIDE           Three Inside Up/Down
CDL3LINESTRIKE       Three-Line Strike
CDL3OUTSIDE          Three Outside Up/Down
CDL3STARSINSOUTH     Three Stars In The South
CDL3WHITESOLDIERS    Three Advancing White Soldiers
CDLABANDONEDBABY     Abandoned Baby
CDLADVANCEBLOCK      Advance Block
CDLBELTHOLD          Belt-hold
CDLBREAKAWAY         Breakaway
CDLCLOSINGMARUBOZU   Closing Marubozu
CDLCONCEALBABYSWALL  Concealing Baby Swallow
CDLCOUNTERATTACK     Counterattack
CDLDARKCLOUDCOVER    Dark Cloud Cover
CDLDOJI              Doji
CDLDOJISTAR          Doji Star
CDLDRAGONFLYDOJI     Dragonfly Doji
CDLENGULFING         Engulfing Pattern
CDLEVENINGDOJISTAR   Evening Doji Star
CDLEVENINGSTAR       Evening Star
CDLGAPSIDESIDEWHITE  Up/Down-gap side-by-side white lines
CDLGRAVESTONEDOJI    Gravestone Doji
CDLHAMMER            Hammer
CDLHANGINGMAN        Hanging Man
CDLHARAMI            Harami Pattern
CDLHARAMICROSS       Harami Cross Pattern
CDLHIGHWAVE          High-Wave Candle
CDLHIKKAKE           Hikkake Pattern
CDLHIKKAKEMOD        Modified Hikkake Pattern
CDLHOMINGPIGEON      Homing Pigeon
CDLIDENTICAL3CROWS   Identical Three Crows
CDLINNECK            In-Neck Pattern
CDLINVERTEDHAMMER    Inverted Hammer
CDLKICKING           Kicking
CDLKICKINGBYLENGTH   Kicking - bull/bear determined by the longer marubozu
CDLLADDERBOTTOM      Ladder Bottom
CDLLONGLEGGEDDOJI    Long Legged Doji
CDLLONGLINE          Long Line Candle
CDLMARUBOZU          Marubozu
CDLMATCHINGLOW       Matching Low
CDLMATHOLD           Mat Hold
CDLMORNINGDOJISTAR   Morning Doji Star
CDLMORNINGSTAR       Morning Star
CDLONNECK            On-Neck Pattern
CDLPIERCING          Piercing Pattern
CDLRICKSHAWMAN       Rickshaw Man
CDLRISEFALL3METHODS  Rising/Falling Three Methods
CDLSEPARATINGLINES   Separating Lines
CDLSHOOTINGSTAR      Shooting Star
CDLSHORTLINE         Short Line Candle
CDLSPINNINGTOP       Spinning Top
CDLSTALLEDPATTERN    Stalled Pattern
CDLSTICKSANDWICH     Stick Sandwich
CDLTAKURI            Takuri (Dragonfly Doji with very long lower shadow)
CDLTASUKIGAP         Tasuki Gap
CDLTHRUSTING         Thrusting Pattern
CDLTRISTAR           Tristar Pattern
CDLUNIQUE3RIVER      Unique 3 River
CDLUPSIDEGAP2CROWS   Upside Gap Two Crows
CDLXSIDEGAP3METHODS  Upside/Downside Gap Three Methods
```

##### Statistic Functions
```
BETA                 Beta
CORREL               Pearson's Correlation Coefficient (r)
LINEARREG            Linear Regression
LINEARREG_ANGLE      Linear Regression Angle
LINEARREG_INTERCEPT  Linear Regression Intercept
LINEARREG_SLOPE      Linear Regression Slope
STDDEV               Standard Deviation
TSF                  Time Series Forecast
VAR                  Variance
```

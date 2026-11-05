# TA-Lib 📈

<!-- Badges -->
![Tests](https://github.com/ta-lib/ta-lib-python/actions/workflows/tests.yml/badge.svg)
[![Release](https://img.shields.io/github/v/release/ta-lib/ta-lib-python?label=Release)](https://github.com/ta-lib/ta-lib-python/releases)
[![PyPI](https://img.shields.io/pypi/v/TA-Lib?label=PyPI)](https://pypi.org/project/TA-Lib/)
[![Wheels](https://img.shields.io/pypi/wheel/TA-Lib?label=Wheels)](https://pypi.org/project/TA-Lib/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/TA-Lib?label=Python)](https://pypi.org/project/TA-Lib/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)

This is a Python wrapper for [TA-Lib](https://ta-lib.org). From the homepage:

> TA-Lib implements the standard technical analysis algorithms used across the
> industry, and is often the reference other libraries test their results
> against.
>
> * 200+ indicators: ADX, MACD, RSI, Stochastic, Bollinger Bands and more,
>   plus candlestick pattern recognition
> * Batch and streaming: compute a whole array, or update one bar at a time
>   without recomputing the history
> * Native in five languages: C/C++, Java, C#, Rust and TypeScript

`ta-lib-python` uses [Cython](https://cython.org) and [Numpy](https://numpy.org)
to efficiently and cleanly bind to the native C library.

In addition, this project also supports the use of the
[Polars](https://www.pola.rs) and [Pandas](https://pandas.pydata.org)
libraries.

## Versions 🗂️

Which TA-Lib C and `numpy` each `ta-lib-python` release line supports:

* `ta-lib-python` 0.4.x (supports `ta-lib` 0.4.x and `numpy` 1)
* `ta-lib-python` 0.5.x (supports `ta-lib` 0.4.x and `numpy` 2)
* `ta-lib-python` 0.6.x (supports `ta-lib` 0.6.x and `numpy` 2)
* `ta-lib-python` 0.7.x (supports `ta-lib` 0.7.x and `numpy` 2)
* `ta-lib-python` 0.8.x (supports `ta-lib` 0.8.x and `numpy` 2)

## Installation 💾

You can install from PyPI:

```shell
python -m pip install TA-Lib
```

Or from [Conda Forge](https://anaconda.org/conda-forge/ta-lib):

```shell
conda install -c conda-forge ta-lib
```

Binary wheels include the [TA-Lib](https://ta-lib.org) C library and do not
require a compiler or a separate TA-Lib C installation.

### Wheels ⚙️

For convenience, and starting with version 0.6.5, we now build binary wheels
for different operating systems, architectures, and Python versions using
GitHub Actions which include the underlying TA-Lib C library and are easy to
install. Starting with version 0.8.3, this library is statically linked into
the extension.

Supported platforms:

* Linux
  * x86_64
  * arm64
* macOS
  * x86_64
  * arm64
* Windows
  * x86_64
  * x86
  * arm64

Supported Python versions:

* 3.9
* 3.10
* 3.11
* 3.12
* 3.13
* 3.14

In the event that your operating system, architecture, or Python version is
not available as a binary wheel, it is fairly easy to install from source
using the instructions in [Building from source](#building-from-source-) below.

## Function API

Similar to TA-Lib, the Function API provides a lightweight wrapper of the
exposed TA-Lib indicators.

Each function returns an output array and has default values for their
parameters, unless specified as keyword arguments. Typically, these functions
will have an initial "lookback" period (a required number of observations
before an output is generated) set to `NaN`.

The lookback is function-specific and does not always match the value passed
to `timeperiod`. For example, `RSI(timeperiod=14)` needs 15 price observations
because the first RSI value depends on 14 price changes between consecutive
bars, not just 14 raw prices.

For convenience, the Function API supports both `numpy.ndarray` and
`pandas.Series` and `polars.Series` inputs.

All of the following examples use the Function API:

```python
import numpy as np
import talib

close = np.random.random(100)
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

Functions marked in the docs as having an unstable period start with that
extra unstable-period setting at `0`. In other words, `get_unstable_period()`
returns `0` until you explicitly change it with `set_unstable_period()`.

Instead of a count, `set_unstable_period()` also takes an Auto level. With
`talib.set_unstable_period('ALL', talib.UNSTABLE_AUTO_PREC_4)` each function
works out how many leading values to discard, so that the first 4 significant
digits of what it returns no longer depend on where your data starts;
`UNSTABLE_AUTO_PREC_8` targets 8. `get_unstable_period()` then returns the
level. See [Unstable Period](https://ta-lib.org/api/unstable-period/#auto).

### NaN's

The underlying TA-Lib C library handles NaN's in a sometimes surprising manner
by typically propagating NaN's to the end of the output, for example:

```python
>>> c = np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0])

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
upper, middle, lower = BBANDS(inputs, 20, 2.0, 2.0)

# uses high, low, close (default)
slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0) # uses high, low, close by default

# uses high, low, open instead
slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0, prices=['high', 'low', 'open'])
```

## Streaming API

The Streaming API keeps a handle per indicator instead of recomputing from the
whole array. Opening one costs a pass over the history; every bar after that is
O(1), and each value it produces is identical to the one the Function API
reports for that bar. On Apple Silicon macOS, a function using a transcendental
math function (SIN, EXP, LN, ...) may differ from it in the last bits.

```python
import talib
from talib import stream

# the Function API: the whole series, from the whole array
output = talib.SMA(close)

# the Streaming API: a handle, positioned at the end of the history
s = stream.SMA(close)
assert s.value == output[-1]

for price in feed:
    latest = s.update(price)      # one closed bar in, its value out

s.peek(forming)                   # what update would return, committing nothing
fork = s.copy()                   # an independent handle at the same bar
```

`stream.SMA` takes exactly the arguments `talib.SMA` takes. A single-output
function answers with a `float` (an `int` where the Function API returns an
integer array); a multi-output one with the same tuple the Function API returns:

```python
m = stream.MACD(close)
macd, macdsignal, macdhist = m.update(price)
m.value[2]                                    # the histogram, last bar

abstract.Function('MACD').output_names        # ['macd', 'macdsignal', 'macdhist']
```

Opening needs at least `lookback + 1` bars, which `abstract` knows, and a little
more where a function's seeding does -- so rather than computing the number,
treat a short history as "not yet":

```python
from talib import abstract

need = abstract.Function('RSI', timeperiod=14).lookback + 1   # 15, usually enough

try:
    s = stream.RSI(history, timeperiod=14)
except talib.InsufficientHistory:
    ...                                                       # collect more bars
```

Leading bars that are NaN in any input are not history. They are skipped, as the
Function API skips them, and do not count toward the warm-up. A NaN or an
infinity anywhere else in the history is undefined behaviour in TA-Lib C, and
for a few window functions a handle and the Function API do then disagree.

A bar that is not finite is likewise rejected: `update` raises and the handle is
left exactly as it was, neither its value nor its range moved. For a bar you
mean to skip rather than re-feed, say so with `advance()`, or two handles on one
feed drift a bar apart.

If you want the series over the history as well, one pass gives both:

```python
s, rsi = stream.RSI.open_and_fill(history, timeperiod=14)     # rsi == talib.RSI(history)
```

A handle also reports the range it has an output for, in the input series'
coordinates, and can be told about a bar it was not fed:

```python
s.out_range                       # (begidx, nbelement), as the Function API's output
s.advance()                       # count a skipped bar: the range moves, the value holds
```

A handle points into the TA-Lib C library, so it cannot be pickled or shared
with another process. Keep the history and re-open instead.

## Threads

TA-Lib supports both regular and free-threaded CPython. Native calculations,
including stream opening, updates, reads and cloning, release the GIL on
regular CPython. Python argument conversion, array allocation and result
construction still require Python thread state. Independent calculations run
concurrently; operations on the same stream handle use a per-handle mutex.

Inputs must remain unchanged during a call: do not write to or resize an input
array from another thread. Calls allocate their own output arrays.

Settings are process-wide. Set unstable periods, compatibility and candle
settings before starting calculations or opening streams. Changing settings
while a calculation is active or any stream is alive raises `RuntimeError`.
After calls finish and all streams (including copies) are released, settings
can be changed again. The settings gate protects lookback calculation, output
allocation and computation together, while allowing simultaneous readers.

The synchronization covers calls through this Python extension; other code
calling the same C library directly must observe TA-Lib's settings and lifecycle
contract too. Separate subinterpreters are not supported.

Free-threaded Python requires a matching free-threaded extension build and
compatible dependencies. Importing TA-Lib does not enable the GIL in that build.

## Building from source 🧩

Checkout the sources and build them yourself:

```shell
python -m pip install .
```

Or build the PyPI source distribution instead of installing a wheel:

```shell
python -m pip install --no-binary=TA-Lib TA-Lib
```

Starting with version 0.8.3, source builds download the pinned TA-Lib C
release, verify its checksum, and build and statically link it into the Python
extension. This requires network access, a C compiler and CMake. On Windows,
source builds require the Microsoft C++ Build Tools. CMake is installed
automatically when using pip's default build isolation.

To use an existing TA-Lib C installation instead, set ``TA_INCLUDE_PATH`` or
``TA_LIBRARY_PATH`` before building. Setting either variable disables the
bundled C build, including its download; if only one is set, the other uses
the standard search paths. These variables accept multiple directories
separated by ``os.pathsep`` (``:`` on Unix, ``;`` on Windows). They apply to
source builds, not prebuilt wheels.

To install the TA-Lib C library yourself, follow its
[installation directions](https://ta-lib.org/install/c/). On macOS, for
example, to link a source build against the Homebrew installation, set:

```shell
export TA_INCLUDE_PATH="$(brew --prefix ta-lib)/include"
export TA_LIBRARY_PATH="$(brew --prefix ta-lib)/lib"
```

> Some Conda Forge users have reported success installing the underlying TA-Lib C
> library using [the libta-lib package](https://anaconda.org/conda-forge/libta-lib):
>
> ``$ conda install -c conda-forge libta-lib``
>
> and then point the two variables above at its ``include`` and ``lib``.

## Troubleshooting 🛠️

Sometimes installation will produce build errors like this:

```shell
talib/_ta_lib.c:601:10: fatal error: ta-lib/ta_defs.h: No such file or directory
  601 | #include "ta-lib/ta_defs.h"
      |          ^~~~~~~~~~~~~~~~~~
compilation terminated.
```

or, on Windows, `LNK2001: unresolved external symbol TA_...` link errors.

When using ``TA_INCLUDE_PATH`` or ``TA_LIBRARY_PATH``, check that they point
to the headers and libraries of a compatible TA-Lib C installation. On Windows,
the library architecture must match Python (for example, a 64-bit Python needs
a 64-bit TA-Lib C library).

---

Sometimes installation will fail with errors like this:

```shell
talib/common.c:8:22: fatal error: pyconfig.h: No such file or directory
 #include "pyconfig.h"
                      ^
compilation terminated.
error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```

This typically means that you need the Python headers, and should run
something like:

```shell
sudo apt-get install python3-dev
```

---

If the build appears to hang, you might be running on a VM with not enough memory - try 1 GB or 2 GB.

## Supported Indicators and Functions 📋

We can show all the TA functions supported by TA-Lib, either as a `list` or
as a `dict` sorted by group (e.g. "Overlap Studies", "Momentum Indicators",
etc):

```python
import talib

# list of functions
for name in talib.get_functions():
    print(name)

# dict of functions by group
for group, names in talib.get_function_groups().items():
    print(group)
    for name in names:
        print(f"  {name}")
```

### Indicator Groups 🏷️

* Overlap Studies
* Momentum Indicators
* Volume Indicators
* Cycle Indicators
* Price Transform
* Volatility Indicators
* Pattern Recognition
* Statistic Functions
* Math Transform
* Math Operators

#### Overlap Studies

```text
ACCBANDS             Acceleration Bands
ALMA                 Arnaud Legoux Moving Average
BBANDS               Bollinger Bands
CKSP                 Chande Kroll Stop
DEMA                 Double Exponential Moving Average
DONCHIAN             Donchian Channels
EMA                  Exponential Moving Average
FRAMA                Fractal Adaptive Moving Average
HMA                  Hull Moving Average
HT_TRENDLINE         Hilbert Transform - Instantaneous Trendline
ICHIMOKU             Ichimoku Kinko Hyo
KAMA                 Kaufman Adaptive Moving Average
KC                   Keltner Channels
MA                   Moving average
MAMA                 MESA Adaptive Moving Average
MAVP                 Moving average with variable period
MCGD                 McGinley Dynamic
MIDPOINT             MidPoint over period
MIDPRICE             Midpoint Price over period
RMA                  Wilder's Smoothed Moving Average
SAR                  Parabolic SAR
SAREXT               Parabolic SAR - Extended
SMA                  Simple Moving Average
SUPERTREND           SuperTrend
SWAK_BUTTER          Swiss Army Knife - Butterworth Filter
SWAK_GAUSS           Swiss Army Knife - Gaussian Filter
T3                   Triple Exponential Moving Average (T3)
TEMA                 Triple Exponential Moving Average
TRIMA                Triangular Moving Average
VIDYA                Variable Index Dynamic Average
VWMA                 Volume Weighted Moving Average
WMA                  Weighted Moving Average
ZIGZAG               Zig Zag
ZLEMA                Zero-Lag Exponential Moving Average
```

#### Momentum Indicators

```text
AC                   Accelerator/Decelerator Oscillator
ADX                  Average Directional Movement Index
ADXR                 Average Directional Movement Index Rating
AO                   Awesome Oscillator
APO                  Absolute Price Oscillator
AROON                Aroon
AROONOSC             Aroon Oscillator
ASI                  Wilder Accumulative Swing Index
BOP                  Balance Of Power
CCI                  Commodity Channel Index
CG                   Center of Gravity Oscillator
CHOP                 Choppiness Index
CHOPTR               Choppiness Index (True Range Box)
CMO                  Chande Momentum Oscillator
CMOU                 Chande Momentum Oscillator (Unsmoothed)
COPPOCK              Coppock Curve
CRSI                 Connors Relative Strength Index
CTI                  Correlation Trend Indicator
DOSC                 Derivative Oscillator
DPO                  Detrended Price Oscillator
DX                   Directional Movement Index
ER                   Kaufman Efficiency Ratio
ERI                  Elder Ray Index (Bull Power / Bear Power)
FISHER               Fisher Transform
FOSC                 Forecast Oscillator
FRACTAL              Williams Fractal
IBS                  Internal Bar Strength
IMI                  Intraday Momentum Index
KDJ                  KDJ Stochastic
KST                  Know Sure Thing (Pring)
KSTEXT               Know Sure Thing with controllable MA type
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
PSO                  Premier Stochastic Oscillator
QSTICK               Qstick
ROC                  Rate of change : ((price/prevPrice)-1)*100
ROCP                 Rate of change Percentage: (price-prevPrice)/prevPrice
ROCR                 Rate of change ratio: (price/prevPrice)
ROCR100              Rate of change ratio 100 scale: (price/prevPrice)*100
RSI                  Relative Strength Index
SI                   Wilder Swing Index
SMI                  Stochastic Momentum Index
SQZMOM               Squeeze Momentum and Level
STC                  Schaff Trend Cycle
STOCH                Stochastic
STOCHF               Stochastic Fast
STOCHRSI             Stochastic Relative Strength Index
TRIX                 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
TSI                  True Strength Index
ULTOSC               Ultimate Oscillator
VHF                  Vertical Horizontal Filter
VORTEX               Vortex Indicator
WAD                  Williams' Accumulation/Distribution
WAVETREND            WaveTrend Oscillator
WILLR                Williams' %R
```

#### Volume Indicators

```text
AD                   Chaikin A/D Line
ADOSC                Chaikin A/D Oscillator
CMF                  Chaikin Money Flow
EFI                  Elder's Force Index
EMV                  Arms Ease of Movement
KVO                  Klinger Volume Oscillator
MARKETFI             Market Facilitation Index
NVI                  Negative Volume Index
OBV                  On Balance Volume
PVI                  Positive Volume Index
PVO                  Percentage Volume Oscillator
PVT                  Price Volume Trend
RVOL                 Relative Volume
VWAP                 Volume Weighted Average Price
```

#### Cycle Indicators

```text
HT_DCPERIOD          Hilbert Transform - Dominant Cycle Period
HT_DCPHASE           Hilbert Transform - Dominant Cycle Phase
HT_PHASOR            Hilbert Transform - Phasor Components
HT_SINE              Hilbert Transform - SineWave
HT_TRENDMODE         Hilbert Transform - Trend vs Cycle Mode
SWAK_2PHP            Swiss Army Knife - Two-Pole High-Pass Filter
SWAK_BP              Swiss Army Knife - Band-Pass Filter
SWAK_HP              Swiss Army Knife - High-Pass Filter
```

#### Price Transform

```text
AVGDEV               Average Deviation
AVGPRICE             Average Price
HA                   Heikin-Ashi Candles
MEDPRICE             Median Price
TYPPRICE             Typical Price
WCLPRICE             Weighted Close Price
```

#### Volatility Indicators

```text
ADR                  Average Day Range
ATR                  Average True Range
BBW                  Bollinger BandWidth
CVI                  Chaikin's Volatility
MASSI                Mass Index
NATR                 Normalized Average True Range
PERCENTB             Bollinger Bands %B
ROGERSSATCHELL       Rogers-Satchell Volatility
RVI                  Relative Volatility Index
RVIR                 Relative Volatility Index, refined high/low form
TRANGE               True Range
```

#### Pattern Recognition

```text
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

#### Statistic Functions

```text
BETA                 Beta
CORREL               Pearson's Correlation Coefficient (r)
KURTOSIS             Rolling Excess Kurtosis
LINEARREG            Linear Regression
LINEARREG_ANGLE      Linear Regression Angle
LINEARREG_INTERCEPT  Linear Regression Intercept
LINEARREG_SLOPE      Linear Regression Slope
MEDIAN               Rolling Median
PERCENTILE           Percentile (nearest rank)
PERCENTRANK          Percent Rank
STDDEV               Standard Deviation
TSF                  Time Series Forecast
VAR                  Variance
```

#### Math Transform

```text
ACOS                 Vector Trigonometric ACos
ASIN                 Vector Trigonometric ASin
ATAN                 Vector Trigonometric ATan
CEIL                 Vector Ceil
COS                  Vector Trigonometric Cos
COSH                 Vector Trigonometric Cosh
EXP                  Vector Arithmetic Exp
FLOOR                Vector Floor
LN                   Vector Log Natural
LOG10                Vector Log10
SIN                  Vector Trigonometric Sin
SINH                 Vector Trigonometric Sinh
SQRT                 Vector Square Root
TAN                  Vector Trigonometric Tan
TANH                 Vector Trigonometric Tanh
```

#### Math Operators

```text
ADD                  Vector Arithmetic Add
CUMSUM               Cumulative Sum
DIV                  Vector Arithmetic Div
MAX                  Highest value over a specified period
MAXINDEX             Index of highest value over a specified period
MIN                  Lowest value over a specified period
MININDEX             Index of lowest value over a specified period
MINMAX               Lowest and highest values over a specified period
MINMAXINDEX          Indexes of lowest and highest values over a specified period
MULT                 Vector Arithmetic Mult
SUB                  Vector Arithmetic Subtraction
SUM                  Summation
```

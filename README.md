# TA-Lib

From [TA-LIB's](http://ta-lib.org) homepage:

> TA-Lib is widely used by trading software developers requiring to perform
> technical analysis of financial market data.

> * Includes 200 indicators such as ADX, MACD, RSI, Stochastic, Bollinger
>   Bands, etc.
> * Candlestick pattern recognition
> * Open-source API for C/C++, Java, Perl, Python and 100% Managed .NET

Unfortunately, the included Python bindings use [SWIG](http://swig.org),
are a little difficult to install (particularly on Mac OS X), and aren't as
efficient as they could be. This project uses Cython and Numpy to efficiently
and cleanly bind to TA-Lib -- producing results 2-4 times faster than the SWIG
interface.

## Installation

You can install from PyPI:

```
$ easy_install TA-Lib
```

Or checkout the sources and run ``setup.py`` yourself:

```
$ python setup.py install
```

Note: this requires that you have already installed the ``TA-Lib`` library
on your computer (you can [download it](http://ta-lib.org/hdr_dw.html) or
use your computer's package manager to install it, e.g.,
``brew install ta-lib`` on Mac OS X).  On Windows, you can download the
[ta-lib-0.4.0-msvc.zip](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip)
and unzip to ``C:\ta-lib``.

## Troubleshooting

If you get build errors like this, it typically means that it can't find the
underlying ``TA-Lib`` library and needs to be installed:

```
func.c:256:28: fatal error: ta-lib/ta_libc.h: No such file or directory
compilation terminated.
```

If you get build errors compiling the underlying ``TA-Lib`` such as these:
```
mv -f .deps/gen_code-gen_code.Tpo .deps/gen_code-gen_code.Po
mv: cannot stat `.deps/gen_code-gen_code.Tpo': No such file or directory
make[3]: *** [gen_code-gen_code.o] Error 1/bin/bash ../../../libtool --tag=CC --mode=link gcc -g -O2 -L../../ta_common -L../../ta_abstract -L../../ta_func -o gen_code gen_code-gen_code.o -lta_common -lta_abstract_gc -lta_func -lm -lpthread -ldl
```
Simply rerunning ``make`` and then ``sudo make install`` seems to always do the trick.


## Function API Examples

Similar to TA-Lib, the function interface provides a lightweight wrapper of
the exposed TA-Lib indicators.

Each function returns an output array and have default values for their
parameters, unless specified as keyword arguments. Typically, these functions
will have an initial "lookback" period (a required number of observations
before an output is generated) set to ``NaN``.

All of the following examples use the function API:

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

## Abstract API Examples

TA-Lib also provides an abstract interface for calling functions. Our
wrapper for the abstract interface is somewhat different from the upstream
implementation. The abstract interface is designed to make the TA-Lib easily
introspectable and dynamically programmable. Of course Python allows for
these things too, but this helps do some of the heavy lifting for you, making
it much easier to for example add a TA-Lib indicator control panel to a
GUI charting program. It also unifies the interface for using and calling
functions making life easier on the developer.

If you're already familiar with using the function API, you should feel right
at home using the abstract API. Every function takes the same input, passed
as a dictionary of observed values:

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

From this input data, let's again calculate a simple moving average (SMA),
this time with the abstract interface:

```python
from talib.abstract import Function
output = Function('sma', input_arrays).outputs

# teaser:
output = Function('sma')(input_arrays, timeperiod=20, price='close')
upper, middle, lower = Function('bbands')(input_arrays, 20, 2, 2)
print Function('STOCH').info
```

You'll notice a few things are different. The function is now a class,
initialized with any supported function name (case insensitive) and optionally
``input_arrays``. To run the TA function with our input data, we access the
``outputs`` property. It wraps a method that ensures the results are always
valid so long as the ``input_arrays`` dict was already set. Speaking of which,
the SMA function only takes one input, and we gave it five!

Certain TA functions define which price series names they expect for input.
Others, like SMA, don't (we'll explain how to figure out which in a moment).
``Function`` will use the closing prices by default on TA functions that take
one undefined input, or the high and the low prices for functions taking two.
We can override the default like so:

```python
sma = Function('sma', input_arrays)
sma.set_function_args(timeperiod=10, price='open')
output = sma.outputs
```

This works by using keyword arguments. For functions with one undefined input,
the keyword is ``price``; for two they are ``price0`` and ``price1``. That's a
lot of typing; let's introduce some shortcuts:

```python
output = Function('sma').run(input_arrays)
output = Function('sma')(input_arrays, price='open')
```

The ``run()`` method is a shortcut to ``outputs`` that also optionally accepts
an ``input_arrays`` dict to use for calculating the function values. You can
also call the ``Function`` instance directly; this shortcut to ``outputs``
allows setting both ``input_arrays`` and/or any function parameters and
keywords. These methods make up all the ways you can call the TA function and
get its values.

``Function.outputs`` returns either a single ndarray or a list of ndarrays,
depending on how many outputs the TA function has. This information can be
found through ``Function.output_names`` or ``Function.info['outputs']``.

``Function.info`` is a very useful property. It returns a dict with almost
every detail of the current state of the ``Function`` instance:

```python
print Function('stoch').info
{
  'name': 'STOCH',
  'display_name': 'Stochastic',
  'group': 'Momentum Indicators',
  'input_names': OrderedDict([
    ('prices', ['high', 'low', 'close']),
  ]),
  'parameters': OrderedDict([
    ('fastk_period', 5),
    ('slowk_period', 3),
    ('slowk_matype', 0),
    ('slowd_period', 3),
    ('slowd_matype', 0),
  ]),
  'output_names': ['slowk', 'slowd'],
  'flags': None,
}
```

Take a look at the value of the ``input_names`` key. There's only one input
price variable, 'prices', and its value is a list of price series names.
This is one of those TA functions where TA-Lib defines which price series it
expects for input. Any time ``input_names`` is an OrderedDict with one key,
``prices``, and a list for a value, it means TA-Lib defined the expected price
series names. You can override these just the same as undefined inputs, just
make sure to use a list with the correct number of price series names! (it
varies across functions)

You can also use ``Function.input_names`` to get/set the price series names,
and ``Function.parameters`` to get/set the function parameters. Let's expand
on the other ways to set TA function arguments:

```python
from talib import MA_Type

output = Function('sma')(input_arrays, timeperiod=10, price='high')

upper, middle, lower = Function('bbands')(input_arrays, timeperiod=20, matype=MA_Type.EMA)

stoch = Function('stoch', input_arrays)
stoch.set_function_args(slowd_period=5)
slowk, slowd = stoch(15, fastd_period=5) # 15 == fastk_period specified positionally
```

``input_arrays`` must be passed as a positional argument (or left out
entirely). TA function parameters can be passed as positional or keyword
arguments. Input price series names must be passed as keyword arguments (or
left out entirely). In fact, the ``__call__`` method of ``Function`` simply
calls ``set_function_args()``.

For your convenience, we create ``Function`` wrappers for all of the available
TA-Lib functions:

```python
from talib.abstract import SMA, BBANDS, STOCH

output = SMA(input_arrays)

upper, middle, lower = BBANDS(input_arrays, timeperiod=20)

slowk, slowd = STOCH(input_arrays, fastk_period=15, fastd_period=5)
```

## Indicators

We can show all the TA functions supported by TA-Lib, either as a ``list`` or
as a ``dict`` sorted by group (e.g. "Overlap Studies", "Momentum Indicators",
etc):

```python
import talib

print talib.get_functions()
print talib.get_function_groups()
```

Here are some of the included indicators:

```
AD                  Chaikin A/D Line
ADOSC               Chaikin A/D Oscillator
ADX                 Average Directional Movement Index
ADXR                Average Directional Movement Index Rating
APO                 Absolute Price Oscillator
AROON               Aroon
AROONOSC            Aroon Oscillator
ATR                 Average True Range
AVGPRICE            Average Price
BBANDS              Bollinger Bands
BETA                Beta
BOP                 Balance Of Power
CCI                 Commodity Channel Index
CDL2CROWS           Two Crows
CDL3BLACKCROWS      Three Black Crows
CDL3INSIDE          Three Inside Up/Down
CDL3LINESTRIKE      Three-Line Strike 
CDL3OUTSIDE         Three Outside Up/Down
CDL3STARSINSOUTH    Three Stars In The South
CDL3WHITESOLDIERS   Three Advancing White Soldiers
CDLABANDONEDBABY    Abandoned Baby
CDLADVANCEBLOCK     Advance Block
CDLBELTHOLD         Belt-hold
CDLBREAKAWAY        Breakaway
CDLCLOSINGMARUBOZU  Closing Marubozu
CDLCONCEALBABYSWALL Concealing Baby Swallow
CDLCOUNTERATTACK    Counterattack
CDLDARKCLOUDCOVER   Dark Cloud Cover
CDLDOJI             Doji
CDLDOJISTAR         Doji Star
CDLDRAGONFLYDOJI    Dragonfly Doji
CDLENGULFING        Engulfing Pattern
CDLEVENINGDOJISTAR  Evening Doji Star
CDLEVENINGSTAR      Evening Star
CDLGAPSIDESIDEWHITE Up/Down-gap side-by-side white lines
CDLGRAVESTONEDOJI   Gravestone Doji
CDLHAMMER           Hammer
CDLHANGINGMAN       Hanging Man
CDLHARAMI           Harami Pattern
CDLHARAMICROSS      Harami Cross Pattern
CDLHIGHWAVE         High-Wave Candle
CDLHIKKAKE          Hikkake Pattern
CDLHIKKAKEMOD       Modified Hikkake Pattern
CDLHOMINGPIGEON     Homing Pigeon
CDLIDENTICAL3CROWS  Identical Three Crows
CDLINNECK           In-Neck Pattern
CDLINVERTEDHAMMER   Inverted Hammer
CDLKICKING          Kicking
CDLKICKINGBYLENGTH  Kicking - bull/bear determined by the longer marubozu
CDLLADDERBOTTOM     Ladder Bottom
CDLLONGLEGGEDDOJI   Long Legged Doji
CDLLONGLINE         Long Line Candle
CDLMARUBOZU         Marubozu
CDLMATCHINGLOW      Matching Low
CDLMATHOLD          Mat Hold
CDLMORNINGDOJISTAR  Morning Doji Star
CDLMORNINGSTAR      Morning Star
CDLONNECK           On-Neck Pattern
CDLPIERCING         Piercing Pattern
CDLRICKSHAWMAN      Rickshaw Man
CDLRISEFALL3METHODS Rising/Falling Three Methods
CDLSEPARATINGLINES  Separating Lines
CDLSHOOTINGSTAR     Shooting Star
CDLSHORTLINE        Short Line Candle
CDLSPINNINGTOP      Spinning Top
CDLSTALLEDPATTERN   Stalled Pattern
CDLSTICKSANDWICH    Stick Sandwich
CDLTAKURI           Takuri (Dragonfly Doji with very long lower shadow)
CDLTASUKIGAP        Tasuki Gap
CDLTHRUSTING        Thrusting Pattern
CDLTRISTAR          Tristar Pattern
CDLUNIQUE3RIVER     Unique 3 River
CDLUPSIDEGAP2CROWS  Upside Gap Two Crows
CDLXSIDEGAP3METHODS Upside/Downside Gap Three Methods
CMO                 Chande Momentum Oscillator
CORREL              Pearson's Correlation Coefficient (r)
DEMA                Double Exponential Moving Average
DX                  Directional Movement Index
EMA                 Exponential Moving Average
HT_DCPERIOD         Hilbert Transform - Dominant Cycle Period
HT_DCPHASE          Hilbert Transform - Dominant Cycle Phase
HT_PHASOR           Hilbert Transform - Phasor Components
HT_SINE             Hilbert Transform - SineWave
HT_TRENDLINE        Hilbert Transform - Instantaneous Trendline
HT_TRENDMODE        Hilbert Transform - Trend vs Cycle Mode
KAMA                Kaufman Adaptive Moving Average
LINEARREG           Linear Regression
LINEARREG_ANGLE     Linear Regression Angle
LINEARREG_INTERCEPT Linear Regression Intercept
LINEARREG_SLOPE     Linear Regression Slope
MA                  All Moving Average
MACD                Moving Average Convergence/Divergence
MACDEXT             MACD with controllable MA type
MACDFIX             Moving Average Convergence/Divergence Fix 12/26
MAMA                MESA Adaptive Moving Average
MAX                 Highest value over a specified period
MAXINDEX            Index of highest value over a specified period
MEDPRICE            Median Price
MFI                 Money Flow Index
MIDPOINT            MidPoint over period
MIDPRICE            Midpoint Price over period
MIN                 Lowest value over a specified period
MININDEX            Index of lowest value over a specified period
MINMAX              Lowest and highest values over a specified period
MINMAXINDEX         Indexes of lowest and highest values over a specified period
MINUS_DI            Minus Directional Indicator
MINUS_DM            Minus Directional Movement
MOM                 Momentum
NATR                Normalized Average True Range
OBV                 On Balance Volume
PLUS_DI             Plus Directional Indicator
PLUS_DM             Plus Directional Movement
PPO                 Percentage Price Oscillator
ROC                 Rate of change : ((price/prevPrice)-1)*100
ROCP                Rate of change Percentage: (price-prevPrice)/prevPrice
ROCR                Rate of change ratio: (price/prevPrice)
ROCR100             Rate of change ratio 100 scale: (price/prevPrice)*100
RSI                 Relative Strength Index
SAR                 Parabolic SAR
SAREXT              Parabolic SAR - Extended
SMA                 Simple Moving Average
STDDEV              Standard Deviation
STOCH               Stochastic
STOCHF              Stochastic Fast
STOCHRSI            Stochastic Relative Strength Index
SUM                 Summation
T3                  Triple Exponential Moving Average (T3)
TEMA                Triple Exponential Moving Average
TRANGE              True Range
TRIMA               Triangular Moving Average
TRIX                1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
TSF                 Time Series Forecast
TYPPRICE            Typical Price
ULTOSC              Ultimate Oscillator
VAR                 Variance
WCLPRICE            Weighted Close Price
WILLR               Williams' %R
WMA                 Weighted Moving Average
```

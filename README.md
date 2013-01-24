# TA-Lib

From [TA-LIB's](http://ta-lib.org) homepage:

> TA-Lib is widely used by trading software developers requiring to perform
> technical analysis of financial market data.

> * Includes 200 indicators such as ADX, MACD, RSI, Stochastic, Bollinger
>   Bands, etc.
> * Candlestick pattern recognition
> * Open-source API for C/C++, Java, Perl, Python and 100% Managed .NET

Unfortunately, the included Python bindings use SWIG, are a little difficult
to install (particularly on Mac OS X), and aren't as efficient as they could
be.  This project uses Cython and Numpy to efficiently and cleanly bind to
TA-Lib -- producing results 2-4 times faster than the SWIG interface.

## Installation

You can install from PyPI:

```
$ easy_install TA-Lib
```

Or checkout the sources and run ``setup.py`` yourself:

```
$ python setup.py install
```

Note: this requires that you have already installed the ``ta-lib`` library
on your computer (you can [download it](http://ta-lib.org/hdr_dw.html) or
use your computers package manager to install it, e.g.,
``brew install ta-lib`` on Mac OS X).

## Troubleshooting

If you get build errors like this, it typically means that it can't find the
underlying ``ta-lib`` library and needs to be installed:

```
func.c:256:28: fatal error: ta-lib/ta_libc.h: No such file or directory
compilation terminated.
```

If you get build errors compiling the underlying ``ta-lib`` library, simply
rerunning ``make`` and then ``sudo make install`` usually does the trick.

## Function API Examples

Similar to TA-Lib, the functions return an index into the input where the
output data begins and have default values for their parameters, unless
specified as keyword arguments.

All of the following examples will use these definitions:

```python
import numpy
import talib

data = numpy.random.random(100)
```

Calculate a simple moving average:

```python
output = talib.SMA(data)
```

Calculating bollinger bands, with triple exponential moving average:

```python
upper, middle, lower = talib.BBANDS(data, matype=talib.MA_T3)
```

Calculating momentum, with a time period of 5:

```python
output = talib.MOM(data, timeperiod=5)
```

## Abstract API Examples

TA-Lib also provides an abstract interface for calling functions. Our wrapper
for the abstract interface is somewhat different from the upstream implentation.
If you're already familiar with using the function API, you should feel right at
home using the abstract interface. To start, every function takes the same input:

```python
import numpy as np
# note that all ndarrays must be the same length!
input_arrays = { 'open': np.random.random(100),
                 'high': np.random.random(100),
                 'low': np.random.random(100),
                 'close': np.random.random(100),
                 'volume': np.random.random(100) }
```

From this input data, let's again calculate a SMA, this time with the abstract
interface:

```python
from talib.abstract import Function
output = Function('sma', input_arrays).get_outputs()

# teaser:
output = Function('sma')(input_arrays, timePeriod=20, price='open')
upper, middle, lower = Function('bbands')(input_arrays, 20, 2, 2)
print Function('STOCH').info
```

You'll notice a few things are different. The function is now a class,
initialized with any supported function name and optionally ``input_arrays``.
To run the TA function with our input data, we access the ``outputs`` property.
It wraps a method that ensures the results are always valid so long as the
``input_arrays`` dict was already set. Speaking of which, the SMA function only
takes one input, and we gave it five!

Certain TA functions define which price series names they expect for input.
Others, like SMA, don't (we'll explain how to figure out which in a moment).
``Function`` will use the closing prices by default on TA functions that take
one undefined input, or the high and the low prices for functions taking two.
We can override the default like so:

```python
sma = Function('sma', input_arrays)
sma.set_function_parameters(price='open')
output = sma.outputs
```

This works by using keyword arguments. For functions with one undefined input,
the keyword is 'price'; for two they are 'price0' and 'price1'. That's a lot of
typing; let's introduce some shortcuts:

```python
output = Function('sma').run(input_arrays)
output = Function('sma')(input_arrays, price='open')
```

The ``run()`` method is a shortcut to ``outputs`` that also optionally accepts
an ``input_arrays`` dict to use for calculating the function values. You can
also call the ``Function`` instance directly; this shortcut to ``outputs``
allows setting both ``input_arrays`` and/or any function parameters and keywords.
These methods make up all the ways you can call the TA function and get its values.

``Function`` returns either a single ndarray or a list of ndarrays, depending
on how many outputs the TA function has. This information can be found through
``Function.output_names`` or ``Function.info['outputs']``.

``Function.info`` is a very useful property. It returns a dict with almost every
detail of the current state of the ``Function`` instance:

```python
print Function('stoch').get_info()
{
  'name': 'STOCH',
  'display_name': 'Stochastic',
  'group': 'Momentum Indicators',
  'input_names': OrderedDict([
    ('prices', ['high', 'low', 'close']),
  ]),
  'parameters': OrderedDict([
    ('fastK_Period', 5),
    ('slowK_Period', 3),
    ('slowK_MAType', 0),
    ('slowD_Period', 3),
    ('slowD_MAType', 0),
  ]),
  'output_names': ['slowK', 'slowD'],
  'flags': None,
}
```

Take a look at the value of the 'inputs' key. There's only one input price
variable, 'prices', and its value is a list of price series names. This is one
of those TA functions where TA-Lib defines which price series it expects for
input. Any time 'inputs' is an OrderedDict with one key, 'prices', and a list
for a value, it means TA-Lib defined the expected price series names. You can
override these just the same as undefined inputs, just make sure to use a list
with the correct number of price series names! (it varies across functions)

You can also use ``Function.input_names`` to get/set the price series names, and
``Function.parameters`` to get/set the function parameters. Let's expand on the
other ways to set TA function arguments:

```python
from talib import MA_Type
output = Function('sma')(input_arrays, timePeriod=10, price='high')
upper, middle, lower = Function('bbands')(input_arrays, timePeriod=20, MAType=MA_Type.EMA)
stoch = Function('stoch', input_arrays)
stoch.set_function_parameters(slowD_Period=5)
slowK, slowD = stoch(15, fastD_Period=5) # 15 == faskK_Period specified positionally
```

``input_arrays`` must be passed as a positional argument (or left out entirely).
TA function parameters can be passed as positional or keyword arguments. Input
price series names must be passed as keyword arguments (or left out entirely).
In fact, the ``__call__`` method of ``Function`` simply calls ``set_function_args()``.

## Indicators

We can show all the TA functions supported by TA-Lib, either as a list or as a
dict sorted by group (eg Overlap Studies, Momentum Indicators, etc):

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

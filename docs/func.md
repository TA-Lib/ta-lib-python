# Function API Examples

Similar to TA-Lib, the function interface provides a lightweight wrapper of
the exposed TA-Lib indicators.

Each function returns an output array and have default values for their
parameters, unless specified as keyword arguments. Typically, these functions
will have an initial "lookback" period (a required number of observations
before an output is generated) set to ``NaN``.

The lookback is function-specific and may be larger than the `timeperiod`
parameter. For example, ``RSI(timeperiod=14)`` needs 15 price observations,
because the first RSI value is derived from 14 consecutive price changes.

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

Every `matype` argument (for example `STOCH`'s `slowk_matype` and
`slowd_matype`) takes an `MA_Type` value. The integer defaults shown in the
function docs are those values: `0` is `MA_Type.SMA`, and `MA_Type[n]` names
value `n`.

Calculating momentum of the close prices, with a time period of 5:

```python
output = talib.MOM(close, timeperiod=5)
```

Functions documented as having an unstable period start with that extra
unstable-period setting at ``0``. ``get_unstable_period()`` only returns a
non-zero value after you set one explicitly with ``set_unstable_period()``.

Instead of a count, ``set_unstable_period()`` also takes an Auto level. With
``talib.set_unstable_period('ALL', talib.UNSTABLE_AUTO_PREC_4)`` each function
works out how many leading values to discard, so that the first 4 significant
digits of what it returns no longer depend on where your data starts;
``UNSTABLE_AUTO_PREC_8`` targets 8. ``get_unstable_period()`` then returns the
level. See [Unstable Period](https://ta-lib.org/api/unstable-period/#auto).

Documentation for all functions:

* [Overlap Studies](func_groups/overlap_studies.md)
* [Momentum Indicators](func_groups/momentum_indicators.md)
* [Volume Indicators](func_groups/volume_indicators.md)
* [Volatility Indicators](func_groups/volatility_indicators.md)
* [Pattern Recognition](func_groups/pattern_recognition.md)
* [Cycle Indicators](func_groups/cycle_indicators.md)
* [Statistic Functions](func_groups/statistic_functions.md)
* [Price Transform](func_groups/price_transform.md)
* [Math Transform](func_groups/math_transform.md)
* [Math Operators](func_groups/math_operators.md)

[Documentation Index](doc_index.md)
[FLOAT_RIGHTNext: Using the Abstract API](abstract.md)

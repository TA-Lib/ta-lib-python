# Function API Examples

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

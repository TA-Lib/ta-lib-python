# Abstract API Quick Start

If you're already familiar with using the function API, you should feel right
at home using the abstract API. Every function takes the same input, passed
as a dictionary of Numpy arrays:

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
sma = abstract.SMA
sma = abstract.Function('sma')
```

From there, calling functions is basically the same as the function API:

```python
from talib.abstract import *
output = SMA(input_arrays, timeperiod=25) # calculate on close prices by default
output = SMA(input_arrays, timeperiod=25, price='open') # calculate on opens
upper, middle, lower = BBANDS(input_arrays, 20, 2, 2)
slowk, slowd = STOCH(input_arrays, 5, 3, 0, 3, 0) # uses high, low, close by default
slowk, slowd = STOCH(input_arrays, 5, 3, 0, 3, 0, prices=['high', 'low', 'open'])
```

## Advanced Usage

For more advanced use cases of TA-Lib, the Abstract API also offers much more
flexibility. You can even subclass ``abstract.Function`` and override
``set_input_arrays`` to customize the type of input data Function accepts
(e.g. a pandas DataFrame).

Details about every function can be accessed via the info property:

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
}

```
Or in human-readable format:
```python
help(STOCH)
str(STOCH)
```

Other useful properties of ``Function``:

```python
Function('x').function_flags
Function('x').input_names
Function('x').input_arrays
Function('x').parameters
Function('x').lookback
Function('x').output_names
Function('x').output_flags
Function('x').outputs
```

Aside from calling the function directly, Functions maintain state and will
remember their parameters/input_arrays after they've been set. You can set
parameters and recalculate with new input data using run():
```python
SMA.parameters = {'timeperiod': 15}
result1 = SMA.run(input_arrays1)
result2 = SMA.run(input_arrays2)

# Or set input_arrays and change the parameters:
SMA.input_arrays = input_arrays1
ma10 = SMA(timeperiod=10)
ma20 = SMA(20)
```

For more details, take a look at the
[code](https://github.com/mrjbq7/ta-lib/blob/master/talib/abstract.pyx#L46).

[Documentation Index](doc_index.md)

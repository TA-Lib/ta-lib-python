
Abstract API Examples
---------------------

TA-Lib also provides an abstract interface for calling functions. Our
wrapper for the abstract interface is somewhat different from the
upstream implementation. The abstract interface is designed to make the
TA-Lib easily introspectable and dynamically programmable. Of course
Python allows for these things too, but this helps do some of the heavy
lifting for you, making it much easier to for example add a TA-Lib
indicator control panel to a GUI charting program. It also unifies the
interface for using and calling functions making life easier on the
developer.

If you're already familiar with using the function API, you should feel
right at home using the abstract API. Every function takes the same
input, passed as a dictionary of observed values:

::

    import numpy as np
    # note that all ndarrays must be the same length!
    inputs = {
        'open': np.random.random(100),
        'high': np.random.random(100),
        'low': np.random.random(100),
        'close': np.random.random(100),
        'volume': np.random.random(100)
    }

From this input data, let's again calculate a simple moving average
(SMA), this time with the abstract interface:

::

    from talib.abstract import Function
    output = Function('sma', input_arrays).outputs

    # teaser:
    output = Function('sma')(input_arrays, timeperiod=20, price='close')
    upper, middle, lower = Function('bbands')(input_arrays, 20, 2, 2)
    print Function('STOCH').info

You'll notice a few things are different. The function is now a class,
initialized with any supported function name (case insensitive) and
optionally ``input_arrays``. To run the TA function with our input data,
we access the ``outputs`` property. It wraps a method that ensures the
results are always valid so long as the ``input_arrays`` dict was
already set. Speaking of which, the SMA function only takes one input,
and we gave it five!

Certain TA functions define which price series names they expect for
input. Others, like SMA, don't (we'll explain how to figure out which in
a moment). ``Function`` will use the closing prices by default on TA
functions that take one undefined input, or the high and the low prices
for functions taking two. We can override the default like so:

::

    sma = Function('sma', input_arrays)
    sma.set_function_args(timeperiod=10, price='open')
    output = sma.outputs

This works by using keyword arguments. For functions with one undefined
input, the keyword is ``price``; for two they are ``price0`` and
``price1``. That's a lot of typing; let's introduce some shortcuts:

::

    output = Function('sma').run(input_arrays)
    output = Function('sma')(input_arrays, price='open')

The ``run()`` method is a shortcut to ``outputs`` that also optionally
accepts an ``input_arrays`` dict to use for calculating the function
values. You can also call the ``Function`` instance directly; this
shortcut to ``outputs`` allows setting both ``input_arrays`` and/or any
function parameters and keywords. These methods make up all the ways you
can call the TA function and get its values.

``Function.outputs`` returns either a single ndarray or a list of
ndarrays, depending on how many outputs the TA function has. This
information can be found through ``Function.output_names`` or
``Function.info['outputs']``.

``Function.info`` is a very useful property. It returns a dict with
almost every detail of the current state of the ``Function`` instance:

::

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

Take a look at the value of the ``input_names`` key. There's only one
input price variable, 'prices', and its value is a list of price series
names. This is one of those TA functions where TA-Lib defines which
price series it expects for input. Any time ``input_names`` is an
OrderedDict with one key, ``prices``, and a list for a value, it means
TA-Lib defined the expected price series names. You can override these
just the same as undefined inputs, just make sure to use a list with the
correct number of price series names! (it varies across functions)

You can also use ``Function.input_names`` to get/set the price series
names, and ``Function.parameters`` to get/set the function parameters.
Let's expand on the other ways to set TA function arguments:

::

    from talib import MA_Type

    output = Function('sma')(input_arrays, timeperiod=10, price='high')

    upper, middle, lower = Function('bbands')(input_arrays, timeperiod=20, matype=MA_Type.EMA)

    stoch = Function('stoch', input_arrays)
    stoch.set_function_args(slowd_period=5)
    slowk, slowd = stoch(15, fastd_period=5) # 15 == fastk_period specified positionally

``input_arrays`` must be passed as a positional argument (or left out
entirely). TA function parameters can be passed as positional or keyword
arguments. Input price series names must be passed as keyword arguments
(or left out entirely). In fact, the ``__call__`` method of ``Function``
simply calls ``set_function_args()``.

For your convenience, we create ``Function`` wrappers for all of the
available TA-Lib functions:

::

    from talib.abstract import SMA, BBANDS, STOCH

    output = SMA(input_arrays)

    upper, middle, lower = BBANDS(input_arrays, timeperiod=20)

    slowk, slowd = STOCH(input_arrays, fastk_period=15, fastd_period=5)

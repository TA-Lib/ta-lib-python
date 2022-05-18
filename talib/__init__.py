
import atexit
from itertools import chain
from functools import wraps

# If polars is available, wrap talib functions so that they support
# polars.Series input
try:
    from polars import Series as _pl_Series
except ImportError:
    # polars not available, nothing to wrap
    _pl_Series = None

# If pandas is available, wrap talib functions so that they support
# pandas.Series input
try:
    from pandas import Series as _pd_Series
except ImportError:
    # pandas not available, nothing to wrap
    _pd_Series = None

if _pl_Series is not None or _pd_Series is not None:

    def _wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwds):

            if _pl_Series is not None:
                use_pl = any(isinstance(arg, _pl_Series) for arg in args) or \
                         any(isinstance(v, _pl_Series) for v in kwds.values())
            else:
                use_pl = False

            if _pd_Series is not None:
                use_pd = any(isinstance(arg, _pd_Series) for arg in args) or \
                         any(isinstance(v, _pd_Series) for v in kwds.values())
            else:
                use_pd = False

            if use_pl and use_pd:
                raise Exception("Cannot mix polars and pandas")

            # Use float64 values if polars or pandas, else use values as passed
            if use_pl:
                _args = [arg.to_numpy().astype(float) if isinstance(arg, _pl_Series) else
                         arg for arg in args]
                _kwds = {k: v.to_numpy().astype(float) if isinstance(v, _pl_Series) else
                            v for k, v in kwds.items()}

            elif use_pd:
                index = next(arg.index
                             for arg in chain(args, kwds.values())
                             if isinstance(arg, _pd_Series))

                _args = [arg.to_numpy().astype(float) if isinstance(arg, _pd_Series) else
                         arg for arg in args]
                _kwds = {k: v.to_numpy().astype(float) if isinstance(v, _pd_Series) else
                            v for k, v in kwds.items()}

            else:
                _args = args
                _kwds = kwds

            result = func(*_args, **_kwds)

            # check to see if we got a streaming result
            first_result = result[0] if isinstance(result, tuple) else result
            is_streaming_fn_result = not hasattr(first_result, '__len__')
            if is_streaming_fn_result:
                return result

            # Series was passed in, Series gets out
            if use_pl:
                if isinstance(result, tuple):
                    return tuple(_pl_Series(arr) for arr in result)
                else:
                    return _pl_Series(result)

            elif use_pd:
                if isinstance(result, tuple):
                    return tuple(_pd_Series(arr, index=index) for arr in result)
                else:
                    return _pd_Series(result, index=index)

            else:
                return result

        return wrapper
else:
    _wrapper = lambda x: x


from ._ta_lib import (
    _ta_initialize, _ta_shutdown, MA_Type, __ta_version__,
    _ta_set_unstable_period as set_unstable_period,
    _ta_get_unstable_period as get_unstable_period,
    _ta_set_compatibility as set_compatibility,
    _ta_get_compatibility as get_compatibility,
    __TA_FUNCTION_NAMES__
)

# import all the func and stream functions
from ._ta_lib import *

# wrap them for polars or pandas support
func = __import__("_ta_lib", globals(), locals(), __TA_FUNCTION_NAMES__, level=1)
for func_name in __TA_FUNCTION_NAMES__:
    wrapped_func = _wrapper(getattr(func, func_name))
    setattr(func, func_name, wrapped_func)
    globals()[func_name] = wrapped_func

stream_func_names = ['stream_%s' % fname for fname in __TA_FUNCTION_NAMES__]
stream = __import__("stream", globals(), locals(), stream_func_names, level=1)
for func_name, stream_func_name in zip(__TA_FUNCTION_NAMES__, stream_func_names):
    wrapped_func = _wrapper(getattr(stream, func_name))
    setattr(stream, func_name, wrapped_func)
    globals()[stream_func_name] = wrapped_func

__version__ = '0.4.24'

# In order to use this python library, talib (i.e. this __file__) will be
# imported at some point, either explicitly or indirectly via talib.func
# or talib.abstract. Here, we handle initializing and shutting down the
# underlying TA-Lib. Initialization happens on import, before any other TA-Lib
# functions are called. Finally, when the python process exits, we shutdown
# the underlying TA-Lib.

_ta_initialize()
atexit.register(_ta_shutdown)

__function_groups__ = {
    'Cycle Indicators': [
        'HT_DCPERIOD',
        'HT_DCPHASE',
        'HT_PHASOR',
        'HT_SINE',
        'HT_TRENDMODE',
        ],
    'Math Operators': [
        'ADD',
        'DIV',
        'MAX',
        'MAXINDEX',
        'MIN',
        'MININDEX',
        'MINMAX',
        'MINMAXINDEX',
        'MULT',
        'SUB',
        'SUM',
        ],
    'Math Transform': [
        'ACOS',
        'ASIN',
        'ATAN',
        'CEIL',
        'COS',
        'COSH',
        'EXP',
        'FLOOR',
        'LN',
        'LOG10',
        'SIN',
        'SINH',
        'SQRT',
        'TAN',
        'TANH',
        ],
    'Momentum Indicators': [
        'ADX',
        'ADXR',
        'APO',
        'AROON',
        'AROONOSC',
        'BOP',
        'CCI',
        'CMO',
        'DX',
        'MACD',
        'MACDEXT',
        'MACDFIX',
        'MFI',
        'MINUS_DI',
        'MINUS_DM',
        'MOM',
        'PLUS_DI',
        'PLUS_DM',
        'PPO',
        'ROC',
        'ROCP',
        'ROCR',
        'ROCR100',
        'RSI',
        'STOCH',
        'STOCHF',
        'STOCHRSI',
        'TRIX',
        'ULTOSC',
        'WILLR',
        ],
    'Overlap Studies': [
        'BBANDS',
        'DEMA',
        'EMA',
        'HT_TRENDLINE',
        'KAMA',
        'MA',
        'MAMA',
        'MAVP',
        'MIDPOINT',
        'MIDPRICE',
        'SAR',
        'SAREXT',
        'SMA',
        'T3',
        'TEMA',
        'TRIMA',
        'WMA',
        ],
    'Pattern Recognition': [
        'CDL2CROWS',
        'CDL3BLACKCROWS',
        'CDL3INSIDE',
        'CDL3LINESTRIKE',
        'CDL3OUTSIDE',
        'CDL3STARSINSOUTH',
        'CDL3WHITESOLDIERS',
        'CDLABANDONEDBABY',
        'CDLADVANCEBLOCK',
        'CDLBELTHOLD',
        'CDLBREAKAWAY',
        'CDLCLOSINGMARUBOZU',
        'CDLCONCEALBABYSWALL',
        'CDLCOUNTERATTACK',
        'CDLDARKCLOUDCOVER',
        'CDLDOJI',
        'CDLDOJISTAR',
        'CDLDRAGONFLYDOJI',
        'CDLENGULFING',
        'CDLEVENINGDOJISTAR',
        'CDLEVENINGSTAR',
        'CDLGAPSIDESIDEWHITE',
        'CDLGRAVESTONEDOJI',
        'CDLHAMMER',
        'CDLHANGINGMAN',
        'CDLHARAMI',
        'CDLHARAMICROSS',
        'CDLHIGHWAVE',
        'CDLHIKKAKE',
        'CDLHIKKAKEMOD',
        'CDLHOMINGPIGEON',
        'CDLIDENTICAL3CROWS',
        'CDLINNECK',
        'CDLINVERTEDHAMMER',
        'CDLKICKING',
        'CDLKICKINGBYLENGTH',
        'CDLLADDERBOTTOM',
        'CDLLONGLEGGEDDOJI',
        'CDLLONGLINE',
        'CDLMARUBOZU',
        'CDLMATCHINGLOW',
        'CDLMATHOLD',
        'CDLMORNINGDOJISTAR',
        'CDLMORNINGSTAR',
        'CDLONNECK',
        'CDLPIERCING',
        'CDLRICKSHAWMAN',
        'CDLRISEFALL3METHODS',
        'CDLSEPARATINGLINES',
        'CDLSHOOTINGSTAR',
        'CDLSHORTLINE',
        'CDLSPINNINGTOP',
        'CDLSTALLEDPATTERN',
        'CDLSTICKSANDWICH',
        'CDLTAKURI',
        'CDLTASUKIGAP',
        'CDLTHRUSTING',
        'CDLTRISTAR',
        'CDLUNIQUE3RIVER',
        'CDLUPSIDEGAP2CROWS',
        'CDLXSIDEGAP3METHODS',
        ],
    'Price Transform': [
        'AVGPRICE',
        'MEDPRICE',
        'TYPPRICE',
        'WCLPRICE',
        ],
    'Statistic Functions': [
        'BETA',
        'CORREL',
        'LINEARREG',
        'LINEARREG_ANGLE',
        'LINEARREG_INTERCEPT',
        'LINEARREG_SLOPE',
        'STDDEV',
        'TSF',
        'VAR',
        ],
    'Volatility Indicators': [
        'ATR',
        'NATR',
        'TRANGE',
        ],
    'Volume Indicators': [
        'AD',
        'ADOSC',
        'OBV'
        ],
    }

def get_functions():
    """
    Returns a list of all the functions supported by TALIB
    """
    ret = []
    for group in __function_groups__:
        ret.extend(__function_groups__[group])
    return ret

def get_function_groups():
    """
    Returns a dict with keys of function-group names and values of lists
    of function names ie {'group_names': ['function_names']}
    """
    return __function_groups__.copy()

__all__ = ['get_functions', 'get_function_groups'] + __TA_FUNCTION_NAMES__ + ["stream_%s" % name for name in __TA_FUNCTION_NAMES__]

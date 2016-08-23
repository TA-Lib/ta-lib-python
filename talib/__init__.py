
import atexit

from ._ta_lib import (
    _ta_initialize, _ta_shutdown, MA_Type, __ta_version__,
    _ta_set_unstable_period as set_unstable_period,
    _ta_get_unstable_period as get_unstable_period,
    __TA_FUNCTION_NAMES__
)

func = __import__("_ta_lib", globals(), locals(), __TA_FUNCTION_NAMES__, level=1)
for func_name in __TA_FUNCTION_NAMES__:
    globals()[func_name] = getattr(func, func_name)

__version__ = '0.4.10'

# In order to use this python library, talib (i.e. this __file__) will be
# imported at some point, either explicitly or indirectly via talib.func
# or talib.abstract. Here, we handle initalizing and shutting down the
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

__all__ = ['get_functions', 'get_function_groups']

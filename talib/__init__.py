
import atexit

from . import common
from . import abstract
from .common import MA_Type, __ta_version__
from .func import *

__version__ = '0.4.8'

# In order to use this python library, talib (i.e. this __file__) will be
# imported at some point, either explicitly or indirectly via talib.func
# or talib.abstract. Here, we handle initalizing and shutting down the
# underlying TA-Lib. Initialization happens on import, before any other TA-Lib
# functions are called. Finally, when the python process exits, we shutdown
# the underlying TA-Lib.

common._ta_initialize()
atexit.register(common._ta_shutdown)

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

def get_functions_df():
    """
    Returns a Pandas DataFrame of all the functions supported by TALIB
    """
    import pandas as pd
    lst_info = []
    for f in get_functions():
        absf = abstract.Function(f)
        lst_info.append(absf.info)
    df_absf = pd.DataFrame(lst_info)
    df_absf = df_absf.set_index('name')
    return(df_absf)

def get_functions_grouped_df():
    """
    Returns a Pandas DataFrame of all the functions supported by TALIB grouped by "group"
    """
    df_absf = get_functions_df()
    df_grp = df_absf.reset_index().set_index(['group', 'name']).sortlevel(0)
    return(df_grp)

__all__ = ['get_functions', 'get_function_groups', 'get_functions_df', 'get_functions_grouped_df']

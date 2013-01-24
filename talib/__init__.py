import atexit

from talib.func import *
from talib import abstract
from talib import common_c


class MA_Type(object):
    SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3 = range(9)

    def __init__(self):
        self._lookup = {
            MA_Type.SMA: 'Simple Moving Average',
            MA_Type.EMA: 'Exponential Moving Average',
            MA_Type.WMA: 'Weighted Moving Average',
            MA_Type.DEMA: 'Double Exponential Moving Average',
            MA_Type.TEMA: 'Triple Exponential Moving Average',
            MA_Type.TRIMA: 'Triangular Moving Average',
            MA_Type.KAMA: 'Kaufman Adaptive Moving Average',
            MA_Type.MAMA: 'MESA Adaptive Moving Average',
            MA_Type.T3: 'Triple Generalized Double Exponential Moving Average',
            }

    def __getitem__(self, type_):
        return self._lookup[type_]

MA_Type = MA_Type()


def get_functions():
    ''' Returns a list of all the functions supported by TALIB
    '''
    ret = []
    for group in abstract._ta_getGroupTable():
        ret.extend(abstract._ta_getFuncTable(group))
    return ret

def get_function_groups():
    ''' Returns a dict with kyes of function-group names and values of lists
    of function names ie {'group_names': ['function_names']}
    '''
    d = {}
    for group in abstract._ta_getGroupTable():
        d[group] = abstract._ta_getFuncTable(group)
    return d


'''
In order to use this python library, talib (ie this __file__) will be imported
at some point, either explicitly or indirectly via talib.func or talib.abstract.
Here, we handle initalizing and shutting down the underlying TA-Lib.
Initialization happens on import, before any other TA-Lib functions are called.
Finally, when the python process exits, we shutdown the underlying TA-Lib.
'''
common_c._ta_initialize()
atexit.register(common_c._ta_shutdown)

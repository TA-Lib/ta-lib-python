import atexit

from talib.func import *
from talib.abstract import _ta_getFuncTable
from talib.abstract import _ta_getGroupTable
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


def initialize():
    ''' Initializes the TALIB library
    '''
    common_c._ta_initialize()

def shutdown():
    ''' Shuts down the TALIB library
    '''
    common_c._ta_shutdown()

def get_functions():
    ''' Returns a list of all the functions supported by TALIB
    '''
    ret = []
    for group in _ta_getGroupTable():
        ret.extend(_ta_getFuncTable(group))
    return ret

def get_function_groups():
    ''' Returns a dict with kyes of function-group names and values of lists
    of function names ie {'group_names': ['function_names']}
    '''
    d = {}
    for group in _ta_getGroupTable():
        d[group] = _ta_getFuncTable(group)
    return d

initialize()
atexit.register(shutdown)

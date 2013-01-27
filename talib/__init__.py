
__version__ = '0.4.3-git'

import atexit

from . import common_c
from . import abstract
from .common_c import MA_Type
from .func import *

'''
In order to use this python library, talib (ie this __file__) will be imported
at some point, either explicitly or indirectly via talib.func or talib.abstract.
Here, we handle initalizing and shutting down the underlying TA-Lib.
Initialization happens on import, before any other TA-Lib functions are called.
Finally, when the python process exits, we shutdown the underlying TA-Lib.
'''
common_c._ta_initialize()
atexit.register(common_c._ta_shutdown)

def get_functions():
    '''
    Returns a list of all the functions supported by TALIB
    '''
    return abstract._ta_get_functions()

def get_function_groups():
    '''
    Returns a dict with kyes of function-group names and values of lists
    of function names ie {'group_names': ['function_names']}
    '''
    return abstract._ta_get_function_groups()



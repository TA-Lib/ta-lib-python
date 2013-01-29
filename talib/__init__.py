
import atexit

from . import common
from . import abstract
from .common import MA_Type, __ta_version__
from .func import *

__version__ = '0.4.3'

# In order to use this python library, talib (ie this __file__) will be
# imported at some point, either explicitly or indirectly via talib.func
# or talib.abstract. Here, we handle initalizing and shutting down the
# underlying TA-Lib. Initialization happens on import, before any other TA-Lib
# functions are called. Finally, when the python process exits, we shutdown
# the underlying TA-Lib.
common._ta_initialize()
atexit.register(common._ta_shutdown)

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



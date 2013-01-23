
import os
import re
import sys

import talib
from talib import abstract

# FIXME: initialize once, then shutdown at the end, rather than each call?
# FIXME: should we pass startIdx and endIdx into function?
# FIXME: don't return number of elements since it always equals allocation?

functions = []
include_paths = ['/usr/include', '/usr/local/include']
ta_func_header = None
for path in include_paths:
    if os.path.exists(path + '/ta-lib/ta_func.h'):
        ta_func_header = path + '/ta-lib/ta_func.h'
        break
if not ta_func_header:
    print >> sys.stderr, 'Error: ta-lib/ta_func.h not found'
    sys.exit(1)
with open(ta_func_header) as f:
    tmp = []
    for line in f:
        line = line.strip()
        if tmp or \
            line.startswith('TA_RetCode TA_') or \
            line.startswith('int TA_'):
            line = re.sub('/\*[^\*]+\*/', '', line) # strip comments
            tmp.append(line)
            if not line:
                s = ' '.join(tmp)
                s = re.sub('\s+', ' ', s)
                functions.append(s)
                tmp = []

# strip "float" functions
functions = [s for s in functions if not s.startswith('TA_RetCode TA_S_')]

# strip non-indicators
functions = [s for s in functions if not s.startswith('TA_RetCode TA_Set')]
functions = [s for s in functions if not s.startswith('TA_RetCode TA_Restore')]

# print headers
print """
from talib import utils
from numpy import nan
from cython import boundscheck, wraparound
cimport numpy as np

ctypedef np.double_t double_t
ctypedef np.int32_t int32_t

ctypedef int TA_RetCode
ctypedef int TA_MAType

cdef double NaN = nan

cdef extern from "math.h":
    bint isnan(double x)

cdef extern from "numpy/arrayobject.h":
    int PyArray_TYPE(np.ndarray)
    object PyArray_EMPTY(int, np.npy_intp*, int, int)
    int PyArray_FLAGS(np.ndarray)
    object PyArray_GETCONTIGUOUS(np.ndarray)

np.import_array() # Initialize the NumPy C API

# extract the needed part of ta_libc.h that I will use in the interface
cdef extern from "ta-lib/ta_libc.h":
    enum: TA_SUCCESS
    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()
    char *TA_GetVersionString()"""

# ! can't use const in function declaration (cython 0.12 restriction)
# just removing them does the trick
for f in functions:
    f = f.replace('const', '')
    f = f.replace(';', '')
    f = f.replace('void', '')
    f = f.strip()
    print '    %s' % f
print

print """
__version__ = TA_GetVersionString()
"""

print """
def initialize():
    ''' Initializes the TALIB library
    '''
    ret_code = TA_Initialize()
    utils._check_success('TA_Initialize', ret_code)
    return ret_code

def shutdown():
    ''' Shuts down the TALIB library
    '''
    ret_code = TA_Shutdown()
    utils._check_success('TA_Shutdown', ret_code)
    return ret_code
"""

# cleanup variable names to make them more pythonic
def cleanup(name):
    if name.startswith('in'):
        return name[2:].lower()
    elif name.startswith('optIn'):
        return name[5:].lower()
    else:
        return name.lower()

descriptions = {
    "AD"                 : "Chaikin A/D Line",
    "ADOSC"              : "Chaikin A/D Oscillator",
    "ADX"                : "Average Directional Movement Index",
    "ADXR"               : "Average Directional Movement Index Rating",
    "APO"                : "Absolute Price Oscillator",
    "AROON"              : "Aroon",
    "AROONOSC"           : "Aroon Oscillator",
    "ATR"                : "Average True Range",
    "AVGPRICE"           : "Average Price",
    "BBANDS"             : "Bollinger Bands",
    "BETA"               : "Beta",
    "BOP"                : "Balance Of Power",
    "CCI"                : "Commodity Channel Index",
    "CDL2CROWS"          : "Two Crows",
    "CDL3BLACKCROWS"     : "Three Black Crows",
    "CDL3INSIDE"         : "Three Inside Up/Down",
    "CDL3LINESTRIKE"     : "Three-Line Strike ",
    "CDL3OUTSIDE"        : "Three Outside Up/Down",
    "CDL3STARSINSOUTH"   : "Three Stars In The South",
    "CDL3WHITESOLDIERS"  : "Three Advancing White Soldiers",
    "CDLABANDONEDBABY"   : "Abandoned Baby",
    "CDLADVANCEBLOCK"    : "Advance Block",
    "CDLBELTHOLD"        : "Belt-hold",
    "CDLBREAKAWAY"       : "Breakaway",
    "CDLCLOSINGMARUBOZU" : "Closing Marubozu",
    "CDLCONCEALBABYSWALL": "Concealing Baby Swallow",
    "CDLCOUNTERATTACK"   : "Counterattack",
    "CDLDARKCLOUDCOVER"  : "Dark Cloud Cover",
    "CDLDOJI"            : "Doji",
    "CDLDOJISTAR"        : "Doji Star",
    "CDLDRAGONFLYDOJI"   : "Dragonfly Doji",
    "CDLENGULFING"       : "Engulfing Pattern",
    "CDLEVENINGDOJISTAR" : "Evening Doji Star",
    "CDLEVENINGSTAR"     : "Evening Star",
    "CDLGAPSIDESIDEWHITE": "Up/Down-gap side-by-side white lines",
    "CDLGRAVESTONEDOJI"  : "Gravestone Doji",
    "CDLHAMMER"          : "Hammer",
    "CDLHANGINGMAN"      : "Hanging Man",
    "CDLHARAMI"          : "Harami Pattern",
    "CDLHARAMICROSS"     : "Harami Cross Pattern",
    "CDLHIGHWAVE"        : "High-Wave Candle",
    "CDLHIKKAKE"         : "Hikkake Pattern",
    "CDLHIKKAKEMOD"      : "Modified Hikkake Pattern",
    "CDLHOMINGPIGEON"    : "Homing Pigeon",
    "CDLIDENTICAL3CROWS" : "Identical Three Crows",
    "CDLINNECK"          : "In-Neck Pattern",
    "CDLINVERTEDHAMMER"  : "Inverted Hammer",
    "CDLKICKING"         : "Kicking",
    "CDLKICKINGBYLENGTH" : "Kicking - bull/bear determined by the longer marubozu",
    "CDLLADDERBOTTOM"    : "Ladder Bottom",
    "CDLLONGLEGGEDDOJI"  : "Long Legged Doji",
    "CDLLONGLINE"        : "Long Line Candle",
    "CDLMARUBOZU"        : "Marubozu",
    "CDLMATCHINGLOW"     : "Matching Low",
    "CDLMATHOLD"         : "Mat Hold",
    "CDLMORNINGDOJISTAR" : "Morning Doji Star",
    "CDLMORNINGSTAR"     : "Morning Star",
    "CDLONNECK"          : "On-Neck Pattern",
    "CDLPIERCING"        : "Piercing Pattern",
    "CDLRICKSHAWMAN"     : "Rickshaw Man",
    "CDLRISEFALL3METHODS": "Rising/Falling Three Methods",
    "CDLSEPARATINGLINES" : "Separating Lines",
    "CDLSHOOTINGSTAR"    : "Shooting Star",
    "CDLSHORTLINE"       : "Short Line Candle",
    "CDLSPINNINGTOP"     : "Spinning Top",
    "CDLSTALLEDPATTERN"  : "Stalled Pattern",
    "CDLSTICKSANDWICH"   : "Stick Sandwich",
    "CDLTAKURI"          : "Takuri (Dragonfly Doji with very long lower shadow)",
    "CDLTASUKIGAP"       : "Tasuki Gap",
    "CDLTHRUSTING"       : "Thrusting Pattern",
    "CDLTRISTAR"         : "Tristar Pattern",
    "CDLUNIQUE3RIVER"    : "Unique 3 River",
    "CDLUPSIDEGAP2CROWS" : "Upside Gap Two Crows",
    "CDLXSIDEGAP3METHODS": "Upside/Downside Gap Three Methods",
    "CMO"                : "Chande Momentum Oscillator",
    "CORREL"             : "Pearson's Correlation Coefficient (r)",
    "DEMA"               : "Double Exponential Moving Average",
    "DX"                 : "Directional Movement Index",
    "EMA"                : "Exponential Moving Average",
    "HT_DCPERIOD"        : "Hilbert Transform - Dominant Cycle Period",
    "HT_DCPHASE"         : "Hilbert Transform - Dominant Cycle Phase",
    "HT_PHASOR"          : "Hilbert Transform - Phasor Components",
    "HT_SINE"            : "Hilbert Transform - SineWave",
    "HT_TRENDLINE"       : "Hilbert Transform - Instantaneous Trendline",
    "HT_TRENDMODE"       : "Hilbert Transform - Trend vs Cycle Mode",
    "KAMA"               : "Kaufman Adaptive Moving Average",
    "LINEARREG"          : "Linear Regression",
    "LINEARREG_ANGLE"    : "Linear Regression Angle",
    "LINEARREG_INTERCEPT": "Linear Regression Intercept",
    "LINEARREG_SLOPE"    : "Linear Regression Slope",
    "MA"                 : "All Moving Average",
    "MACD"               : "Moving Average Convergence/Divergence",
    "MACDEXT"            : "MACD with controllable MA type",
    "MACDFIX"            : "Moving Average Convergence/Divergence Fix 12/26",
    "MAMA"               : "MESA Adaptive Moving Average",
    "MAX"                : "Highest value over a specified period",
    "MAXINDEX"           : "Index of highest value over a specified period",
    "MEDPRICE"           : "Median Price",
    "MFI"                : "Money Flow Index",
    "MIDPOINT"           : "MidPoint over period",
    "MIDPRICE"           : "Midpoint Price over period",
    "MIN"                : "Lowest value over a specified period",
    "MININDEX"           : "Index of lowest value over a specified period",
    "MINMAX"             : "Lowest and highest values over a specified period",
    "MINMAXINDEX"        : "Indexes of lowest and highest values over a specified period",
    "MINUS_DI"           : "Minus Directional Indicator",
    "MINUS_DM"           : "Minus Directional Movement",
    "MOM"                : "Momentum",
    "NATR"               : "Normalized Average True Range",
    "OBV"                : "On Balance Volume",
    "PLUS_DI"            : "Plus Directional Indicator",
    "PLUS_DM"            : "Plus Directional Movement",
    "PPO"                : "Percentage Price Oscillator",
    "ROC"                : "Rate of change : ((price/prevPrice)-1)*100",
    "ROCP"               : "Rate of change Percentage: (price-prevPrice)/prevPrice",
    "ROCR"               : "Rate of change ratio: (price/prevPrice)",
    "ROCR100"            : "Rate of change ratio 100 scale: (price/prevPrice)*100",
    "RSI"                : "Relative Strength Index",
    "SAR"                : "Parabolic SAR",
    "SAREXT"             : "Parabolic SAR - Extended",
    "SMA"                : "Simple Moving Average",
    "STDDEV"             : "Standard Deviation",
    "STOCH"              : "Stochastic",
    "STOCHF"             : "Stochastic Fast",
    "STOCHRSI"           : "Stochastic Relative Strength Index",
    "SUM"                : "Summation",
    "T3"                 : "Triple Exponential Moving Average (T3)",
    "TEMA"               : "Triple Exponential Moving Average",
    "TRANGE"             : "True Range",
    "TRIMA"              : "Triangular Moving Average",
    "TRIX"               : "1-day Rate-Of-Change (ROC) of a Triple Smooth EMA",
    "TSF"                : "Time Series Forecast",
    "TYPPRICE"           : "Typical Price",
    "ULTOSC"             : "Ultimate Oscillator",
    "VAR"                : "Variance",
    "WCLPRICE"           : "Weighted Close Price",
    "WILLR"              : "Williams' %R",
    "WMA"                : "Weighted Moving Average",
}


def get_defaults_and_docs(function):
    handle = abstract.FuncHandle(function)
    func_info = handle.get_info()
    defaults = {}
    INDENT = '    ' # 4 spaces
    docs = []
    docs.append('%s%s' % (INDENT, func_info['display_name']))
    docs.append('Group: %(group)s' % func_info)

    inputs = func_info['inputs']
    docs.append('Inputs:')
    for input_ in inputs:
        value = inputs[input_]
        if not isinstance(value, list):
            value = '(any ndarray)'
        docs.append('%s%s: %s' % (INDENT, input_, value))

    params = func_info['parameters']
    if params:
        docs.append('Parameters:')
    for param in params:
        docs.append('%s%s: %s' % (INDENT, param.lower(), params[param]))
        defaults[param] = params[param]
        if param.lower() == 'matype':
            docs[-1] = ' '.join([docs[-1], '(%s)' % talib.MA_Type[params[param]]])

    outputs = func_info['outputs']
    docs.append('Outputs:')
    for output in outputs:
        if output == 'integer':
            output = 'integer (values are -100, 0 or 100)'
        docs.append('%s%s' % (INDENT, output))
    docs.append('')

    documentation = '\n    '.join(docs) # 4 spaces
    return defaults, documentation


# print functions
names = []
for f in functions:
    if 'Lookback' in f: # skip lookback functions
        continue

    i = f.index('(')
    name = f[:i].split()[1]
    args = f[i:].split(',')
    args = [re.sub('[\(\);]', '', s).strip() for s in args]

    shortname = name[3:]
    names.append(shortname)
    defaults, documentation = get_defaults_and_docs(shortname)

    print '@wraparound(False)  # turn off relative indexing from end of lists'
    print '@boundscheck(False) # turn off bounds-checking for entire function'
    print 'def %s(' % shortname,
    docs = ['%s(' % shortname]
    i = 0
    for arg in args:
        var = arg.split()[-1]

        if var in ('startIdx', 'endIdx'):
            continue

        elif 'out' in var:
            break

        if i > 0:
            print ',',
        i += 1

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            assert arg.startswith('const double'), arg
            print 'np.ndarray %s not None' % var,
            docs.append(var)
            docs.append(', ')

        elif var.startswith('opt'):
            var = cleanup(var)
            default_arg = arg.split()[-1][len('optIn'):] # chop off typedef and 'optIn'
            default_arg = default_arg[0].lower() + default_arg[1:] # lowercase first letter

            if arg.startswith('double'):
                if default_arg in defaults:
                    print 'double %s=%s' % (var, defaults[default_arg]),
                else:
                    print 'double %s=-4e37' % var, # TA_REAL_DEFAULT
            elif arg.startswith('int'):
                if default_arg in defaults:
                    print 'int %s=%s' % (var, defaults[default_arg]),
                else:
                    print 'int %s=-2**31' % var,   # TA_INTEGER_DEFAULT
            elif arg.startswith('TA_MAType'):
                print 'int %s=0' % var,        # TA_MAType_SMA
            else:
                assert False, arg
            if '[, ' not in docs:
                docs[-1] = ('[, ')
            docs.append('%s=?' % var)
            docs.append(', ')

    docs[-1] = '])' if '[, ' in docs else ')'
    if documentation:
        docs.append('\n\n')
        docs.append(documentation)
    print '):'
    print '    """%s"""' % ''.join(docs)
    print '    cdef:'
    print '        np.npy_intp length'
    print '        int begidx, endidx, lookback'
    for arg in args:
        var = arg.split()[-1]

        if 'out' in var:
            break

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print '        double* %s_data' % var
            elif 'int' in arg:
                print '        int* %s_data' % var
            else:
                assert False, args

    for arg in args:
        var = arg.split()[-1]

        if 'out' not in var:
            continue

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            print '        np.ndarray %s' % var
            if 'double' in arg:
                print '        double* %s_data' % var
            elif 'int' in arg:
                print '        int* %s_data' % var
            else:
                assert False, args

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print '        int %s' % var

        else:
            assert False, arg

    for arg in args:
        var = arg.split()[-1]
        if 'out' in var:
            break
        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                cast = '<double*>'
            elif 'int' in arg:
                cast = '<int*>'
            else:
                assert False, arg
            print '    assert PyArray_TYPE(%s) == np.NPY_DOUBLE, "%s is not double"' % (var, var)
            print '    assert %s.ndim == 1, "%s has wrong dimensions"' % (var, var)
            print '    if not (PyArray_FLAGS(%s) & np.NPY_C_CONTIGUOUS):' % var
            print '        %s = PyArray_GETCONTIGUOUS(%s)' % (var, var)
            print '    %s_data = %s%s.data' % (var, cast, var)

    for arg in args:
        var = arg.split()[-1]
        if var in ('inReal0[]', 'inReal1[]', 'inReal[]', 'inHigh[]'):
            var = cleanup(var[:-2])
            print '    length = %s.shape[0]' % var
            print '    begidx = 0'
            print '    for i from 0 <= i < length:'
            print '        if not isnan(%s_data[i]):' % var
            print '            begidx = i'
            print '            break'
            print '    else:'
            print '        raise Exception("inputs are all NaN")'
            print '    endidx = length - begidx - 1'
            break

    print '    initialize()'
    print '    lookback = begidx + %s_Lookback(' % name,
    opts = [arg for arg in args if 'opt' in arg]
    for i, opt in enumerate(opts):
        if i > 0:
            print ',',
        print cleanup(opt.split()[-1]),
    print ')'

    for arg in args:
        var = arg.split()[-1]

        if 'out' not in var:
            continue

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print '    %s = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)' % var
                print '    %s_data = <double*>%s.data' % (var, var)
                print '    for i from 0 <= i < min(lookback, length):'
                print '        %s_data[i] = NaN' % var
            elif 'int' in arg:
                print '    %s = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)' % var
                print '    %s_data = <int*>%s.data' % (var, var)
                print '    for i from 0 <= i < min(lookback, length):'
                print '        %s_data[i] = 0' % var
            else:
                assert False, args

    print '    retCode = %s(' % name,

    for i, arg in enumerate(args):
        if i > 0:
            print ',',
        var = arg.split()[-1]

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'out' in var:
                data = '(%s_data+lookback)' % var
            else:
                data = '(%s_data+begidx)' % var
            if 'double' in arg:
                print '<double *>%s' % data,
            elif 'int' in arg:
                print '<int *>%s' % data,
            else:
                assert False, arg

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print '&%s' % var,

        else:
            print cleanup(var) if var != 'startIdx' else '0',

    print ')'
    print '    shutdown()'
    print '    return',
    i = 0
    for arg in args:
        var = arg.split()[-1]
        if var.endswith('[]'):
            var = var[:-2]
        elif var.startswith('*'):
            var = var[1:]
        if var.startswith('out'):
            if var not in ("outNBElement", "outBegIdx"):
                if i > 0:
                    print ',',
                i += 1
                print cleanup(var),
        else:
            assert re.match('.*(void|startIdx|endIdx|opt|in)/*', arg), arg
    print
    print

print '__all__ = [%s]' % ','.join(['\"%s\"' % name for name in names])


import re

# FIXME: initialize once, then shutdown at the end, rather than each call?
# FIXME: should we check retCode from initialize and shutdown?
# FIXME: should we pass startIdx and endIdx into function?
# FIXME: should we parse the function docstrings from the c header?
# FIXME: don't return number of elements since it always equals allocation?

functions = []
with open('/usr/local/include/ta-lib/ta_func.h') as f:
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
from numpy import empty, nan, int32, double, ascontiguousarray
cimport numpy as np

ctypedef int TA_RetCode
ctypedef int TA_MAType

# TA_MAType enums
MA_SMA, MA_EMA, MA_WMA, MA_DEMA, MA_TEMA, MA_TRIMA, MA_KAMA, MA_MAMA, MA_T3 = range(9)

# TA_RetCode enums
RetCodes = {
    0: 'Success',
    1: 'Library Not Initialized',
    2: 'Bad Parameter',
    3: 'Allocation Error',
    4: 'Group Not Found',
    5: 'Function Not Found',
    6: 'Invalid Handle',
    7: 'Invalid Parameter Holder',
    8: 'Invalid Parameter Holder Type',
    9: 'Invalid Parameter Function',
   10: 'Input Not All Initialized',
   11: 'Output Not All Initialized',
   12: 'Out-of-Range Start Index',
   13: 'Out-of-Range End Index',
   14: 'Invalid List Type',
   15: 'Bad Object',
   16: 'Not Supported',
 5000: 'Internal Error',
65535: 'Unknown Error',
}

# extract the needed part of ta_libc.h that I will use in the interface
cdef extern from "ta_libc.h":
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
    print "def %s(" % shortname,
    docs = ["%s(" % shortname]
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
            print 'np.ndarray[np.double_t, ndim=1] %s' % var,
            docs.append(var)
            docs.append(', ')

        elif var.startswith('opt'):
            var = cleanup(var)
            if arg.startswith('double'):
                print '%s=-4e37' % var,  # TA_REAL_DEFAULT
            elif arg.startswith('int'):
                print '%s=-2**31' % var, # TA_INTEGER_DEFAULT
            elif arg.startswith('TA_MAType'):
                print '%s=0' % var,      # TA_MAType_SMA
            else:
                assert False, arg
            if '[, ' not in docs:
                docs[-1] = ('[, ')
            docs.append('%s=?' % var)
            docs.append(', ')

    docs[-1] = '])' if '[, ' in docs else ')'
    desc = descriptions.get(shortname)
    if desc is not None:
        docs.append('\n\n    ');
        docs.append(desc)
    print '):'
    print '    """%s"""' % ''.join(docs)

    for arg in args:
        var = arg.split()[-1]
        if 'out' in var:
            break
        if var.endswith('[]'):
            var = cleanup(var[:-2])
            print '    %s = ascontiguousarray(%s, dtype=double)' % (var, var)

    for arg in args:
        var = arg.split()[-1]
        if var in ('inReal0[]', 'inReal1[]', 'inReal[]', 'inHigh[]'):
            var = cleanup(var[:-2])
            print '    cdef int endidx = %s.shape[0] - 1' % var
            break

    print '    TA_Initialize()'
    print '    cdef int lookback = %s_Lookback(' % name,
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
                vartype = 'np.double_t'
                dtype = 'double'
            elif 'int' in arg:
                vartype = 'np.int32_t'
                dtype = 'int32'
            else:
                assert False, args
            print '    cdef np.ndarray[%s, ndim=1] %s = empty(endidx + 1, dtype=%s)' % (vartype, var, dtype)
            print '    %s.fill(nan)' % var
            print '    assert id(%s) == id(ascontiguousarray(%s, dtype=%s))' % (var, var, dtype)

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print '    cdef int %s' % var

        else:
            assert False, arg

    print '    retCode = %s(' % name,

    for i, arg in enumerate(args):
        if i > 0:
            print ',',
        var = arg.split()[-1]

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'out' in var:
                data = '%s.data+lookback' % var
            else:
                data = '%s.data' % var
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
    print '    TA_Shutdown()'
    print '    if retCode != TA_SUCCESS:'
    print '        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))'

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

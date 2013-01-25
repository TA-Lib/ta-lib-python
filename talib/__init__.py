
__version__ = '0.4.3-git'

import atexit

#from talib.func import *
from talib import abstract
from talib import common_c

'''
In order to use this python library, talib (ie this __file__) will be imported
at some point, either explicitly or indirectly via talib.func or talib.abstract.
Here, we handle initalizing and shutting down the underlying TA-Lib.
Initialization happens on import, before any other TA-Lib functions are called.
Finally, when the python process exits, we shutdown the underlying TA-Lib.
'''
common_c._ta_initialize()
atexit.register(common_c._ta_shutdown)


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


# --------------  Moving Averages (Overlap Studies) ----------------------------
MA = abstract.Function("MA")             # Moving average
SMA = abstract.Function("SMA")           # Simple Moving Average
EMA = abstract.Function("EMA")           # Exponential Moving Average
WMA = abstract.Function("WMA")           # Weighted Moving Average
DEMA = abstract.Function("DEMA")         # Double Exponential Moving Average
TEMA = abstract.Function("TEMA")         # Triple Exponential Moving Average
TRIMA = abstract.Function("TRIMA")       # Triangular Moving Average
KAMA = abstract.Function("KAMA")         # Kaufman Adaptive Moving Average
MAMA = abstract.Function("MAMA")         # MESA Adaptive Moving Average
T3 = abstract.Function("T3")             # Triple Exponential Moving Average (T3)
MAVP = abstract.Function("MAVP")         # Moving average with variable period

# --------------  Overlap Studies cont.  ---------------------------------------
BBANDS = abstract.Function("BBANDS")     # Bollinger Bands
MIDPOINT = abstract.Function("MIDPOINT") # MidPoint over period
MIDPRICE = abstract.Function("MIDPRICE") # Midpoint Price over period
SAR = abstract.Function("SAR")           # Parabolic SAR
SAREXT = abstract.Function("SAREXT")     # Parabolic SAR - Extended

# --------------  Momentum Indicators  -----------------------------------------
ADX = abstract.Function("ADX")           # Average Directional Movement Index
ADXR = abstract.Function("ADXR")         # Average Directional Movement Index Rating
APO = abstract.Function("APO")           # Absolute Price Oscillator
AROON = abstract.Function("AROON")       # Aroon
AROONOSC = abstract.Function("AROONOSC") # Aroon Oscillator
BOP = abstract.Function("BOP")           # Balance Of Power
CCI = abstract.Function("CCI")           # Commodity Channel Index
CMO = abstract.Function("CMO")           # Chande Momentum Oscillator
DX = abstract.Function("DX")             # Directional Movement Index
MACD = abstract.Function("MACD")         # Moving Average Convergence/Divergence
MACDEXT = abstract.Function("MACDEXT")   # MACD with controllable MA type
MACDFIX = abstract.Function("MACDFIX")   # Moving Average Convergence/Divergence Fix 12/26
MFI = abstract.Function("MFI")           # Money Flow Index
MINUS_DI = abstract.Function("MINUS_DI") # Minus Directional Indicator
MINUS_DM = abstract.Function("MINUS_DM") # Minus Directional Movement
MOM = abstract.Function("MOM")           # Momentum
PLUS_DI = abstract.Function("PLUS_DI")   # Plus Directional Indicator
PLUS_DM = abstract.Function("PLUS_DM")   # Plus Directional Movement
PPO = abstract.Function("PPO")           # Percentage Price Oscillator
ROC = abstract.Function("ROC")           # Rate of change : ((price/prevPrice)-1)*100
ROCP = abstract.Function("ROCP")         # Rate of change Percentage: (price-prevPrice)/prevPrice
ROCR = abstract.Function("ROCR")         # Rate of change ratio: (price/prevPrice)
ROCR100 = abstract.Function("ROCR100")   # Rate of change ratio 100 scale: (price/prevPrice)*100
RSI = abstract.Function("RSI")           # Relative Strength Index
STOCH = abstract.Function("STOCH")       # Stochastic
STOCHF = abstract.Function("STOCHF")     # Stochastic Fast
STOCHRSI = abstract.Function("STOCHRSI") # Stochastic Relative Strength Index
TRIX = abstract.Function("TRIX")         # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
ULTOSC = abstract.Function("ULTOSC")     # Ultimate Oscillator
WILLR = abstract.Function("WILLR")       # Williams' %R

# --------------  Volume Indicators  -------------------------------------------
AD = abstract.Function("AD")             # Chaikin A/D Line
ADOSC = abstract.Function("ADOSC")       # Chaikin A/D Oscillator
OBV = abstract.Function("OBV")           # On Balance Volume

# --------------  Volatility Indicators  ---------------------------------------
ATR = abstract.Function("ATR")           # Average True Range
NATR = abstract.Function("NATR")         # Normalized Average True Range
TRANGE = abstract.Function("TRANGE")     # True Range

# --------------  Pattern Recognition  -----------------------------------------
CDL2CROWS = abstract.Function("CDL2CROWS")                     # Two Crows
CDL3BLACKCROWS = abstract.Function("CDL3BLACKCROWS")           # Three Black Crows
CDL3INSIDE = abstract.Function("CDL3INSIDE")                   # Three Inside Up/Down
CDL3LINESTRIKE = abstract.Function("CDL3LINESTRIKE")           # Three-Line Strike
CDL3OUTSIDE = abstract.Function("CDL3OUTSIDE")                 # Three Outside Up/Down
CDL3STARSINSOUTH = abstract.Function("CDL3STARSINSOUTH")       # Three Stars In The South
CDL3WHITESOLDIERS = abstract.Function("CDL3WHITESOLDIERS")     # Three Advancing White Soldiers
CDLABANDONEDBABY = abstract.Function("CDLABANDONEDBABY")       # Abandoned Baby
CDLADVANCEBLOCK = abstract.Function("CDLADVANCEBLOCK")         # Advance Block
CDLBELTHOLD = abstract.Function("CDLBELTHOLD")                 # Belt-hold
CDLBREAKAWAY = abstract.Function("CDLBREAKAWAY")               # Breakaway
CDLCLOSINGMARUBOZU = abstract.Function("CDLCLOSINGMARUBOZU")   # Closing Marubozu
CDLCONCEALBABYSWALL = abstract.Function("CDLCONCEALBABYSWALL") # Concealing Baby Swallow
CDLCOUNTERATTACK = abstract.Function("CDLCOUNTERATTACK")       # Counterattack
CDLDARKCLOUDCOVER = abstract.Function("CDLDARKCLOUDCOVER")     # Dark Cloud Cover
CDLDOJI = abstract.Function("CDLDOJI")                         # Doji
CDLDOJISTAR = abstract.Function("CDLDOJISTAR")                 # Doji Star
CDLDRAGONFLYDOJI = abstract.Function("CDLDRAGONFLYDOJI")       # Dragonfly Doji
CDLENGULFING = abstract.Function("CDLENGULFING")               # Engulfing Pattern
CDLEVENINGDOJISTAR = abstract.Function("CDLEVENINGDOJISTAR")   # Evening Doji Star
CDLEVENINGSTAR = abstract.Function("CDLEVENINGSTAR")           # Evening Star
CDLGAPSIDESIDEWHITE = abstract.Function("CDLGAPSIDESIDEWHITE") # Up/Down-gap side-by-side white lines
CDLGRAVESTONEDOJI = abstract.Function("CDLGRAVESTONEDOJI")     # Gravestone Doji
CDLHAMMER = abstract.Function("CDLHAMMER")                     # Hammer
CDLHANGINGMAN = abstract.Function("CDLHANGINGMAN")             # Hanging Man
CDLHARAMI = abstract.Function("CDLHARAMI")                     # Harami Pattern
CDLHARAMICROSS = abstract.Function("CDLHARAMICROSS")           # Harami Cross Pattern
CDLHIGHWAVE = abstract.Function("CDLHIGHWAVE")                 # High-Wave Candle
CDLHIKKAKE = abstract.Function("CDLHIKKAKE")                   # Hikkake Pattern
CDLHIKKAKEMOD = abstract.Function("CDLHIKKAKEMOD")             # Modified Hikkake Pattern
CDLHOMINGPIGEON = abstract.Function("CDLHOMINGPIGEON")         # Homing Pigeon
CDLIDENTICAL3CROWS = abstract.Function("CDLIDENTICAL3CROWS")   # Identical Three Crows
CDLINNECK = abstract.Function("CDLINNECK")                     # In-Neck Pattern
CDLINVERTEDHAMMER = abstract.Function("CDLINVERTEDHAMMER")     # Inverted Hammer
CDLKICKING = abstract.Function("CDLKICKING")                   # Kicking
CDLKICKINGBYLENGTH = abstract.Function("CDLKICKINGBYLENGTH")   # Kicking - bull/bear determined by the longer marubozu
CDLLADDERBOTTOM = abstract.Function("CDLLADDERBOTTOM")         # Ladder Bottom
CDLLONGLEGGEDDOJI = abstract.Function("CDLLONGLEGGEDDOJI")     # Long Legged Doji
CDLLONGLINE = abstract.Function("CDLLONGLINE")                 # Long Line Candle
CDLMARUBOZU = abstract.Function("CDLMARUBOZU")                 # Marubozu
CDLMATCHINGLOW = abstract.Function("CDLMATCHINGLOW")           # Matching Low
CDLMATHOLD = abstract.Function("CDLMATHOLD")                   # Mat Hold
CDLMORNINGDOJISTAR = abstract.Function("CDLMORNINGDOJISTAR")   # Morning Doji Star
CDLMORNINGSTAR = abstract.Function("CDLMORNINGSTAR")           # Morning Star
CDLONNECK = abstract.Function("CDLONNECK")                     # On-Neck Pattern
CDLPIERCING = abstract.Function("CDLPIERCING")                 # Piercing Pattern
CDLRICKSHAWMAN = abstract.Function("CDLRICKSHAWMAN")           # Rickshaw Man
CDLRISEFALL3METHODS = abstract.Function("CDLRISEFALL3METHODS") # Rising/Falling Three Methods
CDLSEPARATINGLINES = abstract.Function("CDLSEPARATINGLINES")   # Separating Lines
CDLSHOOTINGSTAR = abstract.Function("CDLSHOOTINGSTAR")         # Shooting Star
CDLSHORTLINE = abstract.Function("CDLSHORTLINE")               # Short Line Candle
CDLSPINNINGTOP = abstract.Function("CDLSPINNINGTOP")           # Spinning Top
CDLSTALLEDPATTERN = abstract.Function("CDLSTALLEDPATTERN")     # Stalled Pattern
CDLSTICKSANDWICH = abstract.Function("CDLSTICKSANDWICH")       # Stick Sandwich
CDLTAKURI = abstract.Function("CDLTAKURI")                     # Takuri (Dragonfly Doji with very long lower shadow)
CDLTASUKIGAP = abstract.Function("CDLTASUKIGAP")               # Tasuki Gap
CDLTHRUSTING = abstract.Function("CDLTHRUSTING")               # Thrusting Pattern
CDLTRISTAR = abstract.Function("CDLTRISTAR")                   # Tristar Pattern
CDLUNIQUE3RIVER = abstract.Function("CDLUNIQUE3RIVER")         # Unique 3 River
CDLUPSIDEGAP2CROWS = abstract.Function("CDLUPSIDEGAP2CROWS")   # Upside Gap Two Crows
CDLXSIDEGAP3METHODS = abstract.Function("CDLXSIDEGAP3METHODS") # Upside/Downside Gap Three Methods

# --------------  Statistic Functions  -----------------------------------------
BETA = abstract.Function("BETA")                               # Beta
CORREL = abstract.Function("CORREL")                           # Pearson's Correlation Coefficient (r)
LINEARREG = abstract.Function("LINEARREG")                     # Linear Regression
LINEARREG_ANGLE = abstract.Function("LINEARREG_ANGLE")         # Linear Regression Angle
LINEARREG_INTERCEPT = abstract.Function("LINEARREG_INTERCEPT") # Linear Regression Intercept
LINEARREG_SLOPE = abstract.Function("LINEARREG_SLOPE")         # Linear Regression Slope
STDDEV = abstract.Function("STDDEV")                           # Standard Deviation
TSF = abstract.Function("TSF")                                 # Time Series Forecast
VAR = abstract.Function("VAR")                                 # Variance

# --------------  Cycle Indicators  --------------------------------------------
HT_DCPERIOD = abstract.Function("HT_DCPERIOD")   # Hilbert Transform - Dominant Cycle Period
HT_DCPHASE = abstract.Function("HT_DCPHASE")     # Hilbert Transform - Dominant Cycle Phase
HT_PHASOR = abstract.Function("HT_PHASOR")       # Hilbert Transform - Phasor Components
HT_SINE = abstract.Function("HT_SINE")           # Hilbert Transform - SineWave
HT_TRENDMODE = abstract.Function("HT_TRENDMODE") # Hilbert Transform - Trend vs Cycle Mode
HT_TRENDLINE = abstract.Function("HT_TRENDLINE") # Hilbert Transform - Instantaneous Trendline >> part of "Overlap Studies" group

# --------------  Price Transform  ---------------------------------------------
AVGPRICE = abstract.Function("AVGPRICE")         # Average Price
MEDPRICE = abstract.Function("MEDPRICE")         # Median Price
TYPPRICE = abstract.Function("TYPPRICE")         # Typical Price
WCLPRICE = abstract.Function("WCLPRICE")         # Weighted Close Price

# --------------  Math Transform and Operators  --------------------------------
ADD = abstract.Function("ADD")                   # Vector Arithmetic Add
SUB = abstract.Function("SUB")                   # Vector Arithmetic Substraction
SUM = abstract.Function("SUM")                   # Summation
MULT = abstract.Function("MULT")                 # Vector Arithmetic Mult
DIV = abstract.Function("DIV")                   # Vector Arithmetic Div

CEIL = abstract.Function("CEIL")                 # Vector Ceil
FLOOR = abstract.Function("FLOOR")               # Vector Floor

MIN = abstract.Function("MIN")                   # Lowest value over a specified period
MAX = abstract.Function("MAX")                   # Highest value over a specified period
MINMAX = abstract.Function("MINMAX")             # Lowest and highest values over a specified period
MININDEX = abstract.Function("MININDEX")         # Index of lowest value over a specified period
MAXINDEX = abstract.Function("MAXINDEX")         # Index of highest value over a specified period
MINMAXINDEX = abstract.Function("MINMAXINDEX")   # Indexes of lowest and highest values over a specified period

SQRT = abstract.Function("SQRT")                 # Vector Square Root
EXP = abstract.Function("EXP")                   # Vector Arithmetic Exp
LOG10 = abstract.Function("LOG10")               # Vector Log10
LN = abstract.Function("LN")                     # Vector Log Natural

SIN = abstract.Function("SIN")                   # Vector Trigonometric Sin
COS = abstract.Function("COS")                   # Vector Trigonometric Cos
TAN = abstract.Function("TAN")                   # Vector Trigonometric Tan

ASIN = abstract.Function("ASIN")                 # Vector Trigonometric ASin
ACOS = abstract.Function("ACOS")                 # Vector Trigonometric ACos
ATAN = abstract.Function("ATAN")                 # Vector Trigonometric ATan

SINH = abstract.Function("SINH")                 # Vector Trigonometric Sinh
COSH = abstract.Function("COSH")                 # Vector Trigonometric Cosh
TANH = abstract.Function("TANH")                 # Vector Trigonometric Tanh

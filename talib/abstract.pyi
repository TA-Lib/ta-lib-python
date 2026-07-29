
from typing import overload, Tuple, Union
import numpy as np
import pandas as pd

"""HT_DCPERIOD(real)

Hilbert Transform - Dominant Cycle Period (Cycle Indicators)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def HT_DCPERIOD(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def HT_DCPERIOD(real: pd.DataFrame) -> pd.Series: ...

"""HT_DCPHASE(real)

Hilbert Transform - Dominant Cycle Phase (Cycle Indicators)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def HT_DCPHASE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def HT_DCPHASE(real: pd.DataFrame) -> pd.Series: ...

"""HT_PHASOR(real)

Hilbert Transform - Phasor Components (Cycle Indicators)

Inputs:
    real: (any ndarray)
Outputs:
    inphase
    quadrature"""
@overload
def HT_PHASOR(real: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def HT_PHASOR(real: pd.DataFrame) -> pd.DataFrame: ...

"""HT_SINE(real)

Hilbert Transform - SineWave (Cycle Indicators)

Inputs:
    real: (any ndarray)
Outputs:
    sine
    leadsine"""
@overload
def HT_SINE(real: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def HT_SINE(real: pd.DataFrame) -> pd.DataFrame: ...

"""HT_TRENDMODE(real)

Hilbert Transform - Trend vs Cycle Mode (Cycle Indicators)

Inputs:
    real: (any ndarray)
Outputs:
    integer"""
@overload
def HT_TRENDMODE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def HT_TRENDMODE(real: pd.DataFrame) -> pd.Series: ...

"""ADD(real0, real1)

Vector Arithmetic Add (Math Operators)

Inputs:
    real0: (any ndarray)
    real1: (any ndarray)
Outputs:
    real"""
@overload
def ADD(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def ADD(real: pd.DataFrame) -> pd.Series: ...

"""CUMSUM(real)

Cumulative Sum (Math Operators)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def CUMSUM(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CUMSUM(real: pd.DataFrame) -> pd.Series: ...

"""DIV(real0, real1)

Vector Arithmetic Div (Math Operators)

Inputs:
    real0: (any ndarray)
    real1: (any ndarray)
Outputs:
    real"""
@overload
def DIV(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def DIV(real: pd.DataFrame) -> pd.Series: ...

"""MAX(real[, timeperiod=?])

Highest value over a specified period (Math Operators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def MAX(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def MAX(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""MAXINDEX(real[, timeperiod=?])

Index of highest value over a specified period (Math Operators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    integer"""
@overload
def MAXINDEX(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def MAXINDEX(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""MIN(real[, timeperiod=?])

Lowest value over a specified period (Math Operators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def MIN(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def MIN(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""MININDEX(real[, timeperiod=?])

Index of lowest value over a specified period (Math Operators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    integer"""
@overload
def MININDEX(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def MININDEX(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""MINMAX(real[, timeperiod=?])

Lowest and highest values over a specified period (Math Operators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    min
    max"""
@overload
def MINMAX(real: Union[pd.Series, np.ndarray], timeperiod=30) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def MINMAX(real: pd.DataFrame, timeperiod=30) -> pd.DataFrame: ...

"""MINMAXINDEX(real[, timeperiod=?])

Indexes of lowest and highest values over a specified period (Math Operators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    minidx
    maxidx"""
@overload
def MINMAXINDEX(real: Union[pd.Series, np.ndarray], timeperiod=30) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def MINMAXINDEX(real: pd.DataFrame, timeperiod=30) -> pd.DataFrame: ...

"""MULT(real0, real1)

Vector Arithmetic Mult (Math Operators)

Inputs:
    real0: (any ndarray)
    real1: (any ndarray)
Outputs:
    real"""
@overload
def MULT(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def MULT(real: pd.DataFrame) -> pd.Series: ...

"""SUB(real0, real1)

Vector Arithmetic Subtraction (Math Operators)

Inputs:
    real0: (any ndarray)
    real1: (any ndarray)
Outputs:
    real"""
@overload
def SUB(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def SUB(real: pd.DataFrame) -> pd.Series: ...

"""SUM(real[, timeperiod=?])

Summation (Math Operators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def SUM(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def SUM(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""ACOS(real)

Vector Trigonometric ACos (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def ACOS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def ACOS(real: pd.DataFrame) -> pd.Series: ...

"""ASIN(real)

Vector Trigonometric ASin (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def ASIN(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def ASIN(real: pd.DataFrame) -> pd.Series: ...

"""ATAN(real)

Vector Trigonometric ATan (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def ATAN(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def ATAN(real: pd.DataFrame) -> pd.Series: ...

"""CEIL(real)

Vector Ceil (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def CEIL(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CEIL(real: pd.DataFrame) -> pd.Series: ...

"""COS(real)

Vector Trigonometric Cos (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def COS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def COS(real: pd.DataFrame) -> pd.Series: ...

"""COSH(real)

Vector Trigonometric Cosh (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def COSH(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def COSH(real: pd.DataFrame) -> pd.Series: ...

"""EXP(real)

Vector Arithmetic Exp (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def EXP(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def EXP(real: pd.DataFrame) -> pd.Series: ...

"""FLOOR(real)

Vector Floor (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def FLOOR(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def FLOOR(real: pd.DataFrame) -> pd.Series: ...

"""LN(real)

Vector Log Natural (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def LN(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def LN(real: pd.DataFrame) -> pd.Series: ...

"""LOG10(real)

Vector Log10 (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def LOG10(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def LOG10(real: pd.DataFrame) -> pd.Series: ...

"""SIN(real)

Vector Trigonometric Sin (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def SIN(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def SIN(real: pd.DataFrame) -> pd.Series: ...

"""SINH(real)

Vector Trigonometric Sinh (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def SINH(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def SINH(real: pd.DataFrame) -> pd.Series: ...

"""SQRT(real)

Vector Square Root (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def SQRT(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def SQRT(real: pd.DataFrame) -> pd.Series: ...

"""TAN(real)

Vector Trigonometric Tan (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def TAN(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def TAN(real: pd.DataFrame) -> pd.Series: ...

"""TANH(real)

Vector Trigonometric Tanh (Math Transform)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def TANH(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def TANH(real: pd.DataFrame) -> pd.Series: ...

"""AC(high, low[, fastperiod=?, slowperiod=?, signalperiod=?])

Accelerator/Decelerator Oscillator (Momentum Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    fastperiod: 5
    slowperiod: 34
    signalperiod: 5
Outputs:
    real"""
@overload
def AC(real: Union[pd.Series, np.ndarray], fastperiod=5, slowperiod=34, signalperiod=5) -> np.ndarray: ...
@overload
def AC(real: pd.DataFrame, fastperiod=5, slowperiod=34, signalperiod=5) -> pd.Series: ...

"""ADX(high, low, close[, timeperiod=?])

Average Directional Movement Index (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def ADX(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def ADX(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""ADXR(high, low, close[, timeperiod=?])

Average Directional Movement Index Rating (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def ADXR(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def ADXR(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""AO(high, low[, fastperiod=?, slowperiod=?])

Awesome Oscillator (Momentum Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    fastperiod: 5
    slowperiod: 34
Outputs:
    real"""
@overload
def AO(real: Union[pd.Series, np.ndarray], fastperiod=5, slowperiod=34) -> np.ndarray: ...
@overload
def AO(real: pd.DataFrame, fastperiod=5, slowperiod=34) -> pd.Series: ...

"""APO(real[, fastperiod=?, slowperiod=?, matype=?])

Absolute Price Oscillator (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    fastperiod: 12
    slowperiod: 26
    matype: 1 (Exponential Moving Average)
Outputs:
    real"""
@overload
def APO(real: Union[pd.Series, np.ndarray], fastperiod=12, slowperiod=26, matype=1) -> np.ndarray: ...
@overload
def APO(real: pd.DataFrame, fastperiod=12, slowperiod=26, matype=1) -> pd.Series: ...

"""AROON(high, low[, timeperiod=?])

Aroon (Momentum Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    timeperiod: 14
Outputs:
    aroondown
    aroonup"""
@overload
def AROON(real: Union[pd.Series, np.ndarray], timeperiod=14) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def AROON(real: pd.DataFrame, timeperiod=14) -> pd.DataFrame: ...

"""AROONOSC(high, low[, timeperiod=?])

Aroon Oscillator (Momentum Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def AROONOSC(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def AROONOSC(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""BOP(open, high, low, close)

Balance Of Power (Momentum Indicators)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    real"""
@overload
def BOP(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def BOP(real: pd.DataFrame) -> pd.Series: ...

"""CCI(high, low, close[, timeperiod=?])

Commodity Channel Index (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def CCI(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def CCI(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""CMO(real[, timeperiod=?])

Chande Momentum Oscillator (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def CMO(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def CMO(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""CMOU(real[, timeperiod=?])

Chande Momentum Oscillator (Unsmoothed) (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def CMOU(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def CMOU(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""COPPOCK(real[, wmaperiod=?, roc1period=?, roc2period=?])

Coppock Curve (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    wmaperiod: 10
    roc1period: 11
    roc2period: 14
Outputs:
    real"""
@overload
def COPPOCK(real: Union[pd.Series, np.ndarray], wmaperiod=10, roc1period=11, roc2period=14) -> np.ndarray: ...
@overload
def COPPOCK(real: pd.DataFrame, wmaperiod=10, roc1period=11, roc2period=14) -> pd.Series: ...

"""DPO(real[, timeperiod=?])

Detrended Price Oscillator (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 20
Outputs:
    real"""
@overload
def DPO(real: Union[pd.Series, np.ndarray], timeperiod=20) -> np.ndarray: ...
@overload
def DPO(real: pd.DataFrame, timeperiod=20) -> pd.Series: ...

"""DX(high, low, close[, timeperiod=?])

Directional Movement Index (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def DX(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def DX(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""ER(real[, timeperiod=?])

Kaufman Efficiency Ratio (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 10
Outputs:
    real"""
@overload
def ER(real: Union[pd.Series, np.ndarray], timeperiod=10) -> np.ndarray: ...
@overload
def ER(real: pd.DataFrame, timeperiod=10) -> pd.Series: ...

"""ERI(high, low, close[, timeperiod=?])

Elder Ray Index (Bull Power / Bear Power) (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 13
Outputs:
    bullpower
    bearpower"""
@overload
def ERI(real: Union[pd.Series, np.ndarray], timeperiod=13) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def ERI(real: pd.DataFrame, timeperiod=13) -> pd.DataFrame: ...

"""FOSC(real[, timeperiod=?])

Forecast Oscillator (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 5
Outputs:
    real"""
@overload
def FOSC(real: Union[pd.Series, np.ndarray], timeperiod=5) -> np.ndarray: ...
@overload
def FOSC(real: pd.DataFrame, timeperiod=5) -> pd.Series: ...

"""FRACTAL(high, low[, leftbars=?, rightbars=?])

Williams Fractal (Momentum Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    leftbars: 2
    rightbars: 2
Outputs:
    swinghigh
    swinglow"""
@overload
def FRACTAL(real: Union[pd.Series, np.ndarray], leftbars=2, rightbars=2) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def FRACTAL(real: pd.DataFrame, leftbars=2, rightbars=2) -> pd.DataFrame: ...

"""IMI(open, close[, timeperiod=?])

Intraday Momentum Index (Momentum Indicators)

Inputs:
    prices: ['open', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def IMI(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def IMI(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""KDJ(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])

KDJ Stochastic (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    fastk_period: 9
    slowk_period: 3
    slowk_matype: 13 (Wilder's Smoothed Moving Average)
    slowd_period: 3
    slowd_matype: 13 (Wilder's Smoothed Moving Average)
Outputs:
    k
    d
    j"""
@overload
def KDJ(real: Union[pd.Series, np.ndarray], fastk_period=9, slowk_period=3, slowk_matype=13, slowd_period=3, slowd_matype=13) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def KDJ(real: pd.DataFrame, fastk_period=9, slowk_period=3, slowk_matype=13, slowd_period=3, slowd_matype=13) -> pd.DataFrame: ...

"""MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])

Moving Average Convergence/Divergence (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    fastperiod: 12
    slowperiod: 26
    signalperiod: 9
Outputs:
    macd
    macdsignal
    macdhist"""
@overload
def MACD(real: Union[pd.Series, np.ndarray], fastperiod=12, slowperiod=26, signalperiod=9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def MACD(real: pd.DataFrame, fastperiod=12, slowperiod=26, signalperiod=9) -> pd.DataFrame: ...

"""MACDEXT(real[, fastperiod=?, fastmatype=?, slowperiod=?, slowmatype=?, signalperiod=?, signalmatype=?])

MACD with controllable MA type (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    fastperiod: 12
    fastmatype: 0 (Simple Moving Average)
    slowperiod: 26
    slowmatype: 0 (Simple Moving Average)
    signalperiod: 9
    signalmatype: 0 (Simple Moving Average)
Outputs:
    macd
    macdsignal
    macdhist"""
@overload
def MACDEXT(real: Union[pd.Series, np.ndarray], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def MACDEXT(real: pd.DataFrame, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0) -> pd.DataFrame: ...

"""MACDFIX(real[, signalperiod=?])

Moving Average Convergence/Divergence Fix 12/26 (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    signalperiod: 9
Outputs:
    macd
    macdsignal
    macdhist"""
@overload
def MACDFIX(real: Union[pd.Series, np.ndarray], signalperiod=9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def MACDFIX(real: pd.DataFrame, signalperiod=9) -> pd.DataFrame: ...

"""MFI(high, low, close, volume[, timeperiod=?])

Money Flow Index (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close', 'volume']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def MFI(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def MFI(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""MINUS_DI(high, low, close[, timeperiod=?])

Minus Directional Indicator (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def MINUS_DI(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def MINUS_DI(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""MINUS_DM(high, low[, timeperiod=?])

Minus Directional Movement (Momentum Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def MINUS_DM(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def MINUS_DM(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""MOM(real[, timeperiod=?])

Momentum (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 10
Outputs:
    real"""
@overload
def MOM(real: Union[pd.Series, np.ndarray], timeperiod=10) -> np.ndarray: ...
@overload
def MOM(real: pd.DataFrame, timeperiod=10) -> pd.Series: ...

"""PLUS_DI(high, low, close[, timeperiod=?])

Plus Directional Indicator (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def PLUS_DI(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def PLUS_DI(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""PLUS_DM(high, low[, timeperiod=?])

Plus Directional Movement (Momentum Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def PLUS_DM(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def PLUS_DM(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""PPO(real[, fastperiod=?, slowperiod=?, matype=?])

Percentage Price Oscillator (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    fastperiod: 12
    slowperiod: 26
    matype: 1 (Exponential Moving Average)
Outputs:
    real"""
@overload
def PPO(real: Union[pd.Series, np.ndarray], fastperiod=12, slowperiod=26, matype=1) -> np.ndarray: ...
@overload
def PPO(real: pd.DataFrame, fastperiod=12, slowperiod=26, matype=1) -> pd.Series: ...

"""QSTICK(open, close[, timeperiod=?])

Qstick (Momentum Indicators)

Inputs:
    prices: ['open', 'close']
Parameters:
    timeperiod: 10
Outputs:
    real"""
@overload
def QSTICK(real: Union[pd.Series, np.ndarray], timeperiod=10) -> np.ndarray: ...
@overload
def QSTICK(real: pd.DataFrame, timeperiod=10) -> pd.Series: ...

"""ROC(real[, timeperiod=?])

Rate of change : ((real/prevPrice)-1)*100 (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 10
Outputs:
    real"""
@overload
def ROC(real: Union[pd.Series, np.ndarray], timeperiod=10) -> np.ndarray: ...
@overload
def ROC(real: pd.DataFrame, timeperiod=10) -> pd.Series: ...

"""ROCP(real[, timeperiod=?])

Rate of change Percentage: (real-prevPrice)/prevPrice (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 10
Outputs:
    real"""
@overload
def ROCP(real: Union[pd.Series, np.ndarray], timeperiod=10) -> np.ndarray: ...
@overload
def ROCP(real: pd.DataFrame, timeperiod=10) -> pd.Series: ...

"""ROCR(real[, timeperiod=?])

Rate of change ratio: (real/prevPrice) (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 10
Outputs:
    real"""
@overload
def ROCR(real: Union[pd.Series, np.ndarray], timeperiod=10) -> np.ndarray: ...
@overload
def ROCR(real: pd.DataFrame, timeperiod=10) -> pd.Series: ...

"""ROCR100(real[, timeperiod=?])

Rate of change ratio 100 scale: (real/prevPrice)*100 (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 10
Outputs:
    real"""
@overload
def ROCR100(real: Union[pd.Series, np.ndarray], timeperiod=10) -> np.ndarray: ...
@overload
def ROCR100(real: pd.DataFrame, timeperiod=10) -> pd.Series: ...

"""RSI(real[, timeperiod=?])

Relative Strength Index (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def RSI(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def RSI(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""SMI(high, low, close[, timeperiod=?, fastperiod=?, slowperiod=?, signalperiod=?])

Stochastic Momentum Index (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 13
    fastperiod: 2
    slowperiod: 25
    signalperiod: 9
Outputs:
    smi
    smisignal"""
@overload
def SMI(real: Union[pd.Series, np.ndarray], timeperiod=13, fastperiod=2, slowperiod=25, signalperiod=9) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def SMI(real: pd.DataFrame, timeperiod=13, fastperiod=2, slowperiod=25, signalperiod=9) -> pd.DataFrame: ...

"""STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])

Stochastic (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    fastk_period: 5
    slowk_period: 3
    slowk_matype: 0 (Simple Moving Average)
    slowd_period: 3
    slowd_matype: 0 (Simple Moving Average)
Outputs:
    slowk
    slowd"""
@overload
def STOCH(real: Union[pd.Series, np.ndarray], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def STOCH(real: pd.DataFrame, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0) -> pd.DataFrame: ...

"""STOCHF(high, low, close[, fastk_period=?, fastd_period=?, fastd_matype=?])

Stochastic Fast (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    fastk_period: 5
    fastd_period: 3
    fastd_matype: 0 (Simple Moving Average)
Outputs:
    fastk
    fastd"""
@overload
def STOCHF(real: Union[pd.Series, np.ndarray], fastk_period=5, fastd_period=3, fastd_matype=0) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def STOCHF(real: pd.DataFrame, fastk_period=5, fastd_period=3, fastd_matype=0) -> pd.DataFrame: ...

"""STOCHRSI(real[, timeperiod=?, fastk_period=?, fastd_period=?, fastd_matype=?])

Stochastic Relative Strength Index (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
    fastk_period: 5
    fastd_period: 3
    fastd_matype: 0 (Simple Moving Average)
Outputs:
    fastk
    fastd"""
@overload
def STOCHRSI(real: Union[pd.Series, np.ndarray], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def STOCHRSI(real: pd.DataFrame, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0) -> pd.DataFrame: ...

"""TRIX(real[, timeperiod=?])

1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def TRIX(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def TRIX(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""TSI(real[, firstperiod=?, secondperiod=?])

True Strength Index (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    firstperiod: 25
    secondperiod: 13
Outputs:
    real"""
@overload
def TSI(real: Union[pd.Series, np.ndarray], firstperiod=25, secondperiod=13) -> np.ndarray: ...
@overload
def TSI(real: pd.DataFrame, firstperiod=25, secondperiod=13) -> pd.Series: ...

"""ULTOSC(high, low, close[, timeperiod1=?, timeperiod2=?, timeperiod3=?])

Ultimate Oscillator (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod1: 7
    timeperiod2: 14
    timeperiod3: 28
Outputs:
    real"""
@overload
def ULTOSC(real: Union[pd.Series, np.ndarray], timeperiod1=7, timeperiod2=14, timeperiod3=28) -> np.ndarray: ...
@overload
def ULTOSC(real: pd.DataFrame, timeperiod1=7, timeperiod2=14, timeperiod3=28) -> pd.Series: ...

"""VHF(real[, timeperiod=?])

Vertical Horizontal Filter (Momentum Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 28
Outputs:
    real"""
@overload
def VHF(real: Union[pd.Series, np.ndarray], timeperiod=28) -> np.ndarray: ...
@overload
def VHF(real: pd.DataFrame, timeperiod=28) -> pd.Series: ...

"""VORTEX(high, low, close[, timeperiod=?])

Vortex Indicator (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    plusvi
    minusvi"""
@overload
def VORTEX(real: Union[pd.Series, np.ndarray], timeperiod=14) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def VORTEX(real: pd.DataFrame, timeperiod=14) -> pd.DataFrame: ...

"""WAD(high, low, close)

Williams' Accumulation/Distribution (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Outputs:
    real"""
@overload
def WAD(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def WAD(real: pd.DataFrame) -> pd.Series: ...

"""WILLR(high, low, close[, timeperiod=?])

Williams' %R (Momentum Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def WILLR(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def WILLR(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""ACCBANDS(high, low, close[, timeperiod=?])

Acceleration Bands (Overlap Studies)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 20
Outputs:
    upperband
    middleband
    lowerband"""
@overload
def ACCBANDS(real: Union[pd.Series, np.ndarray], timeperiod=20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def ACCBANDS(real: pd.DataFrame, timeperiod=20) -> pd.DataFrame: ...

"""BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])

Bollinger Bands (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 20
    nbdevup: 2.0
    nbdevdn: 2.0
    matype: 0 (Simple Moving Average)
Outputs:
    upperband
    middleband
    lowerband"""
@overload
def BBANDS(real: Union[pd.Series, np.ndarray], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def BBANDS(real: pd.DataFrame, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0) -> pd.DataFrame: ...

"""DEMA(real[, timeperiod=?])

Double Exponential Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def DEMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def DEMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""DONCHIAN(high, low[, timeperiod=?])

Donchian Channels (Overlap Studies)

Inputs:
    prices: ['high', 'low']
Parameters:
    timeperiod: 20
Outputs:
    upperband
    middleband
    lowerband"""
@overload
def DONCHIAN(real: Union[pd.Series, np.ndarray], timeperiod=20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def DONCHIAN(real: pd.DataFrame, timeperiod=20) -> pd.DataFrame: ...

"""EMA(real[, timeperiod=?])

Exponential Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def EMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def EMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""HMA(real[, timeperiod=?])

Hull Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 20
Outputs:
    real"""
@overload
def HMA(real: Union[pd.Series, np.ndarray], timeperiod=20) -> np.ndarray: ...
@overload
def HMA(real: pd.DataFrame, timeperiod=20) -> pd.Series: ...

"""HT_TRENDLINE(real)

Hilbert Transform - Instantaneous Trendline (Overlap Studies)

Inputs:
    real: (any ndarray)
Outputs:
    real"""
@overload
def HT_TRENDLINE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def HT_TRENDLINE(real: pd.DataFrame) -> pd.Series: ...

"""KAMA(real[, timeperiod=?])

Kaufman Adaptive Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def KAMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def KAMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""KC(high, low, close[, timeperiod=?, atrperiod=?, nbdev=?])

Keltner Channels (Overlap Studies)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 20
    atrperiod: 10
    nbdev: 2.0
Outputs:
    upperband
    middleband
    lowerband"""
@overload
def KC(real: Union[pd.Series, np.ndarray], timeperiod=20, atrperiod=10, nbdev=2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def KC(real: pd.DataFrame, timeperiod=20, atrperiod=10, nbdev=2.0) -> pd.DataFrame: ...

"""MA(real[, timeperiod=?, matype=?])

Moving average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
    matype: 0 (Simple Moving Average)
Outputs:
    real"""
@overload
def MA(real: Union[pd.Series, np.ndarray], timeperiod=30, matype=0) -> np.ndarray: ...
@overload
def MA(real: pd.DataFrame, timeperiod=30, matype=0) -> pd.Series: ...

"""MAMA(real[, fastlimit=?, slowlimit=?])

MESA Adaptive Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    fastlimit: 0.5
    slowlimit: 0.05
Outputs:
    mama
    fama"""
@overload
def MAMA(real: Union[pd.Series, np.ndarray], fastlimit=0.5, slowlimit=0.05) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def MAMA(real: pd.DataFrame, fastlimit=0.5, slowlimit=0.05) -> pd.DataFrame: ...

"""MAVP(real, periods[, minperiod=?, maxperiod=?, matype=?])

Moving average with variable period (Overlap Studies)

Inputs:
    real: (any ndarray)
    periods: (any ndarray)
Parameters:
    minperiod: 2
    maxperiod: 30
    matype: 0 (Simple Moving Average)
Outputs:
    real"""
@overload
def MAVP(real: Union[pd.Series, np.ndarray], minperiod=2, maxperiod=30, matype=0) -> np.ndarray: ...
@overload
def MAVP(real: pd.DataFrame, minperiod=2, maxperiod=30, matype=0) -> pd.Series: ...

"""MIDPOINT(real[, timeperiod=?])

MidPoint over period (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def MIDPOINT(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def MIDPOINT(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""MIDPRICE(high, low[, timeperiod=?])

Midpoint Price over period (Overlap Studies)

Inputs:
    prices: ['high', 'low']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def MIDPRICE(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def MIDPRICE(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""RMA(real[, timeperiod=?])

Wilder's Smoothed Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def RMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def RMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""SAR(high, low[, acceleration=?, maximum=?])

Parabolic SAR (Overlap Studies)

Inputs:
    prices: ['high', 'low']
Parameters:
    acceleration: 0.02
    maximum: 0.2
Outputs:
    real"""
@overload
def SAR(real: Union[pd.Series, np.ndarray], acceleration=0.02, maximum=0.2) -> np.ndarray: ...
@overload
def SAR(real: pd.DataFrame, acceleration=0.02, maximum=0.2) -> pd.Series: ...

"""SAREXT(high, low[, startvalue=?, offsetonreverse=?, accelerationinitlong=?, accelerationlong=?, accelerationmaxlong=?, accelerationinitshort=?, accelerationshort=?, accelerationmaxshort=?])

Parabolic SAR - Extended (Overlap Studies)

Inputs:
    prices: ['high', 'low']
Parameters:
    startvalue: 0.0
    offsetonreverse: 0.0
    accelerationinitlong: 0.02
    accelerationlong: 0.02
    accelerationmaxlong: 0.2
    accelerationinitshort: 0.02
    accelerationshort: 0.02
    accelerationmaxshort: 0.2
Outputs:
    real"""
@overload
def SAREXT(real: Union[pd.Series, np.ndarray], startvalue=0.0, offsetonreverse=0.0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2) -> np.ndarray: ...
@overload
def SAREXT(real: pd.DataFrame, startvalue=0.0, offsetonreverse=0.0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2) -> pd.Series: ...

"""SMA(real[, timeperiod=?])

Simple Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def SMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def SMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""SUPERTREND(high, low, close[, timeperiod=?, multiplier=?])

SuperTrend (Overlap Studies)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 10
    multiplier: 3.0
Outputs:
    real
    integer"""
@overload
def SUPERTREND(real: Union[pd.Series, np.ndarray], timeperiod=10, multiplier=3.0) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def SUPERTREND(real: pd.DataFrame, timeperiod=10, multiplier=3.0) -> pd.DataFrame: ...

"""T3(real[, timeperiod=?, vfactor=?])

Triple Exponential Moving Average (T3) (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 5
    vfactor: 0.7
Outputs:
    real"""
@overload
def T3(real: Union[pd.Series, np.ndarray], timeperiod=5, vfactor=0.7) -> np.ndarray: ...
@overload
def T3(real: pd.DataFrame, timeperiod=5, vfactor=0.7) -> pd.Series: ...

"""TEMA(real[, timeperiod=?])

Triple Exponential Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def TEMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def TEMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""TRIMA(real[, timeperiod=?])

Triangular Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def TRIMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def TRIMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""VWMA(real, volume[, timeperiod=?])

Volume Weighted Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
    prices: ['volume']
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def VWMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def VWMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""WMA(real[, timeperiod=?])

Weighted Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def WMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def WMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""ZLEMA(real[, timeperiod=?])

Zero-Lag Exponential Moving Average (Overlap Studies)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def ZLEMA(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def ZLEMA(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""CDL2CROWS(open, high, low, close)

Two Crows (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDL2CROWS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDL2CROWS(real: pd.DataFrame) -> pd.Series: ...

"""CDL3BLACKCROWS(open, high, low, close)

Three Black Crows (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDL3BLACKCROWS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDL3BLACKCROWS(real: pd.DataFrame) -> pd.Series: ...

"""CDL3INSIDE(open, high, low, close)

Three Inside Up/Down (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDL3INSIDE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDL3INSIDE(real: pd.DataFrame) -> pd.Series: ...

"""CDL3LINESTRIKE(open, high, low, close)

Three-Line Strike (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDL3LINESTRIKE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDL3LINESTRIKE(real: pd.DataFrame) -> pd.Series: ...

"""CDL3OUTSIDE(open, high, low, close)

Three Outside Up/Down (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDL3OUTSIDE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDL3OUTSIDE(real: pd.DataFrame) -> pd.Series: ...

"""CDL3STARSINSOUTH(open, high, low, close)

Three Stars In The South (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDL3STARSINSOUTH(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDL3STARSINSOUTH(real: pd.DataFrame) -> pd.Series: ...

"""CDL3WHITESOLDIERS(open, high, low, close)

Three Advancing White Soldiers (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDL3WHITESOLDIERS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDL3WHITESOLDIERS(real: pd.DataFrame) -> pd.Series: ...

"""CDLABANDONEDBABY(open, high, low, close[, penetration=?])

Abandoned Baby (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Parameters:
    penetration: 0.3
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLABANDONEDBABY(real: Union[pd.Series, np.ndarray], penetration=0.3) -> np.ndarray: ...
@overload
def CDLABANDONEDBABY(real: pd.DataFrame, penetration=0.3) -> pd.Series: ...

"""CDLADVANCEBLOCK(open, high, low, close)

Advance Block (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLADVANCEBLOCK(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLADVANCEBLOCK(real: pd.DataFrame) -> pd.Series: ...

"""CDLBELTHOLD(open, high, low, close)

Belt-hold (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLBELTHOLD(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLBELTHOLD(real: pd.DataFrame) -> pd.Series: ...

"""CDLBREAKAWAY(open, high, low, close)

Breakaway (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLBREAKAWAY(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLBREAKAWAY(real: pd.DataFrame) -> pd.Series: ...

"""CDLCLOSINGMARUBOZU(open, high, low, close)

Closing Marubozu (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLCLOSINGMARUBOZU(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLCLOSINGMARUBOZU(real: pd.DataFrame) -> pd.Series: ...

"""CDLCONCEALBABYSWALL(open, high, low, close)

Concealing Baby Swallow (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLCONCEALBABYSWALL(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLCONCEALBABYSWALL(real: pd.DataFrame) -> pd.Series: ...

"""CDLCOUNTERATTACK(open, high, low, close)

Counterattack (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLCOUNTERATTACK(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLCOUNTERATTACK(real: pd.DataFrame) -> pd.Series: ...

"""CDLDARKCLOUDCOVER(open, high, low, close[, penetration=?])

Dark Cloud Cover (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Parameters:
    penetration: 0.5
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLDARKCLOUDCOVER(real: Union[pd.Series, np.ndarray], penetration=0.5) -> np.ndarray: ...
@overload
def CDLDARKCLOUDCOVER(real: pd.DataFrame, penetration=0.5) -> pd.Series: ...

"""CDLDOJI(open, high, low, close)

Doji (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLDOJI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLDOJI(real: pd.DataFrame) -> pd.Series: ...

"""CDLDOJISTAR(open, high, low, close)

Doji Star (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLDOJISTAR(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLDOJISTAR(real: pd.DataFrame) -> pd.Series: ...

"""CDLDRAGONFLYDOJI(open, high, low, close)

Dragonfly Doji (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLDRAGONFLYDOJI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLDRAGONFLYDOJI(real: pd.DataFrame) -> pd.Series: ...

"""CDLENGULFING(open, high, low, close)

Engulfing Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLENGULFING(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLENGULFING(real: pd.DataFrame) -> pd.Series: ...

"""CDLEVENINGDOJISTAR(open, high, low, close[, penetration=?])

Evening Doji Star (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Parameters:
    penetration: 0.3
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLEVENINGDOJISTAR(real: Union[pd.Series, np.ndarray], penetration=0.3) -> np.ndarray: ...
@overload
def CDLEVENINGDOJISTAR(real: pd.DataFrame, penetration=0.3) -> pd.Series: ...

"""CDLEVENINGSTAR(open, high, low, close[, penetration=?])

Evening Star (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Parameters:
    penetration: 0.3
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLEVENINGSTAR(real: Union[pd.Series, np.ndarray], penetration=0.3) -> np.ndarray: ...
@overload
def CDLEVENINGSTAR(real: pd.DataFrame, penetration=0.3) -> pd.Series: ...

"""CDLGAPSIDESIDEWHITE(open, high, low, close)

Up/Down-gap side-by-side white lines (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLGAPSIDESIDEWHITE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLGAPSIDESIDEWHITE(real: pd.DataFrame) -> pd.Series: ...

"""CDLGRAVESTONEDOJI(open, high, low, close)

Gravestone Doji (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLGRAVESTONEDOJI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLGRAVESTONEDOJI(real: pd.DataFrame) -> pd.Series: ...

"""CDLHAMMER(open, high, low, close)

Hammer (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLHAMMER(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLHAMMER(real: pd.DataFrame) -> pd.Series: ...

"""CDLHANGINGMAN(open, high, low, close)

Hanging Man (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLHANGINGMAN(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLHANGINGMAN(real: pd.DataFrame) -> pd.Series: ...

"""CDLHARAMI(open, high, low, close)

Harami Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLHARAMI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLHARAMI(real: pd.DataFrame) -> pd.Series: ...

"""CDLHARAMICROSS(open, high, low, close)

Harami Cross Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLHARAMICROSS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLHARAMICROSS(real: pd.DataFrame) -> pd.Series: ...

"""CDLHIGHWAVE(open, high, low, close)

High-Wave Candle (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLHIGHWAVE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLHIGHWAVE(real: pd.DataFrame) -> pd.Series: ...

"""CDLHIKKAKE(open, high, low, close)

Hikkake Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLHIKKAKE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLHIKKAKE(real: pd.DataFrame) -> pd.Series: ...

"""CDLHIKKAKEMOD(open, high, low, close)

Modified Hikkake Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLHIKKAKEMOD(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLHIKKAKEMOD(real: pd.DataFrame) -> pd.Series: ...

"""CDLHOMINGPIGEON(open, high, low, close)

Homing Pigeon (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLHOMINGPIGEON(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLHOMINGPIGEON(real: pd.DataFrame) -> pd.Series: ...

"""CDLIDENTICAL3CROWS(open, high, low, close)

Identical Three Crows (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLIDENTICAL3CROWS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLIDENTICAL3CROWS(real: pd.DataFrame) -> pd.Series: ...

"""CDLINNECK(open, high, low, close)

In-Neck Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLINNECK(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLINNECK(real: pd.DataFrame) -> pd.Series: ...

"""CDLINVERTEDHAMMER(open, high, low, close)

Inverted Hammer (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLINVERTEDHAMMER(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLINVERTEDHAMMER(real: pd.DataFrame) -> pd.Series: ...

"""CDLKICKING(open, high, low, close)

Kicking (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLKICKING(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLKICKING(real: pd.DataFrame) -> pd.Series: ...

"""CDLKICKINGBYLENGTH(open, high, low, close)

Kicking - bull/bear determined by the longer marubozu (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLKICKINGBYLENGTH(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLKICKINGBYLENGTH(real: pd.DataFrame) -> pd.Series: ...

"""CDLLADDERBOTTOM(open, high, low, close)

Ladder Bottom (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLLADDERBOTTOM(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLLADDERBOTTOM(real: pd.DataFrame) -> pd.Series: ...

"""CDLLONGLEGGEDDOJI(open, high, low, close)

Long Legged Doji (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLLONGLEGGEDDOJI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLLONGLEGGEDDOJI(real: pd.DataFrame) -> pd.Series: ...

"""CDLLONGLINE(open, high, low, close)

Long Line Candle (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLLONGLINE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLLONGLINE(real: pd.DataFrame) -> pd.Series: ...

"""CDLMARUBOZU(open, high, low, close)

Marubozu (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLMARUBOZU(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLMARUBOZU(real: pd.DataFrame) -> pd.Series: ...

"""CDLMATCHINGLOW(open, high, low, close)

Matching Low (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLMATCHINGLOW(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLMATCHINGLOW(real: pd.DataFrame) -> pd.Series: ...

"""CDLMATHOLD(open, high, low, close[, penetration=?])

Mat Hold (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Parameters:
    penetration: 0.5
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLMATHOLD(real: Union[pd.Series, np.ndarray], penetration=0.5) -> np.ndarray: ...
@overload
def CDLMATHOLD(real: pd.DataFrame, penetration=0.5) -> pd.Series: ...

"""CDLMORNINGDOJISTAR(open, high, low, close[, penetration=?])

Morning Doji Star (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Parameters:
    penetration: 0.3
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLMORNINGDOJISTAR(real: Union[pd.Series, np.ndarray], penetration=0.3) -> np.ndarray: ...
@overload
def CDLMORNINGDOJISTAR(real: pd.DataFrame, penetration=0.3) -> pd.Series: ...

"""CDLMORNINGSTAR(open, high, low, close[, penetration=?])

Morning Star (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Parameters:
    penetration: 0.3
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLMORNINGSTAR(real: Union[pd.Series, np.ndarray], penetration=0.3) -> np.ndarray: ...
@overload
def CDLMORNINGSTAR(real: pd.DataFrame, penetration=0.3) -> pd.Series: ...

"""CDLONNECK(open, high, low, close)

On-Neck Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLONNECK(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLONNECK(real: pd.DataFrame) -> pd.Series: ...

"""CDLPIERCING(open, high, low, close)

Piercing Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLPIERCING(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLPIERCING(real: pd.DataFrame) -> pd.Series: ...

"""CDLRICKSHAWMAN(open, high, low, close)

Rickshaw Man (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLRICKSHAWMAN(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLRICKSHAWMAN(real: pd.DataFrame) -> pd.Series: ...

"""CDLRISEFALL3METHODS(open, high, low, close)

Rising/Falling Three Methods (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLRISEFALL3METHODS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLRISEFALL3METHODS(real: pd.DataFrame) -> pd.Series: ...

"""CDLSEPARATINGLINES(open, high, low, close)

Separating Lines (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLSEPARATINGLINES(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLSEPARATINGLINES(real: pd.DataFrame) -> pd.Series: ...

"""CDLSHOOTINGSTAR(open, high, low, close)

Shooting Star (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLSHOOTINGSTAR(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLSHOOTINGSTAR(real: pd.DataFrame) -> pd.Series: ...

"""CDLSHORTLINE(open, high, low, close)

Short Line Candle (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLSHORTLINE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLSHORTLINE(real: pd.DataFrame) -> pd.Series: ...

"""CDLSPINNINGTOP(open, high, low, close)

Spinning Top (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLSPINNINGTOP(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLSPINNINGTOP(real: pd.DataFrame) -> pd.Series: ...

"""CDLSTALLEDPATTERN(open, high, low, close)

Stalled Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLSTALLEDPATTERN(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLSTALLEDPATTERN(real: pd.DataFrame) -> pd.Series: ...

"""CDLSTICKSANDWICH(open, high, low, close)

Stick Sandwich (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLSTICKSANDWICH(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLSTICKSANDWICH(real: pd.DataFrame) -> pd.Series: ...

"""CDLTAKURI(open, high, low, close)

Takuri (Dragonfly Doji with very long lower shadow) (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLTAKURI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLTAKURI(real: pd.DataFrame) -> pd.Series: ...

"""CDLTASUKIGAP(open, high, low, close)

Tasuki Gap (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLTASUKIGAP(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLTASUKIGAP(real: pd.DataFrame) -> pd.Series: ...

"""CDLTHRUSTING(open, high, low, close)

Thrusting Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLTHRUSTING(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLTHRUSTING(real: pd.DataFrame) -> pd.Series: ...

"""CDLTRISTAR(open, high, low, close)

Tristar Pattern (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLTRISTAR(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLTRISTAR(real: pd.DataFrame) -> pd.Series: ...

"""CDLUNIQUE3RIVER(open, high, low, close)

Unique 3 River (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLUNIQUE3RIVER(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLUNIQUE3RIVER(real: pd.DataFrame) -> pd.Series: ...

"""CDLUPSIDEGAP2CROWS(open, high, low, close)

Upside Gap Two Crows (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLUPSIDEGAP2CROWS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLUPSIDEGAP2CROWS(real: pd.DataFrame) -> pd.Series: ...

"""CDLXSIDEGAP3METHODS(open, high, low, close)

Upside/Downside Gap Three Methods (Pattern Recognition)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    integer (values are -100, 0 or 100)"""
@overload
def CDLXSIDEGAP3METHODS(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def CDLXSIDEGAP3METHODS(real: pd.DataFrame) -> pd.Series: ...

"""AVGDEV(real[, timeperiod=?])

Average Deviation (Price Transform)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def AVGDEV(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def AVGDEV(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""AVGPRICE(open, high, low, close)

Average Price (Price Transform)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    real"""
@overload
def AVGPRICE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def AVGPRICE(real: pd.DataFrame) -> pd.Series: ...

"""HA(open, high, low, close)

Heikin-Ashi Candles (Price Transform)

Inputs:
    prices: ['open', 'high', 'low', 'close']
Outputs:
    haopen
    hahigh
    halow
    haclose"""
@overload
def HA(real: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def HA(real: pd.DataFrame) -> pd.DataFrame: ...

"""MEDPRICE(high, low)

Median Price (Price Transform)

Inputs:
    prices: ['high', 'low']
Outputs:
    real"""
@overload
def MEDPRICE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def MEDPRICE(real: pd.DataFrame) -> pd.Series: ...

"""TYPPRICE(high, low, close)

Typical Price (Price Transform)

Inputs:
    prices: ['high', 'low', 'close']
Outputs:
    real"""
@overload
def TYPPRICE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def TYPPRICE(real: pd.DataFrame) -> pd.Series: ...

"""WCLPRICE(high, low, close)

Weighted Close Price (Price Transform)

Inputs:
    prices: ['high', 'low', 'close']
Outputs:
    real"""
@overload
def WCLPRICE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def WCLPRICE(real: pd.DataFrame) -> pd.Series: ...

"""BETA(real0, real1[, timeperiod=?])

Beta (Statistic Functions)

Inputs:
    real0: (any ndarray)
    real1: (any ndarray)
Parameters:
    timeperiod: 5
Outputs:
    real"""
@overload
def BETA(real: Union[pd.Series, np.ndarray], timeperiod=5) -> np.ndarray: ...
@overload
def BETA(real: pd.DataFrame, timeperiod=5) -> pd.Series: ...

"""CORREL(real0, real1[, timeperiod=?])

Pearson's Correlation Coefficient (r) (Statistic Functions)

Inputs:
    real0: (any ndarray)
    real1: (any ndarray)
Parameters:
    timeperiod: 30
Outputs:
    real"""
@overload
def CORREL(real: Union[pd.Series, np.ndarray], timeperiod=30) -> np.ndarray: ...
@overload
def CORREL(real: pd.DataFrame, timeperiod=30) -> pd.Series: ...

"""LINEARREG(real[, timeperiod=?])

Linear Regression (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def LINEARREG(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def LINEARREG(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""LINEARREG_ANGLE(real[, timeperiod=?])

Linear Regression Angle (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def LINEARREG_ANGLE(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def LINEARREG_ANGLE(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""LINEARREG_INTERCEPT(real[, timeperiod=?])

Linear Regression Intercept (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def LINEARREG_INTERCEPT(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def LINEARREG_INTERCEPT(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""LINEARREG_SLOPE(real[, timeperiod=?])

Linear Regression Slope (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def LINEARREG_SLOPE(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def LINEARREG_SLOPE(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""PERCENTILE(real[, timeperiod=?, percentile=?])

Percentile (nearest rank) (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 30
    percentile: 50.0
Outputs:
    real"""
@overload
def PERCENTILE(real: Union[pd.Series, np.ndarray], timeperiod=30, percentile=50.0) -> np.ndarray: ...
@overload
def PERCENTILE(real: pd.DataFrame, timeperiod=30, percentile=50.0) -> pd.Series: ...

"""PERCENTRANK(real[, timeperiod=?])

Percent Rank (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 100
Outputs:
    real"""
@overload
def PERCENTRANK(real: Union[pd.Series, np.ndarray], timeperiod=100) -> np.ndarray: ...
@overload
def PERCENTRANK(real: pd.DataFrame, timeperiod=100) -> pd.Series: ...

"""STDDEV(real[, timeperiod=?, nbdev=?])

Standard Deviation (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 5
    nbdev: 1.0
Outputs:
    real"""
@overload
def STDDEV(real: Union[pd.Series, np.ndarray], timeperiod=5, nbdev=1.0) -> np.ndarray: ...
@overload
def STDDEV(real: pd.DataFrame, timeperiod=5, nbdev=1.0) -> pd.Series: ...

"""TSF(real[, timeperiod=?])

Time Series Forecast (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def TSF(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def TSF(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""VAR(real[, timeperiod=?, nbdev=?])

Variance (Statistic Functions)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 5
    nbdev: 1.0
Outputs:
    real"""
@overload
def VAR(real: Union[pd.Series, np.ndarray], timeperiod=5, nbdev=1.0) -> np.ndarray: ...
@overload
def VAR(real: pd.DataFrame, timeperiod=5, nbdev=1.0) -> pd.Series: ...

"""ADR(high, low[, timeperiod=?])

Average Day Range (Volatility Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def ADR(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def ADR(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""ATR(high, low, close[, timeperiod=?])

Average True Range (Volatility Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def ATR(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def ATR(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""CVI(high, low[, timeperiod=?, rocperiod=?])

Chaikin's Volatility (Volatility Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    timeperiod: 10
    rocperiod: 10
Outputs:
    real"""
@overload
def CVI(real: Union[pd.Series, np.ndarray], timeperiod=10, rocperiod=10) -> np.ndarray: ...
@overload
def CVI(real: pd.DataFrame, timeperiod=10, rocperiod=10) -> pd.Series: ...

"""MASSI(high, low[, fastperiod=?, slowperiod=?])

Mass Index (Volatility Indicators)

Inputs:
    prices: ['high', 'low']
Parameters:
    fastperiod: 9
    slowperiod: 25
Outputs:
    real"""
@overload
def MASSI(real: Union[pd.Series, np.ndarray], fastperiod=9, slowperiod=25) -> np.ndarray: ...
@overload
def MASSI(real: pd.DataFrame, fastperiod=9, slowperiod=25) -> pd.Series: ...

"""NATR(high, low, close[, timeperiod=?])

Normalized Average True Range (Volatility Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Parameters:
    timeperiod: 14
Outputs:
    real"""
@overload
def NATR(real: Union[pd.Series, np.ndarray], timeperiod=14) -> np.ndarray: ...
@overload
def NATR(real: pd.DataFrame, timeperiod=14) -> pd.Series: ...

"""RVI(real[, timeperiod=?, stddevperiod=?])

Relative Volatility Index (Volatility Indicators)

Inputs:
    real: (any ndarray)
Parameters:
    timeperiod: 14
    stddevperiod: 10
Outputs:
    real"""
@overload
def RVI(real: Union[pd.Series, np.ndarray], timeperiod=14, stddevperiod=10) -> np.ndarray: ...
@overload
def RVI(real: pd.DataFrame, timeperiod=14, stddevperiod=10) -> pd.Series: ...

"""TRANGE(high, low, close)

True Range (Volatility Indicators)

Inputs:
    prices: ['high', 'low', 'close']
Outputs:
    real"""
@overload
def TRANGE(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def TRANGE(real: pd.DataFrame) -> pd.Series: ...

"""AD(high, low, close, volume)

Chaikin A/D Line (Volume Indicators)

Inputs:
    prices: ['high', 'low', 'close', 'volume']
Outputs:
    real"""
@overload
def AD(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def AD(real: pd.DataFrame) -> pd.Series: ...

"""ADOSC(high, low, close, volume[, fastperiod=?, slowperiod=?])

Chaikin A/D Oscillator (Volume Indicators)

Inputs:
    prices: ['high', 'low', 'close', 'volume']
Parameters:
    fastperiod: 3
    slowperiod: 10
Outputs:
    real"""
@overload
def ADOSC(real: Union[pd.Series, np.ndarray], fastperiod=3, slowperiod=10) -> np.ndarray: ...
@overload
def ADOSC(real: pd.DataFrame, fastperiod=3, slowperiod=10) -> pd.Series: ...

"""CMF(high, low, close, volume[, timeperiod=?])

Chaikin Money Flow (Volume Indicators)

Inputs:
    prices: ['high', 'low', 'close', 'volume']
Parameters:
    timeperiod: 20
Outputs:
    real"""
@overload
def CMF(real: Union[pd.Series, np.ndarray], timeperiod=20) -> np.ndarray: ...
@overload
def CMF(real: pd.DataFrame, timeperiod=20) -> pd.Series: ...

"""EFI(close, volume[, timeperiod=?])

Elder's Force Index (Volume Indicators)

Inputs:
    prices: ['close', 'volume']
Parameters:
    timeperiod: 13
Outputs:
    real"""
@overload
def EFI(real: Union[pd.Series, np.ndarray], timeperiod=13) -> np.ndarray: ...
@overload
def EFI(real: pd.DataFrame, timeperiod=13) -> pd.Series: ...

"""MARKETFI(high, low, volume)

Market Facilitation Index (Volume Indicators)

Inputs:
    prices: ['high', 'low', 'volume']
Outputs:
    real"""
@overload
def MARKETFI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def MARKETFI(real: pd.DataFrame) -> pd.Series: ...

"""NVI(close, volume)

Negative Volume Index (Volume Indicators)

Inputs:
    prices: ['close', 'volume']
Outputs:
    real"""
@overload
def NVI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def NVI(real: pd.DataFrame) -> pd.Series: ...

"""OBV(real, volume)

On Balance Volume (Volume Indicators)

Inputs:
    real: (any ndarray)
    prices: ['volume']
Outputs:
    real"""
@overload
def OBV(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def OBV(real: pd.DataFrame) -> pd.Series: ...

"""PVI(close, volume)

Positive Volume Index (Volume Indicators)

Inputs:
    prices: ['close', 'volume']
Outputs:
    real"""
@overload
def PVI(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def PVI(real: pd.DataFrame) -> pd.Series: ...

"""PVO(volume[, fastperiod=?, slowperiod=?, matype=?])

Percentage Volume Oscillator (Volume Indicators)

Inputs:
    prices: ['volume']
Parameters:
    fastperiod: 12
    slowperiod: 26
    matype: 1 (Exponential Moving Average)
Outputs:
    real"""
@overload
def PVO(real: Union[pd.Series, np.ndarray], fastperiod=12, slowperiod=26, matype=1) -> np.ndarray: ...
@overload
def PVO(real: pd.DataFrame, fastperiod=12, slowperiod=26, matype=1) -> pd.Series: ...

"""PVT(close, volume)

Price Volume Trend (Volume Indicators)

Inputs:
    prices: ['close', 'volume']
Outputs:
    real"""
@overload
def PVT(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def PVT(real: pd.DataFrame) -> pd.Series: ...

"""RVOL(volume[, timeperiod=?])

Relative Volume (Volume Indicators)

Inputs:
    prices: ['volume']
Parameters:
    timeperiod: 20
Outputs:
    real"""
@overload
def RVOL(real: Union[pd.Series, np.ndarray], timeperiod=20) -> np.ndarray: ...
@overload
def RVOL(real: pd.DataFrame, timeperiod=20) -> pd.Series: ...

"""VWAP(high, low, close, volume)

Volume Weighted Average Price (Volume Indicators)

Inputs:
    prices: ['high', 'low', 'close', 'volume']
Outputs:
    real"""
@overload
def VWAP(real: Union[pd.Series, np.ndarray]) -> np.ndarray: ...
@overload
def VWAP(real: pd.DataFrame) -> pd.Series: ...


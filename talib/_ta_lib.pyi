import numpy as np
import polars as pl
import pandas as pd
from enum import Enum
from typing import Tuple, Union
from numpy.typing import NDArray

class MA_Type(Enum):
    SMA = 0
    EMA = 1
    WMA = 2
    DEMA = 3
    TEMA = 4
    TRIMA = 5
    KAMA = 6
    MAMA = 7
    T3 = 8

#Overlap Studies Functions

def BBANDS(
        real: Union[NDArray[np.float64], pd.Series, pl.Series],  
        timeperiod: int= 5, 
        nbdevup: float= 2, 
        nbdevdn: float= 2, 
        matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def DEMA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def EMA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def HT_TRENDLINE(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def KAMA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def MA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30,
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def MAMA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        fastlimit: float= 0, 
        slowlimit: float= 0
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def MAVP(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        periods: float, 
        minperiod: int= 2, 
        maxperiod: int= 30, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def MIDPOINT(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MIDPRICE(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def SAR(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        acceleration: float= 0, 
        maximum: float= 0
        )-> NDArray[np.float64]: ...

def SAREXT(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        startvalue: float= 0, 
        offsetonreverse: float= 0, 
        accelerationinitlong: float= 0, 
        accelerationlong: float= 0, 
        accelerationmaxlong: float= 0, 
        accelerationinitshort: float= 0, 
        accelerationshort: float= 0, 
        accelerationmaxshort: float= 0
        )-> NDArray[np.float64]: ...

def SMA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def T3(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 5, 
        vfactor: float= 0
        )-> NDArray[np.float64]: ...

def TEMA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def TRIMA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def WMA(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

#Momentum Indicator Functions

def ADX(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def ADXR(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def APO(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def AROON(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def AROONOSC(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def BOP(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CCI(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def CMO(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def DX(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MACD(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        signalperiod: int= 9
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def MACDEXT(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        fastperiod: int= 12, 
        fastmatype: MA_Type = MA_Type.SMA, 
        slowperiod: int= 26, 
        slowmatype: MA_Type = MA_Type.SMA, 
        signalperiod: int= 9, 
        signalmatype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def MACDFIX(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        signalperiod: int= 9
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def MFI(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        volume: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MINUS_DI(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MINUS_DM(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MOM(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def PLUS_DI(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def PLUS_DM(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def PPO(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def ROC(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def ROCP(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def ROCR(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def ROCR100(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def RSI(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def STOCH(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        fastk_period: int= 5, 
        slowk_period: int= 3, 
        slowk_matype: MA_Type = MA_Type.SMA, 
        slowd_period: int= 3, 
        slowd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def STOCHF(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        fastk_period: int= 5, 
        fastd_period: int= 3, 
        fastd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def STOCHRSI(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14, 
        fastk_period: int= 5, 
        fastd_period: int= 3, 
        fastd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def TRIX(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def ULTOSC(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod1: int= 7, 
        timeperiod2: int= 14, 
        timeperiod3: int= 28
        )-> NDArray[np.float64]: ...

def WILLR(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

#Volume Indicator Functions

def AD(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        volume: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def ADOSC(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        volume: Union[NDArray[np.float64], pd.Series, pl.Series], 
        fastperiod: int= 3, 
        slowperiod: int= 10
        )-> NDArray[np.float64]: ...

def OBV(
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        volume: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

#Volatility Indicator Functions

def ATR(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def NATR(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def TRANGE(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

# Price Transform Functions

def AVGPRICE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def MEDPRICE(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def TYPPRICE(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def WCLPRICE(
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

# Cycle Indicator Functions

def HT_DCPERIOD(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def HT_DCPHASE(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def HT_PHASOR(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def HT_SINE(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def HT_TRENDMODE(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

#Pattern Recognition Functions

def CDL2CROWS(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDL3BLACKCROWS(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDL3INSIDE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDL3LINESTRIKE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDL3OUTSIDE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDL3STARSINSOUTH(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDL3WHITESOLDIERS(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLABANDONEDBABY(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series],
        penetration: float= 0
        )-> NDArray[np.float64]: ...

def CDLADVANCEBLOCK(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLBELTHOLD(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLBREAKAWAY(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLCLOSINGMARUBOZU(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLCONCEALBABYSWALL(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLCOUNTERATTACK(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLDARKCLOUDCOVER(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series],
        penetration: float= 0
        )-> NDArray[np.float64]: ...

def CDLDOJI(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLDOJISTAR(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLDRAGONFLYDOJI(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLENGULFING(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLEVENINGDOJISTAR(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series],
        penetration: float= 0
        )-> NDArray[np.float64]: ...

def CDLEVENINGSTAR(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series],
        penetration: float= 0
        )-> NDArray[np.float64]: ...

def CDLGAPSIDESIDEWHITE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLGRAVESTONEDOJI(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLHAMMER(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLHANGINGMAN(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLHARAMI(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLHARAMICROSS(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLHIGHWAVE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLHIKKAKE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLHIKKAKEMOD(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLHOMINGPIGEON(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLIDENTICAL3CROWS(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLINNECK(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLINVERTEDHAMMER(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLKICKING(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLKICKINGBYLENGTH(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLLADDERBOTTOM(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLLONGLEGGEDDOJI(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLLONGLINE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLMARUBOZU(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLMATCHINGLOW(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLMATHOLD(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series],
        penetration: float= 0
        )-> NDArray[np.float64]: ...

def CDLMORNINGDOJISTAR(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series],
        penetration: float= 0
        )-> NDArray[np.float64]: ...

def CDLMORNINGSTAR(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series],
        penetration: float= 0
        )-> NDArray[np.float64]: ...

def CDLONNECK(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLPIERCING(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLRICKSHAWMAN(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLRISEFALL3METHODS(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLSEPARATINGLINES(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLSHOOTINGSTAR(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLSHORTLINE(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLSPINNINGTOP(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLSTALLEDPATTERN(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLSTICKSANDWICH(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLTAKURI(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLTASUKIGAP(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLTHRUSTING(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLTRISTAR(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLUNIQUE3RIVER(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLUPSIDEGAP2CROWS(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def CDLXSIDEGAP3METHODS(
        open: Union[NDArray[np.float64], pd.Series, pl.Series], 
        high: Union[NDArray[np.float64], pd.Series, pl.Series], 
        low: Union[NDArray[np.float64], pd.Series, pl.Series], 
        close: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

# Statistic Functions

def BETA(
        real0: Union[NDArray[np.float64], pd.Series, pl.Series], 
        real1: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 5
        )-> NDArray[np.float64]: ...

def CORREL(
        real0: Union[NDArray[np.float64], pd.Series, pl.Series], 
        real1: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def LINEARREG(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def LINEARREG_ANGLE(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def LINEARREG_INTERCEPT(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def LINEARREG_SLOPE(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def STDDEV(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 5, 
        nbdev: float= 1
        )-> NDArray[np.float64]: ...

def TSF(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def VAR(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 5, 
        nbdev: float= 1
        )-> NDArray[np.float64]: ...

# Math Transform Functions

def ACOS(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def ASIN(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def ATAN(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def CEIL(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def COS(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def COSH(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def EXP(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def FLOOR(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def LN(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def LOG10(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def SIN(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def SINH(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def SQRT(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def TAN(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

def TANH(real: Union[NDArray[np.float64], pd.Series, pl.Series])-> NDArray[np.float64]: ...

#Math Operator Functions

def ADD(
        real0: Union[NDArray[np.float64], pd.Series, pl.Series], 
        real1: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def DIV(
        real0: Union[NDArray[np.float64], pd.Series, pl.Series], 
        real1: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def MAX(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def MAXINDEX(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def MIN(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def MININDEX(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def MINMAX(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def MINMAXINDEX(
        real: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def MULT(
        real0: Union[NDArray[np.float64], pd.Series, pl.Series], 
        real1: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def SUB(
        real0: Union[NDArray[np.float64], pd.Series, pl.Series], 
        real1: Union[NDArray[np.float64], pd.Series, pl.Series]
        )-> NDArray[np.float64]: ...

def SUM(
        real0: Union[NDArray[np.float64], pd.Series, pl.Series], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...
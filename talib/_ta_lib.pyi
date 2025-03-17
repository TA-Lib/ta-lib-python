import numpy as np
from enum import Enum
from typing import Tuple
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
        real: NDArray[np.float64],  
        timeperiod: int= 5, 
        nbdevup: float= 2, 
        nbdevdn: float= 2, 
        matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def DEMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def EMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def HT_TRENDLINE(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def KAMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def MA(
        real: NDArray[np.float64], 
        timeperiod: int= 30,
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def MAMA(
        real: NDArray[np.float64], 
        fastlimit: float= 0, 
        slowlimit: float= 0
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def MAVP(
        real: NDArray[np.float64], 
        periods: float, 
        minperiod: int= 2, 
        maxperiod: int= 30, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def MIDPOINT(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MIDPRICE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def SAR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        acceleration: float= 0, 
        maximum: float= 0
        )-> NDArray[np.float64]: ...

def SAREXT(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
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
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def T3(
        real: NDArray[np.float64], 
        timeperiod: int= 5, 
        vfactor: float= 0
        )-> NDArray[np.float64]: ...

def TEMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def TRIMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def WMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

#Momentum Indicator Functions

def ADX(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def ADXR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def APO(
        real: NDArray[np.float64], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def AROON(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def AROONOSC(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def BOP(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def CCI(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def CMO(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def DX(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MACD(
        real: NDArray[np.float64], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        signalperiod: int= 9
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def MACDEXT(
        real: NDArray[np.float64], 
        fastperiod: int= 12, 
        fastmatype: MA_Type = MA_Type.SMA, 
        slowperiod: int= 26, 
        slowmatype: MA_Type = MA_Type.SMA, 
        signalperiod: int= 9, 
        signalmatype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def MACDFIX(
        real: NDArray[np.float64], 
        signalperiod: int= 9
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def MFI(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        volume: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MINUS_DI(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MINUS_DM(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def MOM(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def PLUS_DI(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def PLUS_DM(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def PPO(
        real: NDArray[np.float64], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def ROC(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def ROCP(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def ROCR(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def ROCR100(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def RSI(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def STOCH(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        fastk_period: int= 5, 
        slowk_period: int= 3, 
        slowk_matype: MA_Type = MA_Type.SMA, 
        slowd_period: int= 3, 
        slowd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def STOCHF(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        fastk_period: int= 5, 
        fastd_period: int= 3, 
        fastd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def STOCHRSI(
        real: NDArray[np.float64], 
        timeperiod: int= 14, 
        fastk_period: int= 5, 
        fastd_period: int= 3, 
        fastd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def TRIX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def ULTOSC(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod1: int= 7, 
        timeperiod2: int= 14, 
        timeperiod3: int= 28
        )-> NDArray[np.float64]: ...

def WILLR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

#Volume Indicator Functions

def AD(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        volume: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def ADOSC(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        volume: NDArray[np.float64], 
        fastperiod: int= 3, 
        slowperiod: int= 10
        )-> NDArray[np.float64]: ...

def OBV(
        close: NDArray[np.float64], 
        volume: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

#Volatility Indicator Functions

def ATR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def NATR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def TRANGE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

# Price Transform Functions

def AVGPRICE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def MEDPRICE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def TYPPRICE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def WCLPRICE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

# Cycle Indicator Functions

def HT_DCPERIOD(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def HT_DCPHASE(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def HT_PHASOR(real: NDArray[np.float64])-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def HT_SINE(real: NDArray[np.float64])-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def HT_TRENDMODE(real: NDArray[np.float64])-> Tuple[NDArray[np.float64], NDArray[np.int32]]: ...

#Pattern Recognition Functions

def CDL2CROWS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDL3BLACKCROWS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDL3INSIDE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDL3LINESTRIKE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDL3OUTSIDE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDL3STARSINSOUTH(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDL3WHITESOLDIERS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLABANDONEDBABY(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def CDLADVANCEBLOCK(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLBELTHOLD(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLBREAKAWAY(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLCLOSINGMARUBOZU(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLCONCEALBABYSWALL(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLCOUNTERATTACK(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLDARKCLOUDCOVER(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def CDLDOJI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLDOJISTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLDRAGONFLYDOJI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLENGULFING(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLEVENINGDOJISTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def CDLEVENINGSTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def CDLGAPSIDESIDEWHITE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLGRAVESTONEDOJI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLHAMMER(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLHANGINGMAN(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLHARAMI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLHARAMICROSS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLHIGHWAVE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLHIKKAKE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLHIKKAKEMOD(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLHOMINGPIGEON(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLIDENTICAL3CROWS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLINNECK(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLINVERTEDHAMMER(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLKICKING(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLKICKINGBYLENGTH(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLLADDERBOTTOM(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLLONGLEGGEDDOJI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLLONGLINE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLMARUBOZU(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLMATCHINGLOW(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLMATHOLD(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def CDLMORNINGDOJISTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def CDLMORNINGSTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def CDLONNECK(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLPIERCING(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLRICKSHAWMAN(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLRISEFALL3METHODS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLSEPARATINGLINES(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLSHOOTINGSTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLSHORTLINE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLSPINNINGTOP(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLSTALLEDPATTERN(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLSTICKSANDWICH(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLTAKURI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLTASUKIGAP(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLTHRUSTING(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLTRISTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLUNIQUE3RIVER(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLUPSIDEGAP2CROWS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def CDLXSIDEGAP3METHODS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

# Statistic Functions

def BETA(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64], 
        timeperiod: int= 5
        )-> NDArray[np.float64]: ...

def CORREL(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def LINEARREG(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def LINEARREG_ANGLE(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def LINEARREG_INTERCEPT(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def LINEARREG_SLOPE(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def STDDEV(
        real: NDArray[np.float64], 
        timeperiod: int= 5, 
        nbdev: float= 1
        )-> NDArray[np.float64]: ...

def TSF(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def VAR(
        real: NDArray[np.float64], 
        timeperiod: int= 5, 
        nbdev: float= 1
        )-> NDArray[np.float64]: ...

# Math Transform Functions

def ACOS(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def ASIN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def ATAN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def CEIL(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def COS(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def COSH(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def EXP(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def FLOOR(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def LN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def LOG10(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def SIN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def SINH(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def SQRT(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def TAN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def TANH(real: NDArray[np.float64])-> NDArray[np.float64]: ...

#Math Operator Functions

def ADD(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def DIV(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def MAX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def MAXINDEX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.int32]: ...

def MIN(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def MININDEX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.int32]: ...

def MINMAX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def MINMAXINDEX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def MULT(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def SUB(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def SUM(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def ACCBANDS(
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        timeperiod: int= 20
        )-> NDArray[np.float64]: ...

def AVGDEV(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def IMI(
        open: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

#Overlap Studies Functions

def stream_BBANDS(
        real: NDArray[np.float64],  
        timeperiod: int= 5, 
        nbdevup: float= 2, 
        nbdevdn: float= 2, 
        matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def stream_DEMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_EMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_HT_TRENDLINE(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_KAMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_MA(
        real: NDArray[np.float64], 
        timeperiod: int= 30,
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def stream_MAMA(
        real: NDArray[np.float64], 
        fastlimit: float= 0, 
        slowlimit: float= 0
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_MAVP(
        real: NDArray[np.float64], 
        periods: float, 
        minperiod: int= 2, 
        maxperiod: int= 30, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def stream_MIDPOINT(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_MIDPRICE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_SAR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        acceleration: float= 0, 
        maximum: float= 0
        )-> NDArray[np.float64]: ...

def stream_SAREXT(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        startvalue: float= 0, 
        offsetonreverse: float= 0, 
        accelerationinitlong: float= 0, 
        accelerationlong: float= 0, 
        accelerationmaxlong: float= 0, 
        accelerationinitshort: float= 0, 
        accelerationshort: float= 0, 
        accelerationmaxshort: float= 0
        )-> NDArray[np.float64]: ...

def stream_SMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_T3(
        real: NDArray[np.float64], 
        timeperiod: int= 5, 
        vfactor: float= 0
        )-> NDArray[np.float64]: ...

def stream_TEMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_TRIMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_WMA(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

#Momentum Indicator Functions

def stream_ADX(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_ADXR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_APO(
        real: NDArray[np.float64], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def stream_AROON(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_AROONOSC(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_BOP(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_CCI(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_CMO(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_DX(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_MACD(
        real: NDArray[np.float64], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        signalperiod: int= 9
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def stream_MACDEXT(
        real: NDArray[np.float64], 
        fastperiod: int= 12, 
        fastmatype: MA_Type = MA_Type.SMA, 
        slowperiod: int= 26, 
        slowmatype: MA_Type = MA_Type.SMA, 
        signalperiod: int= 9, 
        signalmatype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def stream_MACDFIX(
        real: NDArray[np.float64], 
        signalperiod: int= 9
        )-> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def stream_MFI(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        volume: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_MINUS_DI(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_MINUS_DM(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_MOM(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def stream_PLUS_DI(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_PLUS_DM(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_PPO(
        real: NDArray[np.float64], 
        fastperiod: int= 12, 
        slowperiod: int= 26, 
        matype: MA_Type = MA_Type.SMA
        )-> NDArray[np.float64]: ...

def stream_ROC(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def stream_ROCP(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def stream_ROCR(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def stream_ROCR100(
        real: NDArray[np.float64], 
        timeperiod: int= 10
        )-> NDArray[np.float64]: ...

def stream_RSI(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_STOCH(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        fastk_period: int= 5, 
        slowk_period: int= 3, 
        slowk_matype: MA_Type = MA_Type.SMA, 
        slowd_period: int= 3, 
        slowd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_STOCHF(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        fastk_period: int= 5, 
        fastd_period: int= 3, 
        fastd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_STOCHRSI(
        real: NDArray[np.float64], 
        timeperiod: int= 14, 
        fastk_period: int= 5, 
        fastd_period: int= 3, 
        fastd_matype: MA_Type = MA_Type.SMA
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_TRIX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_ULTOSC(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod1: int= 7, 
        timeperiod2: int= 14, 
        timeperiod3: int= 28
        )-> NDArray[np.float64]: ...

def stream_WILLR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

#Volume Indicator Functions

def stream_AD(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        volume: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_ADOSC(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        volume: NDArray[np.float64], 
        fastperiod: int= 3, 
        slowperiod: int= 10
        )-> NDArray[np.float64]: ...

def stream_OBV(
        close: NDArray[np.float64], 
        volume: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

#Volatility Indicator Functions

def stream_ATR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_NATR(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_TRANGE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

# Price Transform Functions

def stream_AVGPRICE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_MEDPRICE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_TYPPRICE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_WCLPRICE(
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

# Cycle Indicator Functions

def stream_HT_DCPERIOD(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_HT_DCPHASE(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_HT_PHASOR(real: NDArray[np.float64])-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_HT_SINE(real: NDArray[np.float64])-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_HT_TRENDMODE(real: NDArray[np.float64])-> Tuple[NDArray[np.float64], NDArray[np.int32]]: ...

#Pattern Recognition Functions

def stream_CDL2CROWS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDL3BLACKCROWS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDL3INSIDE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDL3LINESTRIKE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDL3OUTSIDE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDL3STARSINSOUTH(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDL3WHITESOLDIERS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLABANDONEDBABY(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def stream_CDLADVANCEBLOCK(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLBELTHOLD(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLBREAKAWAY(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLCLOSINGMARUBOZU(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLCONCEALBABYSWALL(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLCOUNTERATTACK(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLDARKCLOUDCOVER(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def stream_CDLDOJI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLDOJISTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLDRAGONFLYDOJI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLENGULFING(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLEVENINGDOJISTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def stream_CDLEVENINGSTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def stream_CDLGAPSIDESIDEWHITE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLGRAVESTONEDOJI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLHAMMER(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLHANGINGMAN(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLHARAMI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLHARAMICROSS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLHIGHWAVE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLHIKKAKE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLHIKKAKEMOD(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLHOMINGPIGEON(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLIDENTICAL3CROWS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLINNECK(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLINVERTEDHAMMER(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLKICKING(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLKICKINGBYLENGTH(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLLADDERBOTTOM(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLLONGLEGGEDDOJI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLLONGLINE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLMARUBOZU(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLMATCHINGLOW(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLMATHOLD(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def stream_CDLMORNINGDOJISTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def stream_CDLMORNINGSTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64],
        penetration: float= 0
        )-> NDArray[np.int32]: ...

def stream_CDLONNECK(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLPIERCING(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLRICKSHAWMAN(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLRISEFALL3METHODS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLSEPARATINGLINES(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLSHOOTINGSTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLSHORTLINE(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLSPINNINGTOP(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLSTALLEDPATTERN(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLSTICKSANDWICH(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLTAKURI(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLTASUKIGAP(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLTHRUSTING(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLTRISTAR(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLUNIQUE3RIVER(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLUPSIDEGAP2CROWS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

def stream_CDLXSIDEGAP3METHODS(
        open: NDArray[np.float64], 
        high: NDArray[np.float64], 
        low: NDArray[np.float64], 
        close: NDArray[np.float64]
        )-> NDArray[np.int32]: ...

# Statistic Functions

def stream_BETA(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64], 
        timeperiod: int= 5
        )-> NDArray[np.float64]: ...

def stream_CORREL(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_LINEARREG(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_LINEARREG_ANGLE(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_LINEARREG_INTERCEPT(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_LINEARREG_SLOPE(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_STDDEV(
        real: NDArray[np.float64], 
        timeperiod: int= 5, 
        nbdev: float= 1
        )-> NDArray[np.float64]: ...

def stream_TSF(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_VAR(
        real: NDArray[np.float64], 
        timeperiod: int= 5, 
        nbdev: float= 1
        )-> NDArray[np.float64]: ...

# Math Transform Functions

def stream_ACOS(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_ASIN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_ATAN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_CEIL(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_COS(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_COSH(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_EXP(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_FLOOR(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_LN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_LOG10(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_SIN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_SINH(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_SQRT(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_TAN(real: NDArray[np.float64])-> NDArray[np.float64]: ...

def stream_TANH(real: NDArray[np.float64])-> NDArray[np.float64]: ...

#Math Operator Functions

def stream_ADD(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_DIV(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_MAX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_MAXINDEX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.int32]: ...

def stream_MIN(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_MININDEX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.int32]: ...

def stream_MINMAX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_MINMAXINDEX(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def stream_MULT(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_SUB(
        real0: NDArray[np.float64], 
        real1: NDArray[np.float64]
        )-> NDArray[np.float64]: ...

def stream_SUM(
        real: NDArray[np.float64], 
        timeperiod: int= 30
        )-> NDArray[np.float64]: ...

def stream_ACCBANDS(
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        timeperiod: int= 20
        )-> NDArray[np.float64]: ...

def stream_AVGDEV(
        real: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

def stream_IMI(
        open: NDArray[np.float64], 
        close: NDArray[np.float64], 
        timeperiod: int= 14
        )-> NDArray[np.float64]: ...

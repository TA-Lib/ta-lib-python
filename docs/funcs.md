
## Overlap Studies Functions
```
upperband, middleband, lowerband = BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
real = DEMA(close, timeperiod=30)
real = EMA(close, timeperiod=30)
real = HT_TRENDLINE(close)
real = KAMA(close, timeperiod=30)
real = MA(close, timeperiod=30, matype=0)
mama, fama = MAMA(close, fastlimit=0, slowlimit=0)
real = MAVP(close, minperiod=2, maxperiod=30, matype=0)
real = MIDPOINT(close, timeperiod=14)
real = MIDPRICE(high, low, timeperiod=14)
real = SAR(high, low, acceleration=0, maximum=0)
real = SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
real = SMA(close, timeperiod=30)
real = T3(close, timeperiod=5, vfactor=0)
real = TEMA(close, timeperiod=30)
real = TRIMA(close, timeperiod=30)
real = WMA(close, timeperiod=30)
```

## Momentum Indicators Functions
```
real = ADX(high, low, close, timeperiod=14)
real = ADXR(high, low, close, timeperiod=14)
real = APO(close, fastperiod=12, slowperiod=26, matype=0)
aroondown, aroonup = AROON(high, low, timeperiod=14)
real = AROONOSC(high, low, timeperiod=14)
real = BOP(high, low, close)
real = CCI(high, low, close, timeperiod=14)
real = CMO(close, timeperiod=14)
real = DX(high, low, close, timeperiod=14)
macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
macd, macdsignal, macdhist = MACDFIX(close, signalperiod=9)
real = MFI(high, low, close, volume, timeperiod=14)
real = MINUS_DI(high, low, close, timeperiod=14)
real = MINUS_DM(high, low, timeperiod=14)
real = MOM(close, timeperiod=10)
real = PLUS_DI(high, low, close, timeperiod=14)
real = PLUS_DM(high, low, timeperiod=14)
real = PPO(close, fastperiod=12, slowperiod=26, matype=0)
real = ROC(close, timeperiod=10)
real = ROCP(close, timeperiod=10)
real = ROCR(close, timeperiod=10)
real = ROCR100(close, timeperiod=10)
real = RSI(close, timeperiod=14)
slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
fastk, fastd = STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
real = TRIX(close, timeperiod=30)
real = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
real = WILLR(high, low, close, timeperiod=14)
```

## Volume Indicators Functions
```
real = AD(high, low, close, volume)
real = ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
real = OBV(volume)
```

## Volatility Indicators Functions
```
real = ATR(high, low, close, timeperiod=14)
real = NATR(high, low, close, timeperiod=14)
real = TRANGE(high, low, close)
```

## Pattern Recognition Functions
```
integer = CDL2CROWS(high, low, close)
integer = CDL3BLACKCROWS(high, low, close)
integer = CDL3INSIDE(high, low, close)
integer = CDL3LINESTRIKE(high, low, close)
integer = CDL3OUTSIDE(high, low, close)
integer = CDL3STARSINSOUTH(high, low, close)
integer = CDL3WHITESOLDIERS(high, low, close)
integer = CDLABANDONEDBABY(high, low, close, penetration=0)
integer = CDLADVANCEBLOCK(high, low, close)
integer = CDLBELTHOLD(high, low, close)
integer = CDLBREAKAWAY(high, low, close)
integer = CDLCLOSINGMARUBOZU(high, low, close)
integer = CDLCONCEALBABYSWALL(high, low, close)
integer = CDLCOUNTERATTACK(high, low, close)
integer = CDLDARKCLOUDCOVER(high, low, close, penetration=0)
integer = CDLDOJI(high, low, close)
integer = CDLDOJISTAR(high, low, close)
integer = CDLDRAGONFLYDOJI(high, low, close)
integer = CDLENGULFING(high, low, close)
integer = CDLEVENINGDOJISTAR(high, low, close, penetration=0)
integer = CDLEVENINGSTAR(high, low, close, penetration=0)
integer = CDLGAPSIDESIDEWHITE(high, low, close)
integer = CDLGRAVESTONEDOJI(high, low, close)
integer = CDLHAMMER(high, low, close)
integer = CDLHANGINGMAN(high, low, close)
integer = CDLHARAMI(high, low, close)
integer = CDLHARAMICROSS(high, low, close)
integer = CDLHIGHWAVE(high, low, close)
integer = CDLHIKKAKE(high, low, close)
integer = CDLHIKKAKEMOD(high, low, close)
integer = CDLHOMINGPIGEON(high, low, close)
integer = CDLIDENTICAL3CROWS(high, low, close)
integer = CDLINNECK(high, low, close)
integer = CDLINVERTEDHAMMER(high, low, close)
integer = CDLKICKING(high, low, close)
integer = CDLKICKINGBYLENGTH(high, low, close)
integer = CDLLADDERBOTTOM(high, low, close)
integer = CDLLONGLEGGEDDOJI(high, low, close)
integer = CDLLONGLINE(high, low, close)
integer = CDLMARUBOZU(high, low, close)
integer = CDLMATCHINGLOW(high, low, close)
integer = CDLMATHOLD(high, low, close, penetration=0)
integer = CDLMORNINGDOJISTAR(high, low, close, penetration=0)
integer = CDLMORNINGSTAR(high, low, close, penetration=0)
integer = CDLONNECK(high, low, close)
integer = CDLPIERCING(high, low, close)
integer = CDLRICKSHAWMAN(high, low, close)
integer = CDLRISEFALL3METHODS(high, low, close)
integer = CDLSEPARATINGLINES(high, low, close)
integer = CDLSHOOTINGSTAR(high, low, close)
integer = CDLSHORTLINE(high, low, close)
integer = CDLSPINNINGTOP(high, low, close)
integer = CDLSTALLEDPATTERN(high, low, close)
integer = CDLSTICKSANDWICH(high, low, close)
integer = CDLTAKURI(high, low, close)
integer = CDLTASUKIGAP(high, low, close)
integer = CDLTHRUSTING(high, low, close)
integer = CDLTRISTAR(high, low, close)
integer = CDLUNIQUE3RIVER(high, low, close)
integer = CDLUPSIDEGAP2CROWS(high, low, close)
integer = CDLXSIDEGAP3METHODS(high, low, close)
```

## Cycle Indicators Functions
```
real = HT_DCPERIOD(close)
real = HT_DCPHASE(close)
inphase, quadrature = HT_PHASOR(close)
sine, leadsine = HT_SINE(close)
integer = HT_TRENDMODE(close)
```

## Statistic Functions Functions
```
real = BETA(high, low, timeperiod=5)
real = CORREL(high, low, timeperiod=30)
real = LINEARREG(close, timeperiod=14)
real = LINEARREG_ANGLE(close, timeperiod=14)
real = LINEARREG_INTERCEPT(close, timeperiod=14)
real = LINEARREG_SLOPE(close, timeperiod=14)
real = STDDEV(close, timeperiod=5, nbdev=1)
real = TSF(close, timeperiod=14)
real = VAR(close, timeperiod=5, nbdev=1)
```

## Price Transform Functions
```
real = AVGPRICE(high, low, close)
real = MEDPRICE(high, low)
real = TYPPRICE(high, low, close)
real = WCLPRICE(high, low, close)
```

## Math Transform Functions
```
real = ACOS(close)
real = ASIN(close)
real = ATAN(close)
real = CEIL(close)
real = COS(close)
real = COSH(close)
real = EXP(close)
real = FLOOR(close)
real = LN(close)
real = LOG10(close)
real = SIN(close)
real = SINH(close)
real = SQRT(close)
real = TAN(close)
real = TANH(close)
```

## Math Operators Functions
```
real = ADD(high, low)
real = DIV(high, low)
real = MAX(close, timeperiod=30)
integer = MAXINDEX(close, timeperiod=30)
real = MIN(close, timeperiod=30)
integer = MININDEX(close, timeperiod=30)
min, max = MINMAX(close, timeperiod=30)
minidx, maxidx = MINMAXINDEX(close, timeperiod=30)
real = MULT(high, low)
real = SUB(high, low)
real = SUM(close, timeperiod=30)
```

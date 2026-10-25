# Momentum Indicator Functions
### AC - Accelerator/Decelerator Oscillator
```python
real = AC(high, low, fastperiod=5, slowperiod=34, signalperiod=5)
```

### ADX - Average Directional Movement Index
NOTE: The ``ADX`` function has an unstable period.  
```python
real = ADX(high, low, close, timeperiod=14)
```

### ADXR - Average Directional Movement Index Rating
```python
real = ADXR(high, low, close, timeperiod=14)
```

### AO - Awesome Oscillator
```python
real = AO(high, low, fastperiod=5, slowperiod=34)
```

### APO - Absolute Price Oscillator
```python
real = APO(close, fastperiod=12, slowperiod=26, matype=1)
```

### AROON - Aroon
```python
aroondown, aroonup = AROON(high, low, timeperiod=14)
```

### AROONOSC - Aroon Oscillator
```python
real = AROONOSC(high, low, timeperiod=14)
```

### ASI - Wilder Accumulative Swing Index
```python
real = ASI(open, high, low, close, limitmove=3)
```

### BOP - Balance Of Power
```python
real = BOP(open, high, low, close)
```

### CCI - Commodity Channel Index
```python
real = CCI(high, low, close, timeperiod=14)
```

### CG - Center of Gravity Oscillator
```python
real = CG(close, timeperiod=10)
```

### CHOP - Choppiness Index
```python
real = CHOP(high, low, close, timeperiod=14)
```

### CHOPTR - Choppiness Index (True Range Box)
```python
real = CHOPTR(high, low, close, timeperiod=14)
```

### CMO - Chande Momentum Oscillator
NOTE: The ``CMO`` function has an unstable period.  
```python
real = CMO(close, timeperiod=14)
```

### CMOU - Chande Momentum Oscillator (Unsmoothed)
```python
real = CMOU(close, timeperiod=14)
```

### COPPOCK - Coppock Curve
```python
real = COPPOCK(close, wmaperiod=10, roc1period=11, roc2period=14)
```

### CRSI - Connors Relative Strength Index
```python
real = CRSI(close, timeperiod=3, streakperiod=2, rankperiod=100)
```

### CTI - Correlation Trend Indicator
```python
real = CTI(close, timeperiod=20)
```

### DOSC - Derivative Oscillator
```python
real = DOSC(close, timeperiod=14, firstperiod=5, secondperiod=3, signalperiod=9)
```

### DPO - Detrended Price Oscillator
```python
real = DPO(close, timeperiod=20)
```

### DX - Directional Movement Index
NOTE: The ``DX`` function has an unstable period.  
```python
real = DX(high, low, close, timeperiod=14)
```

### ER - Kaufman Efficiency Ratio
```python
real = ER(close, timeperiod=10)
```

### ERI - Elder Ray Index (Bull Power / Bear Power)
```python
bullpower, bearpower = ERI(high, low, close, timeperiod=13)
```

### FISHER - Fisher Transform
NOTE: The ``FISHER`` function has an unstable period.  
```python
fisher, trigger = FISHER(high, low, timeperiod=10)
```

### FOSC - Forecast Oscillator
```python
real = FOSC(close, timeperiod=5)
```

### FRACTAL - Williams Fractal
```python
swinghigh, swinglow = FRACTAL(high, low, leftbars=2, rightbars=2)
```

### IBS - Internal Bar Strength
```python
real = IBS(high, low, close)
```

### IMI - Intraday Momentum Index
```python
real = IMI(open, close, timeperiod=14)
```

### KDJ - KDJ Stochastic
```python
k, d, j = KDJ(high, low, close, fastk_period=9, slowk_period=3, slowk_matype=13, slowd_period=3, slowd_matype=13)
```

### KST - Know Sure Thing (Pring)
```python
kst, kstsignal = KST(close, roc1period=10, roc2period=15, roc3period=20, roc4period=30, sma1period=10, sma2period=10, sma3period=10, sma4period=15, signalperiod=9)
```

### KSTEXT - Know Sure Thing with controllable MA type
```python
kst, kstsignal = KSTEXT(close, roc1period=10, roc2period=15, roc3period=20, roc4period=30, ma1period=10, ma2period=10, ma3period=10, ma4period=15, signalperiod=9, rocmatype=0, signalmatype=0)
```

### MACD - Moving Average Convergence/Divergence
```python
macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
```

### MACDEXT - MACD with controllable MA type
```python
macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
```

### MACDFIX - Moving Average Convergence/Divergence Fix 12/26
```python
macd, macdsignal, macdhist = MACDFIX(close, signalperiod=9)
```

### MFI - Money Flow Index
```python
real = MFI(high, low, close, volume, timeperiod=14)
```

### MINUS_DI - Minus Directional Indicator
NOTE: The ``MINUS_DI`` function has an unstable period.  
```python
real = MINUS_DI(high, low, close, timeperiod=14)
```

### MINUS_DM - Minus Directional Movement
NOTE: The ``MINUS_DM`` function has an unstable period.  
```python
real = MINUS_DM(high, low, timeperiod=14)
```

### MOM - Momentum
```python
real = MOM(close, timeperiod=10)
```

### PLUS_DI - Plus Directional Indicator
NOTE: The ``PLUS_DI`` function has an unstable period.  
```python
real = PLUS_DI(high, low, close, timeperiod=14)
```

### PLUS_DM - Plus Directional Movement
NOTE: The ``PLUS_DM`` function has an unstable period.  
```python
real = PLUS_DM(high, low, timeperiod=14)
```

### PPO - Percentage Price Oscillator
```python
real = PPO(close, fastperiod=12, slowperiod=26, matype=1)
```

### PSO - Premier Stochastic Oscillator
```python
real = PSO(high, low, close, fastk_period=8, emaperiod=5)
```

### QSTICK - Qstick
```python
real = QSTICK(open, close, timeperiod=10)
```

### ROC - Rate of change : ((price/prevPrice)-1)*100
```python
real = ROC(close, timeperiod=10)
```

### ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
```python
real = ROCP(close, timeperiod=10)
```

### ROCR - Rate of change ratio: (price/prevPrice)
```python
real = ROCR(close, timeperiod=10)
```

### ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
```python
real = ROCR100(close, timeperiod=10)
```

### RSI - Relative Strength Index
NOTE: The ``RSI`` function has an unstable period.  
```python
real = RSI(close, timeperiod=14)
```

### SI - Wilder Swing Index
```python
real = SI(open, high, low, close, limitmove=3)
```

### SMI - Stochastic Momentum Index
```python
smi, smisignal = SMI(high, low, close, timeperiod=13, fastperiod=2, slowperiod=25, signalperiod=9)
```

### SQZMOM - Squeeze Momentum and Level
```python
momentum, squeeze = SQZMOM(high, low, close, bbperiod=20, nbdev=2, kcperiod=20, factorwide=2, factornormal=1.5, factornarrow=1)
```

### STC - Schaff Trend Cycle
NOTE: The ``STC`` function has an unstable period.  
```python
real = STC(close, fastperiod=23, slowperiod=50, cycleperiod=10)
```

### STOCH - Stochastic
NOTE: `slowk_matype` and `slowd_matype` take an `MA_Type` value. See [STOCH](https://ta-lib.org/functions/stoch) for the formula.
```python
slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
```

### STOCHF - Stochastic Fast
```python
fastk, fastd = STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
```

### STOCHRSI - Stochastic Relative Strength Index
```python
fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
```

### TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
```python
real = TRIX(close, timeperiod=30)
```

### TSI - True Strength Index
```python
real = TSI(close, firstperiod=25, secondperiod=13)
```

### ULTOSC - Ultimate Oscillator
```python
real = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
```

### VHF - Vertical Horizontal Filter
```python
real = VHF(close, timeperiod=28)
```

### VORTEX - Vortex Indicator
```python
plusvi, minusvi = VORTEX(high, low, close, timeperiod=14)
```

### WAD - Williams' Accumulation/Distribution
```python
real = WAD(high, low, close)
```

### WAVETREND - WaveTrend Oscillator
```python
wt1, wt2 = WAVETREND(high, low, close, channelperiod=10, averageperiod=21, signalperiod=4)
```

### WILLR - Williams' %R
```python
real = WILLR(high, low, close, timeperiod=14)
```


[Documentation Index](../doc_index.md)
[FLOAT_RIGHTAll Function Groups](../funcs.md)

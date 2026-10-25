# Overlap Studies Functions
### ACCBANDS - Acceleration Bands
```python
upperband, middleband, lowerband = ACCBANDS(high, low, close, timeperiod=20)
```

### ALMA - Arnaud Legoux Moving Average
```python
real = ALMA(close, timeperiod=9, sigma=6, offset=0.85)
```

### BBANDS - Bollinger Bands
```python
upperband, middleband, lowerband = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
```

### CKSP - Chande Kroll Stop
```python
highstop, lowstop = CKSP(high, low, close, timeperiod=10, multiplier=1, stopperiod=9)
```

### DEMA - Double Exponential Moving Average
```python
real = DEMA(close, timeperiod=30)
```

### DONCHIAN - Donchian Channels
```python
upperband, middleband, lowerband = DONCHIAN(high, low, timeperiod=20)
```

### EMA - Exponential Moving Average
NOTE: The ``EMA`` function has an unstable period.  
```python
real = EMA(close, timeperiod=30)
```

### FRAMA - Fractal Adaptive Moving Average
NOTE: The ``FRAMA`` function has an unstable period.  
```python
real = FRAMA(high, low, timeperiod=16)
```

### HMA - Hull Moving Average
```python
real = HMA(close, timeperiod=20)
```

### HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
NOTE: The ``HT_TRENDLINE`` function has an unstable period.  
```python
real = HT_TRENDLINE(close)
```

### ICHIMOKU - Ichimoku Kinko Hyo
```python
tenkansen, kijunsen, senkouspana, senkouspanb = ICHIMOKU(high, low, tenkanperiod=9, kijunperiod=26, senkoubperiod=52)
```

### KAMA - Kaufman Adaptive Moving Average
NOTE: The ``KAMA`` function has an unstable period.  
```python
real = KAMA(close, timeperiod=30)
```

### KC - Keltner Channels
```python
upperband, middleband, lowerband = KC(high, low, close, timeperiod=20, atrperiod=10, nbdev=2)
```

### MA - Moving average
```python
real = MA(close, timeperiod=30, matype=0)
```

### MAMA - MESA Adaptive Moving Average
NOTE: The ``MAMA`` function has an unstable period.  
```python
mama, fama = MAMA(close, fastlimit=0, slowlimit=0)
```

### MAVP - Moving average with variable period
```python
real = MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)
```

### MCGD - McGinley Dynamic
NOTE: The ``MCGD`` function has an unstable period.  
```python
real = MCGD(close, timeperiod=14)
```

### MIDPOINT - MidPoint over period
```python
real = MIDPOINT(close, timeperiod=14)
```

### MIDPRICE - Midpoint Price over period
```python
real = MIDPRICE(high, low, timeperiod=14)
```

### RMA - Wilder's Smoothed Moving Average
NOTE: The ``RMA`` function has an unstable period.  
```python
real = RMA(close, timeperiod=30)
```

### SAR - Parabolic SAR
```python
real = SAR(high, low, acceleration=0, maximum=0)
```

### SAREXT - Parabolic SAR - Extended
```python
real = SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
```

### SMA - Simple Moving Average
```python
real = SMA(close, timeperiod=30)
```

### SUPERTREND - SuperTrend
```python
supertrend, trend = SUPERTREND(high, low, close, timeperiod=10, multiplier=3)
```

### SWAK_BUTTER - Swiss Army Knife - Butterworth Filter
NOTE: The ``SWAK_BUTTER`` function has an unstable period.  
```python
real = SWAK_BUTTER(close, timeperiod=20)
```

### SWAK_GAUSS - Swiss Army Knife - Gaussian Filter
NOTE: The ``SWAK_GAUSS`` function has an unstable period.  
```python
real = SWAK_GAUSS(close, timeperiod=20)
```

### T3 - Triple Exponential Moving Average (T3)
NOTE: The ``T3`` function has an unstable period.  
```python
real = T3(close, timeperiod=5, vfactor=0)
```

### TEMA - Triple Exponential Moving Average
```python
real = TEMA(close, timeperiod=30)
```

### TRIMA - Triangular Moving Average
```python
real = TRIMA(close, timeperiod=30)
```

### VIDYA - Variable Index Dynamic Average
NOTE: The ``VIDYA`` function has an unstable period.  
```python
real = VIDYA(close, timeperiod=12, cmoperiod=9)
```

### VWMA - Volume Weighted Moving Average
```python
real = VWMA(close, volume, timeperiod=30)
```

### WMA - Weighted Moving Average
```python
real = WMA(close, timeperiod=30)
```

### ZIGZAG - Zig Zag
```python
zigzag, trend, pivotidx = ZIGZAG(high, low, sensitivity=5, mintrendlength=1)
```

### ZLEMA - Zero-Lag Exponential Moving Average
```python
real = ZLEMA(close, timeperiod=30)
```


[Documentation Index](../doc_index.md)
[FLOAT_RIGHTAll Function Groups](../funcs.md)

# Overlap Studies Functions
### BBANDS - Bollinger Bands
```python
upperband, middleband, lowerband = BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
```

### DEMA - Double Exponential Moving Average
```python
real = DEMA(close, timeperiod=30)
```

### EMA - Exponential Moving Average
NOTE: The ``EMA`` function has an unstable period.  
```python
real = EMA(close, timeperiod=30)
```

### HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
NOTE: The ``HT_TRENDLINE`` function has an unstable period.  
```python
real = HT_TRENDLINE(close)
```

### KAMA - Kaufman Adaptive Moving Average
NOTE: The ``KAMA`` function has an unstable period.  
```python
real = KAMA(close, timeperiod=30)
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

### MIDPOINT - MidPoint over period
```python
real = MIDPOINT(close, timeperiod=14)
```

### MIDPRICE - Midpoint Price over period
```python
real = MIDPRICE(high, low, timeperiod=14)
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

### WMA - Weighted Moving Average
```python
real = WMA(close, timeperiod=30)
```


[Documentation Index](../doc_index.html)
[FLOAT_RIGHTAll Function Groups](../funcs.html)

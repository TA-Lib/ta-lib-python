# Momentum Indicator Functions
### ADX - Average Directional Movement Index
NOTE: The ``ADX`` function has an unstable period.  
```python
real = ADX(high, low, close, timeperiod=14)
```

### ADXR - Average Directional Movement Index Rating
NOTE: The ``ADXR`` function has an unstable period.  
```python
real = ADXR(high, low, close, timeperiod=14)
```

### APO - Absolute Price Oscillator
```python
real = APO(close, fastperiod=12, slowperiod=26, matype=0)
```

### AROON - Aroon
```python
aroondown, aroonup = AROON(high, low, timeperiod=14)
```

### AROONOSC - Aroon Oscillator
```python
real = AROONOSC(high, low, timeperiod=14)
```

### BOP - Balance Of Power
```python
real = BOP(open, high, low, close)
```

### CCI - Commodity Channel Index
```python
real = CCI(high, low, close, timeperiod=14)
```

### CMO - Chande Momentum Oscillator
NOTE: The ``CMO`` function has an unstable period.  
```python
real = CMO(close, timeperiod=14)
```

### DX - Directional Movement Index
NOTE: The ``DX`` function has an unstable period.  
```python
real = DX(high, low, close, timeperiod=14)
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
NOTE: The ``MFI`` function has an unstable period.  
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
real = PPO(close, fastperiod=12, slowperiod=26, matype=0)
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

### STOCH - Stochastic
```python
slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
```

### STOCHF - Stochastic Fast
```python
fastk, fastd = STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
```

### STOCHRSI - Stochastic Relative Strength Index
NOTE: The ``STOCHRSI`` function has an unstable period.  
```python
fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
```

### TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
```python
real = TRIX(close, timeperiod=30)
```

### ULTOSC - Ultimate Oscillator
```python
real = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
```

### WILLR - Williams' %R
```python
real = WILLR(high, low, close, timeperiod=14)
```


[Documentation Index](../doc_index.html)
[FLOAT_RIGHTAll Function Groups](../funcs.html)

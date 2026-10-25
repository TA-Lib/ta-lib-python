# Volatility Indicator Functions
### ADR - Average Day Range
```python
real = ADR(high, low, timeperiod=14)
```

### ATR - Average True Range
NOTE: The ``ATR`` function has an unstable period.  
```python
real = ATR(high, low, close, timeperiod=14)
```

### BBW - Bollinger BandWidth
```python
real = BBW(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
```

### CVI - Chaikin's Volatility
```python
real = CVI(high, low, timeperiod=10, rocperiod=10)
```

### MASSI - Mass Index
```python
real = MASSI(high, low, fastperiod=9, slowperiod=25)
```

### NATR - Normalized Average True Range
NOTE: The ``NATR`` function has an unstable period.  
```python
real = NATR(high, low, close, timeperiod=14)
```

### PERCENTB - Bollinger Bands %B
```python
real = PERCENTB(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
```

### ROGERSSATCHELL - Rogers-Satchell Volatility
```python
real = ROGERSSATCHELL(open, high, low, close, timeperiod=10, annualization=252)
```

### RVI - Relative Volatility Index
NOTE: The ``RVI`` function has an unstable period.  
```python
real = RVI(close, timeperiod=14, stddevperiod=10)
```

### RVIR - Relative Volatility Index, refined high/low form
```python
real = RVIR(high, low, timeperiod=14, stddevperiod=10)
```

### TRANGE - True Range
```python
real = TRANGE(high, low, close)
```


[Documentation Index](../doc_index.md)
[FLOAT_RIGHTAll Function Groups](../funcs.md)

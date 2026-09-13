# Volume Indicator Functions
### AD - Chaikin A/D Line
```python
real = AD(high, low, close, volume)
```

### ADOSC - Chaikin A/D Oscillator
```python
real = ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
```

### CMF - Chaikin Money Flow
```python
real = CMF(high, low, close, volume, timeperiod=20)
```

### EFI - Elder's Force Index
```python
real = EFI(close, volume, timeperiod=13)
```

### MARKETFI - Market Facilitation Index
```python
real = MARKETFI(high, low, volume)
```

### NVI - Negative Volume Index
```python
real = NVI(close, volume)
```

### OBV - On Balance Volume
```python
real = OBV(close, volume)
```

### PVI - Positive Volume Index
```python
real = PVI(close, volume)
```

### PVO - Percentage Volume Oscillator
```python
real = PVO(volume, fastperiod=12, slowperiod=26, matype=1)
```

### PVT - Price Volume Trend
```python
real = PVT(close, volume)
```

### RVOL - Relative Volume
```python
real = RVOL(volume, timeperiod=20)
```

### VWAP - Volume Weighted Average Price
```python
real = VWAP(high, low, close, volume)
```


[Documentation Index](../doc_index.md)
[FLOAT_RIGHTAll Function Groups](../funcs.md)

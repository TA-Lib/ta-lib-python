# Overlap Studies Functions
### BBANDS - Bollinger Bands
```
upperband, middleband, lowerband = BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
```

Learn more about the Bollinger Bands at [tadoc.org](http://www.tadoc.org/indicator/BBANDS.htm).  
### DEMA - Double Exponential Moving Average
```
real = DEMA(close, timeperiod=30)
```

Learn more about the Double Exponential Moving Average at [tadoc.org](http://www.tadoc.org/indicator/DEMA.htm).  
### EMA - Exponential Moving Average
```
real = EMA(close, timeperiod=30)
```

Learn more about the Exponential Moving Average at [tadoc.org](http://www.tadoc.org/indicator/EMA.htm).  
### HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
```
real = HT_TRENDLINE(close)
```

Learn more about the Hilbert Transform - Instantaneous Trendline at [tadoc.org](http://www.tadoc.org/indicator/HT_TRENDLINE.htm).  
### KAMA - Kaufman Adaptive Moving Average
```
real = KAMA(close, timeperiod=30)
```

Learn more about the Kaufman Adaptive Moving Average at [tadoc.org](http://www.tadoc.org/indicator/KAMA.htm).  
### MA - Moving average
```
real = MA(close, timeperiod=30, matype=0)
```

### MAMA - MESA Adaptive Moving Average
```
mama, fama = MAMA(close, fastlimit=0, slowlimit=0)
```

Learn more about the MESA Adaptive Moving Average at [tadoc.org](http://www.tadoc.org/indicator/MAMA.htm).  
### MAVP - Moving average with variable period
```
real = MAVP(close, minperiod=2, maxperiod=30, matype=0)
```

### MIDPOINT - MidPoint over period
```
real = MIDPOINT(close, timeperiod=14)
```

Learn more about the MidPoint over period at [tadoc.org](http://www.tadoc.org/indicator/MIDPOINT.htm).  
### MIDPRICE - Midpoint Price over period
```
real = MIDPRICE(high, low, timeperiod=14)
```

Learn more about the Midpoint Price over period at [tadoc.org](http://www.tadoc.org/indicator/MIDPRICE.htm).  
### SAR - Parabolic SAR
```
real = SAR(high, low, acceleration=0, maximum=0)
```

Learn more about the Parabolic SAR at [tadoc.org](http://www.tadoc.org/indicator/SAR.htm).  
### SAREXT - Parabolic SAR - Extended
```
real = SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
```

### SMA - Simple Moving Average
```
real = SMA(close, timeperiod=30)
```

Learn more about the Simple Moving Average at [tadoc.org](http://www.tadoc.org/indicator/SMA.htm).  
### T3 - Triple Exponential Moving Average (T3)
```
real = T3(close, timeperiod=5, vfactor=0)
```

Learn more about the Triple Exponential Moving Average (T3) at [tadoc.org](http://www.tadoc.org/indicator/T3.htm).  
### TEMA - Triple Exponential Moving Average
```
real = TEMA(close, timeperiod=30)
```

Learn more about the Triple Exponential Moving Average at [tadoc.org](http://www.tadoc.org/indicator/TEMA.htm).  
### TRIMA - Triangular Moving Average
```
real = TRIMA(close, timeperiod=30)
```

Learn more about the Triangular Moving Average at [tadoc.org](http://www.tadoc.org/indicator/TRIMA.htm).  
### WMA - Weighted Moving Average
```
real = WMA(close, timeperiod=30)
```

Learn more about the Weighted Moving Average at [tadoc.org](http://www.tadoc.org/indicator/WMA.htm).  

[Documentation Index](../doc_index.html)
[FLOAT_RIGHTAll Function Groups](../funcs.html)

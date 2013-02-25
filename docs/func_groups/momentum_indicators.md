# Momentum Indicator Functions
### ADX - Average Directional Movement Index
```
real = ADX(high, low, close, timeperiod=14)
```

Learn more about the Average Directional Movement Index at [tadoc.org](http://www.tadoc.org/indicator/ADX.htm).  
### ADXR - Average Directional Movement Index Rating
```
real = ADXR(high, low, close, timeperiod=14)
```

Learn more about the Average Directional Movement Index Rating at [tadoc.org](http://www.tadoc.org/indicator/ADXR.htm).  
### APO - Absolute Price Oscillator
```
real = APO(close, fastperiod=12, slowperiod=26, matype=0)
```

Learn more about the Absolute Price Oscillator at [tadoc.org](http://www.tadoc.org/indicator/APO.htm).  
### AROON - Aroon
```
aroondown, aroonup = AROON(high, low, timeperiod=14)
```

Learn more about the Aroon at [tadoc.org](http://www.tadoc.org/indicator/AROON.htm).  
### AROONOSC - Aroon Oscillator
```
real = AROONOSC(high, low, timeperiod=14)
```

Learn more about the Aroon Oscillator at [tadoc.org](http://www.tadoc.org/indicator/AROONOSC.htm).  
### BOP - Balance Of Power
```
real = BOP(open, high, low, close)
```

Learn more about the Balance Of Power at [tadoc.org](http://www.tadoc.org/indicator/BOP.htm).  
### CCI - Commodity Channel Index
```
real = CCI(high, low, close, timeperiod=14)
```

Learn more about the Commodity Channel Index at [tadoc.org](http://www.tadoc.org/indicator/CCI.htm).  
### CMO - Chande Momentum Oscillator
```
real = CMO(close, timeperiod=14)
```

Learn more about the Chande Momentum Oscillator at [tadoc.org](http://www.tadoc.org/indicator/CMO.htm).  
### DX - Directional Movement Index
```
real = DX(high, low, close, timeperiod=14)
```

Learn more about the Directional Movement Index at [tadoc.org](http://www.tadoc.org/indicator/DX.htm).  
### MACD - Moving Average Convergence/Divergence
```
macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
```

Learn more about the Moving Average Convergence/Divergence at [tadoc.org](http://www.tadoc.org/indicator/MACD.htm).  
### MACDEXT - MACD with controllable MA type
```
macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
```

### MACDFIX - Moving Average Convergence/Divergence Fix 12/26
```
macd, macdsignal, macdhist = MACDFIX(close, signalperiod=9)
```

### MFI - Money Flow Index
```
real = MFI(high, low, close, volume, timeperiod=14)
```

Learn more about the Money Flow Index at [tadoc.org](http://www.tadoc.org/indicator/MFI.htm).  
### MINUS_DI - Minus Directional Indicator
```
real = MINUS_DI(high, low, close, timeperiod=14)
```

Learn more about the Minus Directional Indicator at [tadoc.org](http://www.tadoc.org/indicator/MINUS_DI.htm).  
### MINUS_DM - Minus Directional Movement
```
real = MINUS_DM(high, low, timeperiod=14)
```

Learn more about the Minus Directional Movement at [tadoc.org](http://www.tadoc.org/indicator/MINUS_DM.htm).  
### MOM - Momentum
```
real = MOM(close, timeperiod=10)
```

Learn more about the Momentum at [tadoc.org](http://www.tadoc.org/indicator/MOM.htm).  
### PLUS_DI - Plus Directional Indicator
```
real = PLUS_DI(high, low, close, timeperiod=14)
```

Learn more about the Plus Directional Indicator at [tadoc.org](http://www.tadoc.org/indicator/PLUS_DI.htm).  
### PLUS_DM - Plus Directional Movement
```
real = PLUS_DM(high, low, timeperiod=14)
```

Learn more about the Plus Directional Movement at [tadoc.org](http://www.tadoc.org/indicator/PLUS_DM.htm).  
### PPO - Percentage Price Oscillator
```
real = PPO(close, fastperiod=12, slowperiod=26, matype=0)
```

Learn more about the Percentage Price Oscillator at [tadoc.org](http://www.tadoc.org/indicator/PPO.htm).  
### ROC - Rate of change : ((price/prevPrice)-1)*100
```
real = ROC(close, timeperiod=10)
```

Learn more about the Rate of change : ((price/prevPrice)-1)*100 at [tadoc.org](http://www.tadoc.org/indicator/ROC.htm).  
### ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
```
real = ROCP(close, timeperiod=10)
```

Learn more about the Rate of change Percentage: (price-prevPrice)/prevPrice at [tadoc.org](http://www.tadoc.org/indicator/ROCP.htm).  
### ROCR - Rate of change ratio: (price/prevPrice)
```
real = ROCR(close, timeperiod=10)
```

Learn more about the Rate of change ratio: (price/prevPrice) at [tadoc.org](http://www.tadoc.org/indicator/ROCR.htm).  
### ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
```
real = ROCR100(close, timeperiod=10)
```

Learn more about the Rate of change ratio 100 scale: (price/prevPrice)*100 at [tadoc.org](http://www.tadoc.org/indicator/ROCR100.htm).  
### RSI - Relative Strength Index
```
real = RSI(close, timeperiod=14)
```

Learn more about the Relative Strength Index at [tadoc.org](http://www.tadoc.org/indicator/RSI.htm).  
### STOCH - Stochastic
```
slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
```

Learn more about the Stochastic at [tadoc.org](http://www.tadoc.org/indicator/STOCH.htm).  
### STOCHF - Stochastic Fast
```
fastk, fastd = STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
```

Learn more about the Stochastic Fast at [tadoc.org](http://www.tadoc.org/indicator/STOCHF.htm).  
### STOCHRSI - Stochastic Relative Strength Index
```
fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
```

Learn more about the Stochastic Relative Strength Index at [tadoc.org](http://www.tadoc.org/indicator/STOCHRSI.htm).  
### TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
```
real = TRIX(close, timeperiod=30)
```

Learn more about the 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA at [tadoc.org](http://www.tadoc.org/indicator/TRIX.htm).  
### ULTOSC - Ultimate Oscillator
```
real = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
```

Learn more about the Ultimate Oscillator at [tadoc.org](http://www.tadoc.org/indicator/ULTOSC.htm).  
### WILLR - Williams' %R
```
real = WILLR(high, low, close, timeperiod=14)
```

Learn more about the Williams' %R at [tadoc.org](http://www.tadoc.org/indicator/WILLR.htm).  

[Documentation Index](../doc_index.html)
[FLOAT_RIGHTAll Function Groups](../funcs.html)

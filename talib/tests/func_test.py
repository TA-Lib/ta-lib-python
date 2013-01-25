import unittest

import numpy as np
import talib

from talib.tests.data import series

def test_MAX():
    result = talib.MAX(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result), (len(series), len(result))
    assert result[i + 2] == 95.095
    assert result[i + 3] == 95.095
    assert result[i + 4] == 94.625
    assert result[i + 5] == 94.625

def test_MIN():
    result = talib.MIN(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert result[i + 1] == 93.780
    assert result[i + 2] == 93.780
    assert result[i + 3] == 92.530
    assert result[i + 4] == 92.530

def test_BBANDS():
    upper, middle, lower = talib.BBANDS(series, timeperiod=20,
                                        nbdevup=2.0, nbdevdn=2.0,
                                        matype=talib.MA_EMA)
    i = np.where(~np.isnan(upper))[0][0]
    assert len(upper) == len(middle) == len(lower) == len(series)
    assert abs(upper[i + 0] - 98.0734) < 1e-3
    assert abs(middle[i + 0] - 92.8910) < 1e-3
    assert abs(lower[i + 0] - 87.7086) < 1e-3
    assert abs(upper[i + 13] - 93.674) < 1e-3
    assert abs(middle[i + 13] - 87.679) < 1e-3
    assert abs(lower[i + 13] - 81.685) < 1e-3

def test_DEMA():
    result = talib.DEMA(series)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert abs(result[i + 1] - 86.765) < 1e-3
    assert abs(result[i + 2] - 86.942) < 1e-3
    assert abs(result[i + 3] - 87.089) < 1e-3
    assert abs(result[i + 4] - 87.656) < 1e-3

def test_EMAEMA():
    result = talib.EMA(series, timeperiod=2)
    result = talib.EMA(result, timeperiod=2)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert i == 2, i

def get_test_cases():
    ret = []
    return ret

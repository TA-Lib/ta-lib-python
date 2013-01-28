import numpy as np
from nose.tools import assert_equals, assert_true

import talib
from talib import func

series = np.array([ 91.50, 94.81, 94.38, 95.09, 93.78, 94.62, 92.53, 92.75,
     90.31, 92.47, 96.12, 97.25, 98.50, 89.88, 91.00, 92.81, 89.16, 89.34,
     91.62, 89.88, 88.38, 87.62, 84.78, 83.00, 83.50, 81.38, 84.44, 89.25,
     86.38, 86.25, 85.25, 87.12, 85.81, 88.97, 88.47, 86.88, 86.81, 84.88,
     84.19, 83.88, 83.38, 85.50, 89.19, 89.44, 91.09, 90.75, 91.44, 89.00,
     91.00, 90.50, 89.03, 88.81, 84.28, 83.50, 82.69, 84.75, 85.66, 86.19,
     88.94, 89.28, 88.62, 88.50, 91.97, 91.50, 93.25, 93.50, 93.16, 91.72,
     90.00, 89.69, 88.88, 85.19, 83.38, 84.88, 85.94, 97.25, 99.88, 104.94,
     106.00, 102.50, 102.41, 104.59, 106.12, 106.00, 106.06, 104.62, 108.62,
     109.31, 110.50, 112.75, 123.00, 119.62, 118.75, 119.25, 117.94, 116.44,
     115.19, 111.88, 110.59, 118.12, 116.00, 116.00, 112.00, 113.75, 112.94,
     116.00, 120.50, 116.62, 117.00, 115.25, 114.31, 115.50, 115.87, 120.69,
     120.19, 120.75, 124.75, 123.37, 122.94, 122.56, 123.12, 122.56, 124.62,
     129.25, 131.00, 132.25, 131.00, 132.81, 134.00, 137.38, 137.81, 137.88,
     137.25, 136.31, 136.25, 134.63, 128.25, 129.00, 123.87, 124.81, 123.00,
     126.25, 128.38, 125.37, 125.69, 122.25, 119.37, 118.50, 123.19, 123.50,
     122.19, 119.31, 123.31, 121.12, 123.37, 127.37, 128.50, 123.87, 122.94,
     121.75, 124.44, 122.00, 122.37, 122.94, 124.00, 123.19, 124.56, 127.25,
     125.87, 128.86, 132.00, 130.75, 134.75, 135.00, 132.38, 133.31, 131.94,
     130.00, 125.37, 130.13, 127.12, 125.19, 122.00, 125.00, 123.00, 123.50,
     120.06, 121.00, 117.75, 119.87, 122.00, 119.19, 116.37, 113.50, 114.25,
     110.00, 105.06, 107.00, 107.87, 107.00, 107.12, 107.00, 91.00, 93.94,
     93.87, 95.50, 93.00, 94.94, 98.25, 96.75, 94.81, 94.37, 91.56, 90.25,
     93.94, 93.62, 97.00, 95.00, 95.87, 94.06, 94.62, 93.75, 98.00, 103.94,
     107.87, 106.06, 104.50, 105.00, 104.19, 103.06, 103.42, 105.27, 111.87,
     116.00, 116.62, 118.28, 113.37, 109.00, 109.70, 109.25, 107.00, 109.19,
     110.00, 109.20, 110.12, 108.00, 108.62, 109.75, 109.81, 109.00, 108.75,
     107.87 ])

def test_MIN():
    result = func.MIN(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert_equals(len(series), len(result))
    assert_equals(result[i + 1], 93.780)
    assert_equals(result[i + 2], 93.780)
    assert_equals(result[i + 3], 92.530)
    assert_equals(result[i + 4], 92.530)

def test_MAX():
    result = func.MAX(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert_equals(len(series), len(result))
    assert_equals(result[i + 2], 95.090)
    assert_equals(result[i + 3], 95.090)
    assert_equals(result[i + 4], 94.620)
    assert_equals(result[i + 5], 94.620)

def test_BBANDS():
    upper, middle, lower = func.BBANDS(series, timeperiod=20,
                                        nbdevup=2.0, nbdevdn=2.0,
                                        matype=talib.MA_Type.EMA)
    i = np.where(~np.isnan(upper))[0][0]
    assert_true(len(upper) == len(middle) == len(lower) == len(series))
    #assert_true(abs(upper[i + 0] - 98.0734) < 1e-3)
    assert_true(abs(middle[i + 0] - 92.8910) < 1e-3)
    assert_true(abs(lower[i + 0] - 87.7086) < 1e-3)
    #assert_true(abs(upper[i + 13] - 93.674) < 1e-3)
    assert_true(abs(middle[i + 13] - 87.679) < 1e-3)
    assert_true(abs(lower[i + 13] - 81.685) < 1e-3)

def test_DEMA():
    result = func.DEMA(series)
    i = np.where(~np.isnan(result))[0][0]
    assert_true(len(series) == len(result))
    assert_true(abs(result[i + 1] - 86.765) < 1e-3)
    assert_true(abs(result[i + 2] - 86.942) < 1e-3)
    assert_true(abs(result[i + 3] - 87.089) < 1e-3)
    assert_true(abs(result[i + 4] - 87.656) < 1e-3)

def test_EMAEMA():
    result = func.EMA(series, timeperiod=2)
    result = func.EMA(result, timeperiod=2)
    i = np.where(~np.isnan(result))[0][0]
    assert_true(len(series) == len(result))
    assert_equals(i, 2)

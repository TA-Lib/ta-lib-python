import numpy as np
from nose.tools import assert_equals, assert_true, assert_raises

import talib
from talib import func
from talib.test_data import series, assert_np_arrays_equal, assert_np_arrays_not_equal

def test_input_lengths():
    a1 = np.arange(10, dtype=float)
    a2 = np.arange(11, dtype=float)
    with assert_raises(Exception):
        func.BOP(a2, a1, a1, a1)
    with assert_raises(Exception):
        func.BOP(a1, a2, a1, a1)
    with assert_raises(Exception):
        func.BOP(a1, a1, a2, a1)
    with assert_raises(Exception):
        func.BOP(a1, a1, a1, a2)

def test_input_nans():
    a1 = np.arange(10, dtype=float)
    a2 = np.arange(10, dtype=float)
    a2[0] = np.nan
    a2[1] = np.nan
    r1, r2 = func.AROON(a1, a2, 2)
    assert_np_arrays_equal(r1, [np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0])
    assert_np_arrays_equal(r2, [np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 100, 100, 100])
    r1, r2 = func.AROON(a2, a1, 2)
    assert_np_arrays_equal(r1, [np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0])
    assert_np_arrays_equal(r2, [np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 100, 100, 100])

def test_MIN():
    result = func.MIN(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert_equals(len(series), len(result))
    assert_equals(result[i + 1], 93.780)
    assert_equals(result[i + 2], 93.780)
    assert_equals(result[i + 3], 92.530)
    assert_equals(result[i + 4], 92.530)
    values = np.array([np.nan, 5., 4., 3., 5., 7.])
    result = func.MIN(values, timeperiod=2)
    assert_np_arrays_equal(result, [np.nan, np.nan, 4, 3, 3, 5])

def test_MAX():
    result = func.MAX(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert_equals(len(series), len(result))
    assert_equals(result[i + 2], 95.090)
    assert_equals(result[i + 3], 95.090)
    assert_equals(result[i + 4], 94.620)
    assert_equals(result[i + 5], 94.620)

def test_MOM():
    values = np.array([90.0,88.0,89.0])
    result = func.MOM(values, timeperiod=1)
    assert_np_arrays_equal(result, [np.nan, -2, 1])
    result = func.MOM(values, timeperiod=2)
    assert_np_arrays_equal(result, [np.nan, np.nan, -1])
    result = func.MOM(values, timeperiod=3)
    assert_np_arrays_equal(result, [np.nan, np.nan, np.nan])
    result = func.MOM(values, timeperiod=4)
    assert_np_arrays_equal(result, [np.nan, np.nan, np.nan])

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

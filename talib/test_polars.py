import numpy as np
import polars as pl
from nose.tools import assert_equals, assert_is_instance, assert_true

import talib
from talib.test_data import series, assert_np_arrays_equal

def test_MOM():
    values = pl.Series([90.0,88.0,89.0])
    result = talib.MOM(values, timeperiod=1)
    assert_is_instance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan, -2, 1])
    result = talib.MOM(values, timeperiod=2)
    assert_is_instance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan, np.nan, -1])
    result = talib.MOM(values, timeperiod=3)
    assert_is_instance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan, np.nan, np.nan])
    result = talib.MOM(values, timeperiod=4)
    assert_is_instance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan, np.nan, np.nan])

def test_MAVP():
    a = pl.Series([1,5,3,4,7,3,8,1,4,6], dtype=pl.Float64)
    b = pl.Series([2,4,2,4,2,4,2,4,2,4], dtype=pl.Float64)
    result = talib.MAVP(a, b, minperiod=2, maxperiod=4)
    assert_is_instance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan,np.nan,np.nan,3.25,5.5,4.25,5.5,4.75,2.5,4.75])
    sma2 = talib.SMA(a, 2)
    assert_is_instance(sma2, pl.Series)
    assert_np_arrays_equal(result.to_numpy()[4::2], sma2.to_numpy()[4::2])
    sma4 = talib.SMA(a, 4)
    assert_is_instance(sma4, pl.Series)
    assert_np_arrays_equal(result.to_numpy()[3::2], sma4.to_numpy()[3::2])
    result = talib.MAVP(a, b, minperiod=2, maxperiod=3)
    assert_is_instance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan,np.nan,4,4,5.5,4.666666666666667,5.5,4,2.5,3.6666666666666665])
    sma3 = talib.SMA(a, 3)
    assert_is_instance(sma3, pl.Series)
    assert_np_arrays_equal(result.to_numpy()[2::2], sma2.to_numpy()[2::2])
    assert_np_arrays_equal(result.to_numpy()[3::2], sma3.to_numpy()[3::2])

import numpy as np
import pandas as pd
from nose.tools import assert_equals, assert_is_instance

import talib
from talib.test_data import assert_np_arrays_equal

def test_MOM():
    values = pd.Series([90.0,88.0,89.0], index=[10, 20, 30])
    result = talib.MOM(values, timeperiod=1)
    assert_is_instance(result, pd.Series)
    assert_np_arrays_equal(result.values, [np.nan, -2, 1])
    assert_np_arrays_equal(result.index, [10, 20, 30])
    result = talib.MOM(values, timeperiod=2)
    assert_is_instance(result, pd.Series)
    assert_np_arrays_equal(result.values, [np.nan, np.nan, -1])
    assert_np_arrays_equal(result.index, [10, 20, 30])
    result = talib.MOM(values, timeperiod=3)
    assert_is_instance(result, pd.Series)
    assert_np_arrays_equal(result.values, [np.nan, np.nan, np.nan])
    assert_np_arrays_equal(result.index, [10, 20, 30])
    result = talib.MOM(values, timeperiod=4)
    assert_is_instance(result, pd.Series)
    assert_np_arrays_equal(result.values, [np.nan, np.nan, np.nan])
    assert_np_arrays_equal(result.index, [10, 20, 30])

import numpy as np
import polars as pl

import talib
from talib import abstract
from talib.test_data import series, assert_np_arrays_equal

def test_MOM():
    values = pl.Series([90.0,88.0,89.0])
    result = talib.MOM(values, timeperiod=1)
    assert isinstance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan, -2, 1])
    result = talib.MOM(values, timeperiod=2)
    assert isinstance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan, np.nan, -1])
    result = talib.MOM(values, timeperiod=3)
    assert isinstance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan, np.nan, np.nan])
    result = talib.MOM(values, timeperiod=4)
    assert isinstance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan, np.nan, np.nan])

def test_MAVP():
    a = pl.Series([1,5,3,4,7,3,8,1,4,6], dtype=pl.Float64)
    b = pl.Series([2,4,2,4,2,4,2,4,2,4], dtype=pl.Float64)
    result = talib.MAVP(a, b, minperiod=2, maxperiod=4)
    assert isinstance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan,np.nan,np.nan,3.25,5.5,4.25,5.5,4.75,2.5,4.75])
    sma2 = talib.SMA(a, 2)
    assert isinstance(sma2, pl.Series)
    assert_np_arrays_equal(result.to_numpy()[4::2], sma2.to_numpy()[4::2])
    sma4 = talib.SMA(a, 4)
    assert isinstance(sma4, pl.Series)
    assert_np_arrays_equal(result.to_numpy()[3::2], sma4.to_numpy()[3::2])
    result = talib.MAVP(a, b, minperiod=2, maxperiod=3)
    assert isinstance(result, pl.Series)
    assert_np_arrays_equal(result.to_numpy(), [np.nan,np.nan,4,4,5.5,4.666666666666667,5.5,4,2.5,3.6666666666666665])
    sma3 = talib.SMA(a, 3)
    assert isinstance(sma3, pl.Series)
    assert_np_arrays_equal(result.to_numpy()[2::2], sma2.to_numpy()[2::2])
    assert_np_arrays_equal(result.to_numpy()[3::2], sma3.to_numpy()[3::2])

def test_TEVA():
    size = 50
    df = pl.DataFrame(
        {
            "open": np.random.uniform(low=0.0, high=100.0, size=size).astype("float32"),
            "high": np.random.uniform(low=0.0, high=100.0, size=size).astype("float32"),
            "low": np.random.uniform(low=0.0, high=100.0, size=size).astype("float32"),
            "close": np.random.uniform(low=0.0, high=100.0, size=size).astype("float32"),
            "volume": np.random.uniform(low=0.0, high=100.0, size=size).astype("float32")
        }
    )
    tema1 = abstract.TEMA(df, timeperiod=9)
    assert isinstance(tema1, pl.Series)
    assert len(tema1) == 50
    inputs = abstract.TEMA.get_input_arrays()
    assert inputs.columns == df.columns
    for column in df.columns:
        assert_np_arrays_equal(inputs[column].to_numpy(), df[column].to_numpy())

    tema2 = abstract.TEMA(df, timeperiod=9)
    assert isinstance(tema2, pl.Series)
    assert len(tema2) == 50
    inputs = abstract.TEMA.get_input_arrays()
    assert inputs.columns == df.columns
    for column in df.columns:
        assert_np_arrays_equal(inputs[column].to_numpy(), df[column].to_numpy())

    assert_np_arrays_equal(tema1.to_numpy(), tema2.to_numpy())

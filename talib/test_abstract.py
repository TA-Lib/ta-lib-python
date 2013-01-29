import numpy as np

from talib import func
from talib import abstract
from talib.test_data import ford_2012, assert_np_arrays_equal, assert_np_arrays_not_equal

def test_SMA():
    expected = func.SMA(ford_2012['close'], 10)
    assert_np_arrays_equal(expected, abstract.Function('sma', ford_2012, 10).outputs)
    assert_np_arrays_equal(expected, abstract.Function('sma')(ford_2012, 10, price='close'))
    assert_np_arrays_equal(expected, abstract.Function('sma')(ford_2012, timeperiod=10))
    expected = func.SMA(ford_2012['open'], 10)
    assert_np_arrays_equal(expected, abstract.Function('sma', ford_2012, 10, price='open').outputs)
    assert_np_arrays_equal(expected, abstract.Function('sma', price='low')(ford_2012, 10, price='open'))
    assert_np_arrays_not_equal(expected, abstract.Function('sma', ford_2012, 10, price='open')(timeperiod=20))
    assert_np_arrays_not_equal(expected, abstract.Function('sma', ford_2012)(10, price='close'))
    assert_np_arrays_not_equal(expected, abstract.Function('sma', 10)(ford_2012, price='high'))
    assert_np_arrays_not_equal(expected, abstract.Function('sma', price='low')(ford_2012, 10))

def test_STOCH():
    # check defaults match
    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close']) # 5, 3, 0, 3, 0
    got_k, got_d = abstract.Function('stoch', ford_2012).outputs
    assert_np_arrays_equal(expected_k, got_k)
    assert_np_arrays_equal(expected_d, got_d)

    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'])
    got_k, got_d = abstract.Function('stoch', ford_2012)(5, 3, 0, 3, 0)
    assert_np_arrays_equal(expected_k, got_k)
    assert_np_arrays_equal(expected_d, got_d)

    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'], 15)
    got_k, got_d = abstract.Function('stoch', ford_2012)(15, 5, 0, 5, 0)
    assert_np_arrays_not_equal(expected_k, got_k)
    assert_np_arrays_not_equal(expected_d, got_d)

    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'], 15, 5, 1, 5, 1)
    got_k, got_d = abstract.Function('stoch', ford_2012)(15, 5, 1, 5, 1)
    assert_np_arrays_equal(expected_k, got_k)
    assert_np_arrays_equal(expected_d, got_d)

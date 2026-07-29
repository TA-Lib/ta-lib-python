import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

import talib
from talib import func


def test_talib_version():
    assert talib.__ta_version__[:5] == b'0.7.1'


def test_num_functions():
    assert len(talib.get_functions()) == 161


def test_input_wrong_type():
    a1 = np.arange(10, dtype=int)
    with pytest.raises(Exception):
        func.MOM(a1)


def test_input_lengths():
    a1 = np.arange(10, dtype=float)
    a2 = np.arange(11, dtype=float)
    with pytest.raises(Exception):
        func.BOP(a2, a1, a1, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a2, a1, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a1, a2, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a1, a1, a2)


def test_input_allnans():
    a = np.arange(20, dtype=float)
    a[:] = np.nan
    r = func.RSI(a)
    assert np.all(np.isnan(r))


def test_input_nans():
    a1 = np.arange(10, dtype=float)
    a2 = np.arange(10, dtype=float)
    a2[0] = np.nan
    a2[1] = np.nan
    r1, r2 = func.AROON(a1, a2, 2)
    assert_array_equal(r1, [np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0])
    assert_array_equal(r2, [np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 100, 100, 100])
    r1, r2 = func.AROON(a2, a1, 2)
    assert_array_equal(r1, [np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0])
    assert_array_equal(r2, [np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 100, 100, 100])


def test_unstable_period():
    a = np.arange(10, dtype=float)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8])
    talib.set_unstable_period('EMA', 5)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, 7, 8])
    talib.set_unstable_period('EMA', 0)


# One entry per real unstable-period id in TA-Lib C.  Each id is named after
# the function it controls, so every case simply calls that same function.
def _unstable_period_cases():
    n = 400
    rs = np.random.RandomState(1)
    close = np.cumsum(rs.randn(n)) + 100.0
    high = close + rs.rand(n) + 0.5
    low = close - rs.rand(n) - 0.5
    return {
        'ADX': lambda: func.ADX(high, low, close),
        'ATR': lambda: func.ATR(high, low, close),
        'CMO': lambda: func.CMO(close),
        'DX': lambda: func.DX(high, low, close),
        'EMA': lambda: func.EMA(close),
        'HT_DCPERIOD': lambda: func.HT_DCPERIOD(close),
        'HT_DCPHASE': lambda: func.HT_DCPHASE(close),
        'HT_PHASOR': lambda: func.HT_PHASOR(close)[0],
        'HT_SINE': lambda: func.HT_SINE(close)[0],
        'HT_TRENDLINE': lambda: func.HT_TRENDLINE(close),
        'HT_TRENDMODE': lambda: func.HT_TRENDMODE(close),
        'KAMA': lambda: func.KAMA(close),
        'MAMA': lambda: func.MAMA(close)[0],
        'MINUS_DI': lambda: func.MINUS_DI(high, low, close),
        'MINUS_DM': lambda: func.MINUS_DM(high, low),
        'NATR': lambda: func.NATR(high, low, close),
        'PLUS_DI': lambda: func.PLUS_DI(high, low, close),
        'PLUS_DM': lambda: func.PLUS_DM(high, low),
        'RSI': lambda: func.RSI(close),
        'T3': lambda: func.T3(close),
    }


UNSTABLE_PERIOD_CASES = _unstable_period_cases()


def _leading_unset(result):
    # TA-Lib writes nothing before its lookback: real outputs stay NaN there,
    # the one integer output (HT_TRENDMODE) stays 0.
    valid = ~np.isnan(result) if result.dtype.kind == 'f' else result != 0
    assert valid.any(), 'no valid output to measure'
    return int(np.argmax(valid))


# NOTE: this has to be a behavioural test.  set_unstable_period() and
# get_unstable_period() look the id up in the same table, so a wrong id
# round-trips perfectly -- which is how ta-lib-python shipped a table that
# pointed 'RSI' at PLUS_DM for two years (issue #752).  Only checking that the
# setting moves THAT function's own output can catch it.  Please do not
# "simplify" this into a get/set assertion.
@pytest.mark.parametrize('name', sorted(UNSTABLE_PERIOD_CASES))
def test_unstable_period_moves_its_own_function(name):
    call = UNSTABLE_PERIOD_CASES[name]
    talib.set_unstable_period(name, 0)
    unshifted = call()
    baseline = _leading_unset(unshifted)
    assert baseline > 0, 'nothing to shift'
    try:
        talib.set_unstable_period(name, 5)
        shifted = call()
        assert _leading_unset(shifted) == baseline + 5
        # an unstable period only discards warm-up bars, so whatever both runs
        # do emit has to be identical -- this pins the shift to a pure delay
        assert_array_equal(shifted[baseline + 5:], unshifted[baseline + 5:])
    finally:
        talib.set_unstable_period(name, 0)
    assert _leading_unset(call()) == baseline


def test_unstable_period_all():
    talib.set_unstable_period('ALL', 0)
    baseline = {name: _leading_unset(call())
                for name, call in UNSTABLE_PERIOD_CASES.items()}
    try:
        talib.set_unstable_period('ALL', 3)
        for name, call in UNSTABLE_PERIOD_CASES.items():
            assert _leading_unset(call()) == baseline[name] + 3, name
    finally:
        talib.set_unstable_period('ALL', 0)
    for name, call in UNSTABLE_PERIOD_CASES.items():
        assert _leading_unset(call()) == baseline[name], name


# These three are accepted for backwards compatibility only: TA-Lib C retired
# their unstable-period slots.  They must stay no-ops -- aliasing them to the
# inner ADX/RSI would give them side effects on every other function.
@pytest.mark.parametrize('name', ['ADXR', 'MFI', 'STOCHRSI'])
def test_unstable_period_retired_is_a_warning_and_a_noop(name):
    talib.set_unstable_period('ALL', 0)
    before = {n: _leading_unset(call())
              for n, call in UNSTABLE_PERIOD_CASES.items()}
    with pytest.deprecated_call():
        talib.set_unstable_period(name, 5)
    with pytest.deprecated_call():
        assert talib.get_unstable_period(name) == 0
    for n, call in UNSTABLE_PERIOD_CASES.items():
        assert _leading_unset(call()) == before[n], n


def test_compatibility():
    a = np.arange(10, dtype=float)
    talib.set_compatibility(0)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8])
    talib.set_compatibility(1)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan,1.25,2.125,3.0625,4.03125,5.015625,6.0078125,7.00390625,8.001953125])
    talib.set_compatibility(0)


def test_MIN(series):
    result = func.MIN(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert result[i + 1] == 93.780
    assert result[i + 2] == 93.780
    assert result[i + 3] == 92.530
    assert result[i + 4] == 92.530
    values = np.array([np.nan, 5., 4., 3., 5., 7.])
    result = func.MIN(values, timeperiod=2)
    assert_array_equal(result, [np.nan, np.nan, 4, 3, 3, 5])


def test_MAX(series):
    result = func.MAX(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert result[i + 2] == 95.090
    assert result[i + 3] == 95.090
    assert result[i + 4] == 94.620
    assert result[i + 5] == 94.620


def test_MOM():
    values = np.array([90.0,88.0,89.0])
    result = func.MOM(values, timeperiod=1)
    assert_array_equal(result, [np.nan, -2, 1])
    result = func.MOM(values, timeperiod=2)
    assert_array_equal(result, [np.nan, np.nan, -1])
    result = func.MOM(values, timeperiod=3)
    assert_array_equal(result, [np.nan, np.nan, np.nan])
    result = func.MOM(values, timeperiod=4)
    assert_array_equal(result, [np.nan, np.nan, np.nan])


def test_BBANDS(series):
    upper, middle, lower = func.BBANDS(
        series,
        timeperiod=20,
        nbdevup=2.0,
        nbdevdn=2.0,
        matype=talib.MA_Type.EMA
    )
    i = np.where(~np.isnan(upper))[0][0]
    assert len(upper) == len(middle) == len(lower) == len(series)
    # assert abs(upper[i + 0] - 98.0734) < 1e-3
    assert abs(middle[i + 0] - 92.8910) < 1e-3
    assert abs(lower[i + 0] - 87.7086) < 1e-3
    # assert abs(upper[i + 13] - 93.674) < 1e-3
    assert abs(middle[i + 13] - 87.679) < 1e-3
    assert abs(lower[i + 13] - 81.685) < 1e-3


def test_DEMA(series):
    result = func.DEMA(series)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert abs(result[i + 1] - 86.765) < 1e-3
    assert abs(result[i + 2] - 86.942) < 1e-3
    assert abs(result[i + 3] - 87.089) < 1e-3
    assert abs(result[i + 4] - 87.656) < 1e-3


def test_EMAEMA(series):
    result = func.EMA(series, timeperiod=2)
    result = func.EMA(result, timeperiod=2)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert i == 2


def test_CDL3BLACKCROWS():
    o = np.array([39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 40.32, 40.51, 38.09, 35.00, 27.66, 30.80])
    h = np.array([40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 41.69, 40.84, 38.12, 35.50, 31.74, 32.51])
    l = np.array([35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 39.26, 36.73, 33.37, 30.03, 27.03, 28.31])
    c = np.array([40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.46, 37.08, 33.37, 30.03, 31.46, 28.31])

    result = func.CDL3BLACKCROWS(o, h, l, c)
    assert_array_equal(result, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0])


def test_RSI():
    a = np.array([0.00000024, 0.00000024, 0.00000024,
      0.00000024, 0.00000024, 0.00000023,
      0.00000024, 0.00000024, 0.00000024,
      0.00000024, 0.00000023, 0.00000024,
      0.00000023, 0.00000024, 0.00000023,
      0.00000024, 0.00000024, 0.00000023,
      0.00000023, 0.00000023], dtype='float64')
    result = func.RSI(a, 10)
    assert_array_almost_equal(result, [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,33.333333333333329,51.351351351351347,39.491916859122398,51.84807024709005,42.25953803191981,52.101824405061215,52.101824405061215,43.043664867691085,43.043664867691085,43.043664867691085])
    result = func.RSI(a * 100000, 10)
    assert_array_almost_equal(result, [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,33.333333333333329,51.351351351351347,39.491916859122398,51.84807024709005,42.25953803191981,52.101824405061215,52.101824405061215,43.043664867691085,43.043664867691085,43.043664867691085])


def test_MAVP():
    a = np.array([1,5,3,4,7,3,8,1,4,6], dtype=float)
    b = np.array([2,4,2,4,2,4,2,4,2,4], dtype=float)
    result = func.MAVP(a, b, minperiod=2, maxperiod=4)
    assert_array_equal(result, [np.nan,np.nan,np.nan,3.25,5.5,4.25,5.5,4.75,2.5,4.75])
    sma2 = func.SMA(a, 2)
    assert_array_equal(result[4::2], sma2[4::2])
    sma4 = func.SMA(a, 4)
    assert_array_equal(result[3::2], sma4[3::2])
    result = func.MAVP(a, b, minperiod=2, maxperiod=3)
    assert_array_equal(result, [np.nan,np.nan,4,4,5.5,4.666666666666667,5.5,4,2.5,3.6666666666666665])
    sma3 = func.SMA(a, 3)
    assert_array_equal(result[2::2], sma2[2::2])
    assert_array_equal(result[3::2], sma3[3::2])


def test_MAXINDEX():
    import talib as func
    import numpy as np
    a = np.array([1., 2, 3, 4, 5, 6, 7, 8, 7, 7, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 15])
    b = func.MA(a, 10)
    c = func.MAXINDEX(b, 10)
    assert_array_equal(c, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,21])
    d = np.array([1., 2, 3])
    e = func.MAXINDEX(d, 10)
    assert_array_equal(e, [0,0,0])

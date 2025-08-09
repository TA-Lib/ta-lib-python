import numpy as np
import pandas as pd

from talib import stream


def test_streaming():
    a = np.array([1,1,2,3,5,8,13], dtype=float)
    r = stream.MOM(a, timeperiod=1)
    assert r == 5
    r = stream.MOM(a, timeperiod=2)
    assert r == 8
    r = stream.MOM(a, timeperiod=3)
    assert r == 10
    r = stream.MOM(a, timeperiod=4)
    assert r == 11
    r = stream.MOM(a, timeperiod=5)
    assert r == 12
    r = stream.MOM(a, timeperiod=6)
    assert r == 12
    r = stream.MOM(a, timeperiod=7)
    assert np.isnan(r)


def test_streaming_pandas():
    a = pd.Series([1,1,2,3,5,8,13])
    r = stream.MOM(a, timeperiod=1)
    assert r == 5
    r = stream.MOM(a, timeperiod=2)
    assert r == 8
    r = stream.MOM(a, timeperiod=3)
    assert r == 10
    r = stream.MOM(a, timeperiod=4)
    assert r == 11
    r = stream.MOM(a, timeperiod=5)
    assert r == 12
    r = stream.MOM(a, timeperiod=6)
    assert r == 12
    r = stream.MOM(a, timeperiod=7)
    assert np.isnan(r)


def test_CDL3BLACKCROWS():
    o = np.array([39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 40.32, 40.51, 38.09, 35.00])
    h = np.array([40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 41.69, 40.84, 38.12, 35.50])
    l = np.array([35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 39.26, 36.73, 33.37, 30.03])
    c = np.array([40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.46, 37.08, 33.37, 30.03])

    r = stream.CDL3BLACKCROWS(o, h, l, c)
    assert r == -100


def test_CDL3BLACKCROWS_pandas():
    o = pd.Series([39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 40.32, 40.51, 38.09, 35.00])
    h = pd.Series([40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 41.69, 40.84, 38.12, 35.50])
    l = pd.Series([35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 39.26, 36.73, 33.37, 30.03])
    c = pd.Series([40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.46, 37.08, 33.37, 30.03])

    r = stream.CDL3BLACKCROWS(o, h, l, c)
    assert r == -100


def test_MAXINDEX():
    a = np.array([1., 2, 3, 4, 5, 6, 7, 8, 7, 7, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 15])
    r = stream.MAXINDEX(a, 10)
    assert r == 21

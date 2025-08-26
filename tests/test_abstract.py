import copy
import re
import threading
import time

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_raises

try:
    from collections import OrderedDict
except ImportError: # handle python 2.6 and earlier
    from ordereddict import OrderedDict

import talib
from talib import abstract, func


def assert_array_not_equal(x, y):
    assert_raises(AssertionError, assert_array_equal, x, y)


def test_pararmeters():
    parameters = abstract.MACD.parameters
    assert all(type(v) == int for k, v in parameters.items())


def test_pandas(ford_2012):
    import pandas
    input_df = pandas.DataFrame(ford_2012)
    input_dict = dict((k, pandas.Series(v)) for k, v in ford_2012.items())

    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close']) # 5, 3, 0, 3, 0
    output = abstract.Function('stoch', input_df).outputs
    assert isinstance(output, pandas.DataFrame)
    assert_array_equal(expected_k, output['slowk'])
    assert_array_equal(expected_d, output['slowd'])
    output = abstract.Function('stoch', input_dict).outputs
    assert isinstance(output, list)
    assert_array_equal(expected_k, output[0])
    assert_array_equal(expected_d, output[1])

    expected = func.SMA(ford_2012['close'], 10)
    output = abstract.Function('sma', input_df, 10).outputs
    assert isinstance(output, pandas.Series)
    assert_array_equal(expected, output)
    output = abstract.Function('sma', input_dict, 10).outputs
    assert isinstance(output, np.ndarray)
    assert_array_equal(expected, output)


def test_pandas_series(ford_2012):
    import pandas
    input_df = pandas.DataFrame(ford_2012)
    output = talib.SMA(input_df['close'], 10)
    expected = pandas.Series(func.SMA(ford_2012['close'], 10),
                             index=input_df.index)
    pandas.testing.assert_series_equal(output, expected)

    # Test kwargs
    output = talib.SMA(real=input_df['close'], timeperiod=10)
    pandas.testing.assert_series_equal(output, expected)

    # Test talib.func API
    output = func.SMA(input_df['close'], timeperiod=10)
    pandas.testing.assert_series_equal(output, expected)

    # Test multiple outputs such as from BBANDS
    _, output, _ = talib.BBANDS(input_df['close'], 10)
    expected = pandas.Series(func.BBANDS(ford_2012['close'], 10)[1],
                             index=input_df.index)
    pandas.testing.assert_series_equal(output, expected)


def test_SMA(ford_2012):
    expected = func.SMA(ford_2012['close'], 10)
    assert_array_equal(expected, abstract.Function('sma', ford_2012, 10).outputs)
    assert_array_equal(expected, abstract.Function('sma')(ford_2012, 10, price='close'))
    assert_array_equal(expected, abstract.Function('sma')(ford_2012, timeperiod=10))
    expected = func.SMA(ford_2012['open'], 10)
    assert_array_equal(expected, abstract.Function('sma', ford_2012, 10, price='open').outputs)
    assert_array_equal(expected, abstract.Function('sma', price='low')(ford_2012, 10, price='open'))
    assert_array_not_equal(expected, abstract.Function('sma', ford_2012, 10, price='open')(timeperiod=20))
    assert_array_not_equal(expected, abstract.Function('sma', ford_2012)(10, price='close'))
    assert_array_not_equal(expected, abstract.Function('sma', 10)(ford_2012, price='high'))
    assert_array_not_equal(expected, abstract.Function('sma', price='low')(ford_2012, 10))
    input_arrays = {'foobarbaz': ford_2012['open']}
    assert_array_equal(expected, abstract.SMA(input_arrays, 10, price='foobarbaz'))


def test_STOCH(ford_2012):
    # check defaults match
    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close']) # 5, 3, 0, 3, 0
    got_k, got_d = abstract.Function('stoch', ford_2012).outputs
    assert_array_equal(expected_k, got_k)
    assert_array_equal(expected_d, got_d)

    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'])
    got_k, got_d = abstract.Function('stoch', ford_2012)(5, 3, 0, 3, 0)
    assert_array_equal(expected_k, got_k)
    assert_array_equal(expected_d, got_d)

    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'], 15)
    got_k, got_d = abstract.Function('stoch', ford_2012)(15, 5, 0, 5, 0)
    assert_array_not_equal(expected_k, got_k)
    assert_array_not_equal(expected_d, got_d)

    expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'], 15, 5, 1, 5, 1)
    got_k, got_d = abstract.Function('stoch', ford_2012)(15, 5, 1, 5, 1)
    assert_array_equal(expected_k, got_k)
    assert_array_equal(expected_d, got_d)


def test_doji_candle(ford_2012):
    expected = func.CDLDOJI(ford_2012['open'], ford_2012['high'], ford_2012['low'], ford_2012['close'])
    got = abstract.Function('CDLDOJI').run(ford_2012)
    assert_array_equal(got, expected)


def test_MAVP(ford_2012):
    mavp = abstract.MAVP
    with pytest.raises(Exception):
        mavp.set_input_arrays(ford_2012)
    input_d = {}
    input_d['close'] = ford_2012['close']
    input_d['periods'] = np.arange(30)
    assert mavp.set_input_arrays(input_d)
    assert mavp.input_arrays == input_d


def test_info():
    stochrsi = abstract.Function('STOCHRSI')
    stochrsi.input_names = {'price': 'open'}
    stochrsi.parameters = {'fastd_matype': talib.MA_Type.EMA}
    expected = {
        'display_name': 'Stochastic Relative Strength Index',
        'function_flags': ['Function has an unstable period'],
        'group': 'Momentum Indicators',
        'input_names': OrderedDict([('price', 'open')]),
        'name': 'STOCHRSI',
        'output_flags': OrderedDict([
            ('fastk', ['Line']),
            ('fastd', ['Line']),
            ]),
        'output_names': ['fastk', 'fastd'],
        'parameters': OrderedDict([
            ('timeperiod', 14),
            ('fastk_period', 5),
            ('fastd_period', 3),
            ('fastd_matype', 1),
            ]),
        }
    assert expected == stochrsi.info

    expected = {
        'display_name': 'Bollinger Bands',
        'function_flags': ['Output scale same as input'],
        'group': 'Overlap Studies',
        'input_names': OrderedDict([('price', 'close')]),
        'name': 'BBANDS',
        'output_flags': OrderedDict([
            ('upperband', ['Values represent an upper limit']),
            ('middleband', ['Line']),
            ('lowerband', ['Values represent a lower limit']),
            ]),
        'output_names': ['upperband', 'middleband', 'lowerband'],
        'parameters': OrderedDict([
            ('timeperiod', 5),
            ('nbdevup', 2),
            ('nbdevdn', 2),
            ('matype', 0),
            ]),
        }
    assert expected == abstract.Function('BBANDS').info


def test_input_names():
    expected = OrderedDict([('price', 'close')])
    assert expected == abstract.Function('MAMA').input_names

    # test setting input_names
    obv = abstract.Function('OBV')
    expected = OrderedDict([
        ('price', 'open'),
        ('prices', ['volume']),
        ])
    obv.input_names = expected
    assert obv.input_names == expected

    obv.input_names = {
        'price': 'open',
        'prices': ['volume'],
        }
    assert obv.input_names == expected


def test_input_arrays(ford_2012):
    mama = abstract.Function('MAMA')

    # test default setting
    assert mama.get_input_arrays() == {}

    # test setting/getting input_arrays
    assert mama.set_input_arrays(ford_2012)
    assert mama.get_input_arrays() == ford_2012
    with pytest.raises(Exception):
        mama.set_input_arrays({'hello': 'fail', 'world': 'bye'})

    # test only required keys are needed
    willr = abstract.Function('WILLR')
    reqd = willr.input_names['prices']
    input_d = dict([(key, ford_2012[key]) for key in reqd])
    assert willr.set_input_arrays(input_d)
    assert willr.input_arrays == input_d

    # test extraneous keys are ignored
    input_d['extra_stuffs'] = 'you should never see me'
    input_d['date'] = np.random.rand(100)
    assert willr.set_input_arrays(input_d)

    # test missing keys get detected
    input_d['open'] = ford_2012['open']
    input_d.pop('close')
    with pytest.raises(Exception):
        willr.set_input_arrays(input_d)

    # test changing input_names on the Function
    willr.input_names = {'prices': ['high', 'low', 'open']}
    assert willr.set_input_arrays(input_d)


def test_parameters():
    stoch = abstract.Function('STOCH')
    expected = OrderedDict([
        ('fastk_period', 5),
        ('slowk_period', 3),
        ('slowk_matype', 0),
        ('slowd_period', 3),
        ('slowd_matype', 0),
        ])
    assert expected == stoch.parameters

    stoch.parameters = {'fastk_period': 10}
    expected['fastk_period'] = 10
    assert expected == stoch.parameters

    stoch.parameters = {'slowk_period': 8, 'slowd_period': 5}
    expected['slowk_period'] = 8
    expected['slowd_period'] = 5
    assert expected == stoch.parameters

    stoch.parameters = {'slowd_matype': talib.MA_Type.T3}
    expected['slowd_matype'] = 8
    assert expected == stoch.parameters

    stoch.parameters = {
        'slowk_matype': talib.MA_Type.WMA,
        'slowd_matype': talib.MA_Type.EMA,
        }
    expected['slowk_matype'] = 2
    expected['slowd_matype'] = 1
    assert expected == stoch.parameters


def test_lookback():
    assert abstract.Function('SMA', 10).lookback == 9

    stochrsi = abstract.Function('stochrsi', 20, 5, 3)
    assert stochrsi.lookback == 26


def test_call_supports_same_signature_as_func_module(ford_2012):
    adx = abstract.Function('ADX')

    expected = func.ADX(ford_2012['open'], ford_2012['high'], ford_2012['low'])
    output = adx(ford_2012['open'], ford_2012['high'], ford_2012['low'])
    assert_array_equal(expected, output)

    expected_error = re.escape('Too many price arguments: expected 3 (high, low, close)')

    with pytest.raises(TypeError, match=expected_error):
        adx(ford_2012['open'], ford_2012['high'], ford_2012['low'], ford_2012['close'])

    expected_error = re.escape('Not enough price arguments: expected 3 (high, low, close)')

    with pytest.raises(TypeError, match=expected_error):
        adx(ford_2012['open'], ford_2012['high'])


def test_parameter_type_checking(ford_2012):
    sma = abstract.Function('SMA', timeperiod=10)

    expected_error = re.escape('Invalid parameter value for timeperiod (expected int, got float)')

    with pytest.raises(TypeError, match=expected_error):
        sma(ford_2012['close'], 35.5)

    with pytest.raises(TypeError, match=expected_error):
        abstract.Function('SMA', timeperiod=35.5)

    with pytest.raises(TypeError, match=expected_error):
        sma.parameters = {'timeperiod': 35.5}

    with pytest.raises(TypeError, match=expected_error):
        sma.set_parameters(timeperiod=35.5)


def test_call_doesnt_cache_parameters(ford_2012):
    sma = abstract.Function('SMA', timeperiod=10)

    expected = func.SMA(ford_2012['open'], 20)
    output = sma(ford_2012, timeperiod=20, price='open')
    assert_array_equal(expected, output)

    expected = func.SMA(ford_2012['close'], 20)
    output = sma(ford_2012, timeperiod=20)
    assert_array_equal(expected, output)

    expected = func.SMA(ford_2012['close'], 10)
    output = sma(ford_2012)
    assert_array_equal(expected, output)


def test_call_without_arguments():
    with pytest.raises(TypeError, match='Not enough price arguments'):
        abstract.Function('SMA')()

    with pytest.raises(TypeError, match='Not enough price arguments'):
        abstract.Function('SMA')(10)


def test_threading():
    import pandas as pd
    TEST_LEN_SHORT = 999
    TEST_LEN_LONG = 4005
    LOOPS = 1000
    THREADS = 4

    data_short = np.random.rand(TEST_LEN_SHORT, 5)
    data_long = np.random.rand(TEST_LEN_LONG, 5)

    df_short = pd.DataFrame(data_short, columns=['open', 'high', 'low', 'close', 'volume'])
    df_long = pd.DataFrame(data_long, columns=['open', 'high', 'low', 'close', 'volume'])

    total = 0

    def loop(i):
        nonlocal total
        if i % 2 == 0:
            df = copy.deepcopy(df_short)
        else:
            df = copy.deepcopy(df_long)

        for _ in range(LOOPS):
            total += 1
            df['RSI'] = abstract.RSI(df)

    t0 = time.time()

    threads = []
    for i in range(THREADS):
        t = threading.Thread(target=lambda: loop(i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    t1 = time.time()
    print('test_len: %d, loops: %d' % (TEST_LEN_LONG, LOOPS))
    print('%.6f' % (t1 - t0))
    print('%.6f' % ((t1 - t0) / LOOPS))

    assert total == THREADS * LOOPS

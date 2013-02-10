import numpy as np

from collections import OrderedDict

import talib
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

def test_doji_candle():
    expected = func.CDLDOJI(ford_2012['open'], ford_2012['high'], ford_2012['low'], ford_2012['close'])
    got = abstract.Function('CDLDOJI').run(ford_2012)
    assert_np_arrays_equal(got, expected)

def test_info():
    stochrsi = abstract.Function('STOCHRSI')
    stochrsi.input_names = {'price': 'open'}
    stochrsi.parameters = {'fastd_matype': talib.MA_Type.EMA}
    expected = {
        'display_name': 'Stochastic Relative Strength Index',
        'flags': None,
        'group': 'Momentum Indicators',
        'input_names': OrderedDict([('price', 'open')]),
        'name': 'STOCHRSI',
        'output_names': ['fastk', 'fastd'],
        'parameters': OrderedDict([
            ('timeperiod', 14),
            ('fastk_period', 5),
            ('fastd_period', 3),
            ('fastd_matype', 1),
            ]),
        }
    assert(expected == stochrsi.info)

    expected = {
        'display_name': 'Bollinger Bands',
        'flags': None,
        'group': 'Overlap Studies',
        'input_names': OrderedDict([('price', 'close')]),
        'name': 'BBANDS',
        'output_names': ['upperband', 'middleband', 'lowerband'],
        'parameters': OrderedDict([
            ('timeperiod', 5),
            ('nbdevup', 2),
            ('nbdevdn', 2),
            ('matype', 0),
            ]),
        }
    assert(expected == abstract.Function('BBANDS').info)

def test_input_names():
    expected = OrderedDict([('price', 'close')])
    assert(expected == abstract.Function('MAMA').input_names)

    # test setting input_names
    obv = abstract.Function('OBV')
    expected = OrderedDict([
        ('price', 'open'),
        ('prices', ['volume']),
        ])
    obv.input_names = expected
    assert(obv.input_names == expected)

    obv.input_names = {
        'price': 'open',
        'prices': ['volume'],
        }
    assert(obv.input_names == expected)

def test_input_arrays():
    mama = abstract.Function('MAMA')
    # test default setting
    expected = {
        'open': None,
        'high': None,
        'low': None,
        'close': None,
        'volume': None,
        }
    assert(expected == mama.get_input_arrays())
    # test setting/getting input_arrays
    assert(mama.set_input_arrays(ford_2012))
    assert(mama.get_input_arrays() == ford_2012)
    assert(not mama.set_input_arrays({'hello': 'fail', 'world': 'bye'}))

def test_parameters():
    stoch = abstract.Function('STOCH')
    expected = OrderedDict([
        ('fastk_period', 5),
        ('slowk_period', 3),
        ('slowk_matype', 0),
        ('slowd_period', 3),
        ('slowd_matype', 0),
        ])
    assert(expected == stoch.parameters)

    stoch.parameters = {'fastk_period': 10}
    expected['fastk_period'] = 10
    assert(expected == stoch.parameters)

    stoch.parameters = {'slowk_period': 8, 'slowd_period': 5}
    expected['slowk_period'] = 8
    expected['slowd_period'] = 5
    assert(expected == stoch.parameters)

    stoch.parameters = {'slowd_matype': talib.MA_Type.T3}
    expected['slowd_matype'] = 8
    assert(expected == stoch.parameters)

    stoch.parameters = {
        'slowk_matype': talib.MA_Type.WMA,
        'slowd_matype': talib.MA_Type.EMA,
        }
    expected['slowk_matype'] = 2
    expected['slowd_matype'] = 1
    assert(expected == stoch.parameters)

def test_lookback():
    assert(abstract.Function('SMA', 10).lookback == 9)

    stochrsi = abstract.Function('stochrsi', 20, 5, 3)
    assert(stochrsi.lookback == 26)

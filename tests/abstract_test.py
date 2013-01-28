import unittest
import numpy as np

from talib import func
from talib import abstract

from data import ford_2012


class AbstractTestCase(unittest.TestCase):

    def test_SMA(self):
        expected = func.SMA(ford_2012['close'], 10)
        self.__assert_np_arrays_equal(expected, abstract.Function('sma', ford_2012, 10).outputs)
        self.__assert_np_arrays_equal(expected, abstract.Function('sma')(ford_2012, 10, price='close'))
        self.__assert_np_arrays_equal(expected, abstract.Function('sma')(ford_2012, timeperiod=10))
        expected = func.SMA(ford_2012['open'], 10)
        self.__assert_np_arrays_equal(expected, abstract.Function('sma', ford_2012, 10, price='open').outputs)
        self.__assert_np_arrays_equal(expected, abstract.Function('sma', price='low')(ford_2012, 10, price='open'))
        self.__assert_np_arrays_not_equal(expected, abstract.Function('sma', ford_2012, 10, price='open')(timeperiod=20))
        self.__assert_np_arrays_not_equal(expected, abstract.Function('sma', ford_2012)(10, price='close'))
        self.__assert_np_arrays_not_equal(expected, abstract.Function('sma', 10)(ford_2012, price='high'))
        self.__assert_np_arrays_not_equal(expected, abstract.Function('sma', price='low')(ford_2012, 10))

    def test_STOCH(self):
        # check defaults match
        expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close']) # 5, 3, 0, 3, 0
        got_k, got_d = abstract.Function('stoch', ford_2012).outputs
        self.__assert_np_arrays_equal(expected_k, got_k)
        self.__assert_np_arrays_equal(expected_d, got_d)

        expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'])
        got_k, got_d = abstract.Function('stoch', ford_2012)(5, 3, 0, 3, 0)
        self.__assert_np_arrays_equal(expected_k, got_k)
        self.__assert_np_arrays_equal(expected_d, got_d)

        expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'], 15)
        got_k, got_d = abstract.Function('stoch', ford_2012)(15, 5, 0, 5, 0)
        self.__assert_np_arrays_not_equal(expected_k, got_k)
        self.__assert_np_arrays_not_equal(expected_d, got_d)

        expected_k, expected_d = func.STOCH(ford_2012['high'], ford_2012['low'], ford_2012['close'], 15, 5, 1, 5, 1)
        got_k, got_d = abstract.Function('stoch', ford_2012)(15, 5, 1, 5, 1)
        self.__assert_np_arrays_equal(expected_k, got_k)
        self.__assert_np_arrays_equal(expected_d, got_d)

    def __assert_np_arrays_equal(self, expected, got):
        for i, value in enumerate(expected):
            if np.isnan(value):
                self.assertTrue(np.isnan(got[i]))
            else:
                self.assertTrue(value == got[i])

    def __assert_np_arrays_not_equal(self, expected, got):
        ''' Verifies expected and got have the same number of leading nan fields,
        followed by different floats.
        '''
        nans = []
        equals = []
        for i, value in enumerate(expected):
            if np.isnan(value):
                self.assertTrue(np.isnan(got[i]))
                nans.append(value)
            else:
                try:
                    assert(value != got[i])
                except AssertionError:
                    equals.append(got[i])
        if len(equals) == len(expected[len(nans):]):
            raise AssertionError('Arrays were equal.')
        elif equals:
            print 'Arrays had %i/%i equivalent values.' % (len(equals), len(expected[len(nans):]))


def get_test_cases():
    ret = []
    ret.append(AbstractTestCase('test_SMA'))
    ret.append(AbstractTestCase('test_STOCH'))
    return ret

if __name__ == '__main__':
    unittest.main()

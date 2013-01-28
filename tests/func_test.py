import unittest
import numpy as np

import talib
from talib import func

from data import series

class FuncTestCase(unittest.TestCase):

    def test_MIN(self):
        result = func.MIN(series, timeperiod=4)
        i = np.where(~np.isnan(result))[0][0]
        self.assertTrue(len(series) == len(result))
        self.assertEquals(result[i + 1], 93.780)
        self.assertEquals(result[i + 2], 93.780)
        self.assertEquals(result[i + 3], 92.530)
        self.assertEquals(result[i + 4], 92.530)

    def test_MAX(self):
        result = func.MAX(series, timeperiod=4)
        i = np.where(~np.isnan(result))[0][0]
        assert len(series) == len(result), (len(series), len(result))
        self.assertEquals(result[i + 2], 95.090)
        self.assertEquals(result[i + 3], 95.090)
        self.assertEquals(result[i + 4], 94.620)
        self.assertEquals(result[i + 5], 94.620)

    def test_BBANDS(self):
        upper, middle, lower = func.BBANDS(series, timeperiod=20,
                                            nbdevup=2.0, nbdevdn=2.0,
                                            matype=talib.MA_Type.EMA)
        i = np.where(~np.isnan(upper))[0][0]
        self.assertTrue(len(upper) == len(middle) == len(lower) == len(series))
        #self.assertTrue(abs(upper[i + 0] - 98.0734) < 1e-3)
        self.assertTrue(abs(middle[i + 0] - 92.8910) < 1e-3)
        self.assertTrue(abs(lower[i + 0] - 87.7086) < 1e-3)
        #self.assertTrue(abs(upper[i + 13] - 93.674) < 1e-3)
        self.assertTrue(abs(middle[i + 13] - 87.679) < 1e-3)
        self.assertTrue(abs(lower[i + 13] - 81.685) < 1e-3)

    def test_DEMA(self):
        result = func.DEMA(series)
        i = np.where(~np.isnan(result))[0][0]
        self.assertTrue(len(series) == len(result))
        self.assertTrue(abs(result[i + 1] - 86.765) < 1e-3)
        self.assertTrue(abs(result[i + 2] - 86.942) < 1e-3)
        self.assertTrue(abs(result[i + 3] - 87.089) < 1e-3)
        self.assertTrue(abs(result[i + 4] - 87.656) < 1e-3)

    def test_EMAEMA(self):
        result = func.EMA(series, timeperiod=2)
        result = func.EMA(result, timeperiod=2)
        i = np.where(~np.isnan(result))[0][0]
        self.assertTrue(len(series) == len(result))
        assert i == 2, i

def get_test_cases():
    ret = []
    ret.append(FuncTestCase('test_MIN'))
    ret.append(FuncTestCase('test_MAX'))
    ret.append(FuncTestCase('test_BBANDS'))
    ret.append(FuncTestCase('test_DEMA'))
    ret.append(FuncTestCase('test_EMAEMA'))
    return ret

if __name__ == '__main__':
    unittest.main()

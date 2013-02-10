from __future__ import print_function

import numpy
import talib
import sys

TEST_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
LOOPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

data = numpy.random.random(TEST_LEN)

if False: # fill array with nans
    data[:-1] = numpy.nan

import time
t0 = time.time()
for _ in range(LOOPS):
    talib.MA(data)
    talib.BBANDS(data)
    talib.KAMA(data)
    talib.CDLMORNINGDOJISTAR(data, data, data, data)
t1 = time.time()
print('%d' % TEST_LEN)
print('%.6f' % ((t1 - t0) / 1000.0))

import numpy
import talib
import sys

TEST_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

if True:  # test all floats
    data = numpy.random.random(TEST_LEN)
else:     # test all-but-one NaNs
    data = numpy.empty(TEST_LEN)
    data.fill(numpy.nan)
    data[-1] = numpy.randon.random()

import time
t0 = time.time()
for _ in range(1000):
    talib.MA(data)
    talib.BBANDS(data)
    talib.KAMA(data)
    talib.CDLMORNINGDOJISTAR(data, data, data, data)
t1 = time.time()
print '%d' % TEST_LEN
print '%.6f' % ((t1 - t0) / 1000.)

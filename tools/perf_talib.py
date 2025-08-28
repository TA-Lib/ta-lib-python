import numpy
import talib
import time
import sys

TEST_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
LOOPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

data = numpy.random.random(TEST_LEN)

if False: # fill array with nans
    data[:-1] = numpy.nan

t0 = time.time()
for _ in range(LOOPS):
    talib.MA(data)
    talib.BBANDS(data)
    talib.KAMA(data)
    talib.CDLMORNINGDOJISTAR(data, data, data, data)
t1 = time.time()
print('test_len: %d, loops: %d' % (TEST_LEN, LOOPS))
print('%.6f' % (t1 - t0))
print('%.6f' % ((t1 - t0) / LOOPS))

import numpy

import talib

TEST_LEN = 10000

r = numpy.arange(TEST_LEN)
idata = numpy.random.random((TEST_LEN))

import time
t0 = time.time()
for _ in range(1000):
    (i, n, odata) = talib.MA(idata)
    (i, n, upper, middle, lower) = talib.BBANDS(idata)
    (i, n, kama) = talib.KAMA(idata)
t1 = time.time()
print '%.6f' % ((t1 - t0) / 1000.)

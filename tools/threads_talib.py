from __future__ import print_function

import numpy
from numpy.testing import assert_array_equal
import talib
import sys

TEST_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
LOOPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

data = numpy.random.random(TEST_LEN)

ref1 = talib.MA(data)
ref2, ref3, ref4 = talib.BBANDS(data)
ref5 = talib.KAMA(data)
ref6 = talib.CDLMORNINGDOJISTAR(data, data, data, data)

import threading

total = 0

def loop():
    global total
    while total < LOOPS:
        total += 1
        out1 = talib.MA(data)
        assert_array_equal(out1, ref1)
        out2, out3, out4 = talib.BBANDS(data)
        assert_array_equal(out2, ref2)
        assert_array_equal(out3, ref3)
        assert_array_equal(out4, ref4)
        out5 = talib.KAMA(data)
        assert_array_equal(out5, ref5)
        out6 = talib.CDLMORNINGDOJISTAR(data, data, data, data)
        assert_array_equal(out6, ref6)

import time
t0 = time.time()

threads = []
for i in range(10):
    t = threading.Thread(target=loop)
    threads.append(t)
    t.start()

for t in threads:
    t.join()
t1 = time.time()
print('test_len: %d, loops: %d' % (TEST_LEN, LOOPS))
print('%.6f' % (t1 - t0))
print('%.6f' % ((t1 - t0) / LOOPS))

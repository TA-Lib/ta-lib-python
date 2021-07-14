from __future__ import print_function

import numpy
import talib
import sys

TEST_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
LOOPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

data = numpy.random.random(TEST_LEN)

import threading

total = 0

def loop():
    global total
    while total < LOOPS:
        total += 1
        talib.MA(data)
        talib.BBANDS(data)
        talib.KAMA(data)
        talib.CDLMORNINGDOJISTAR(data, data, data, data)

import time
t0 = time.time()

threads = []
for i in range(5):
    t = threading.Thread(target=loop)
    threads.append(t)
    t.start()

for t in threads:
    t.join()
t1 = time.time()
print('test_len: %d, loops: %d' % (TEST_LEN, LOOPS))
print('%.6f' % (t1 - t0))
print('%.6f' % ((t1 - t0) / LOOPS))

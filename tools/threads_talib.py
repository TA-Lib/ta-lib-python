import time
import threading

import numpy
import copy
from talib.abstract import RSI
import sys
import pandas as pd

TEST_LEN_SHORT = int(sys.argv[1]) if len(sys.argv) > 1 else 999
TEST_LEN_LONG = int(sys.argv[1]) if len(sys.argv) > 1 else 4005
LOOPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

data_short = numpy.random.rand(TEST_LEN_SHORT, 5)
data_long = numpy.random.rand(TEST_LEN_LONG, 5)

df_short = pd.DataFrame(data_short, columns={
                        'open', 'high', 'low', 'close', 'volume'})
df_long = pd.DataFrame(data_long, columns={
                       'open', 'high', 'low', 'close', 'volume'})

total = 0


def loop():
    global total
    if threading.get_native_id() % 2 == 0:
        df = copy.deepcopy(df_short)
    else:
        df = copy.deepcopy(df_long)

    while total < LOOPS:
        total += 1
        try:
            df['RSI'] = RSI(df)
        except ValueError as msg:
            raise ValueError(msg)


t0 = time.time()

threads = []
for i in range(4):
    t = threading.Thread(target=loop)
    threads.append(t)
    t.start()

for t in threads:
    t.join()
t1 = time.time()
print('test_len: %d, loops: %d' % (TEST_LEN_LONG, LOOPS))
print('%.6f' % (t1 - t0))
print('%.6f' % ((t1 - t0) / LOOPS))

import numpy
import pylab

import talib

TEST_LEN = 100

r = numpy.arange(TEST_LEN)
idata = numpy.random.random((TEST_LEN))

i, odata = talib.MA(idata)

j, upper, middle, lower = talib.BBANDS(idata)

k, kama = talib.KAMA(idata)

pylab.plot(r, idata, 'b-', label="original")
pylab.plot(r[i:], odata, 'g-', label="MA")
pylab.plot(r[j:], upper, 'r-', label="Upper")
pylab.plot(r[j:], middle, 'r-', label="Middle")
pylab.plot(r[j:], lower, 'r-', label="Lower")
pylab.plot(r[k:], kama, 'g', label="KAMA")
pylab.legend()
pylab.show()


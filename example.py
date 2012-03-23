import numpy
import pylab

import talib

TEST_LEN = 100

r = numpy.arange(TEST_LEN)
idata = numpy.random.random((TEST_LEN))

i, odata = talib.MA(idata)

i, upper, middle, lower = talib.BBANDS(idata)

i, kama = talib.KAMA(idata)

pylab.plot(r, idata, 'b-', label="original")
pylab.plot(r[-len(odata):], odata, 'g-', label="MA")
pylab.plot(r[-len(upper):], upper, 'r-', label="Upper")
pylab.plot(r[-len(middle):], middle, 'r-', label="Middle")
pylab.plot(r[-len(lower):], lower, 'r-', label="Lower")
pylab.plot(r[-len(kama):], kama, 'g', label="KAMA")
pylab.legend()
pylab.show()


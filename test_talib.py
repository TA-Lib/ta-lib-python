import numpy
import pylab

import talib

TEST_LEN = 1000

r = numpy.arange(TEST_LEN)
idata = numpy.random.random((TEST_LEN))

(bidx, elements, odata) = talib.ma(idata)

(bidx, elements, upper, middle, lower) = talib.bbands(idata)

(bidx, elements, kama) = talib.kama(idata)

pylab.plot(r, idata, 'b-', label="original")
pylab.plot(r, odata, 'g-', label="MA")
pylab.plot(r, upper, 'r-', label="Upper")
pylab.plot(r, middle, 'r-', label="Middle")
pylab.plot(r, lower, 'r-', label="Lower")
pylab.plot(r, kama, 'g', label="KAMA")
pylab.legend()
pylab.show()


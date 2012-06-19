import numpy
import talib
import pylab
import sys

TEST_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 100

r = numpy.arange(TEST_LEN)

idata = numpy.random.random(TEST_LEN)

odata = talib.MA(idata)

upper, middle, lower = talib.BBANDS(idata)

kama = talib.KAMA(idata)

pylab.plot(r, idata, 'b-', label="original")
pylab.plot(r, odata, 'g-', label="MA")
pylab.plot(r, upper, 'r-', label="Upper")
pylab.plot(r, middle, 'r-', label="Middle")
pylab.plot(r, lower, 'r-', label="Lower")
pylab.plot(r, kama, 'g', label="KAMA")
pylab.legend()
pylab.show()


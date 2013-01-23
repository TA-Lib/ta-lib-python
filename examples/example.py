import numpy
import talib
from talib import abstract
from talib.abstract import Function
import pylab
import sys

TEST_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 100
r = numpy.arange(TEST_LEN)
idata = numpy.random.random(TEST_LEN)

def talib_example():
    odata = talib.MA(idata)
    upper, middle, lower = talib.BBANDS(idata)
    kama = talib.KAMA(idata)
    plot(odata, upper, middle, lower, kama)

def abstract_example():
    sma = Function('ma')
    input_arrays = sma.get_input_arrays()
    for key in input_arrays.keys():
        input_arrays[key] = idata
    sma.set_input_arrays(input_arrays)
    odata = sma(30)['real'] # timePeriod=30, specified as an arg. output selected by name

    bbands = Function('bbands', input_arrays)
    bbands.set_function_parameters(timePeriod=20, nbDevUp=2, nbDevDown=2)
    upper, middle, lower = bbands().values() # multiple output values unpacked (these will always have the correct order)

    kama = Function('kama').run(input_arrays).values()[0] # generic-name output selected. alternative run() calling method.
    plot(odata, upper, middle, lower, kama)

def plot(odata, upper, middle, lower, kama):
    pylab.plot(r, idata, 'b-', label="original")
    pylab.plot(r, odata, 'g-', label="MA")
    pylab.plot(r, upper, 'r-', label="Upper")
    pylab.plot(r, middle, 'r-', label="Middle")
    pylab.plot(r, lower, 'r-', label="Lower")
    pylab.plot(r, kama, 'g', label="KAMA")
    pylab.legend()
    pylab.show()

if __name__ == '__main__':
    print 'All functions (sorted by group):'
    groups = abstract.get_groups_of_functions()
    for group, functions in sorted(groups.items()):
        print '%s functions: ' % group, functions

    talib_example()
    #abstract_example()

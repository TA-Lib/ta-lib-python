
import re

# fixme: use lookback to do exact allocations

functions = []
with open('/usr/local/include/ta-lib/ta_func.h') as f:
    tmp = []
    for line in f:
        line = line.strip()
        if tmp or \
            line.startswith('TA_RetCode TA_') or \
            line.startswith('int TA_'):
            line = re.sub('/\*[^\*]+\*/', '', line) # strip comments
            tmp.append(line)
            if not line:
                s = ' '.join(tmp)
                s = re.sub('\s+', ' ', s)
                functions.append(s)
                tmp = []

# strip "float" functions
functions = [s for s in functions if not s.startswith('TA_RetCode TA_S_')]

# strip non-indicators
functions = [s for s in functions if not s.startswith('TA_RetCode TA_Set')]
functions = [s for s in functions if not s.startswith('TA_RetCode TA_Restore')]

# print headers
print """
import numpy
cimport numpy as np

ctypedef int TA_RetCode
ctypedef int TA_MAType

# extract the needed part of ta_libc.h that I will use in the interface
cdef extern from "ta_libc.h":
    enum: TA_SUCCESS
    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()"""

# ! can't use const in function declaration (cython 0.12 restriction)
# just removing them does the trick
for f in functions:
    print '    %s' % f.replace('const', '').replace(';', '').replace('void', '')
print

inputs = {
    'const double inReal[]' : 'np.ndarray[np.float_t, ndim=1] inReal',
    'const double inReal0[]' : 'np.ndarray[np.float_t, ndim=1] inReal0',
    'const double inReal1[]' : 'np.ndarray[np.float_t, ndim=1] inReal1',
    'const double inOpen[]' : 'np.ndarray[np.float_t, ndim=1] inOpen',
    'const double inHigh[]' : 'np.ndarray[np.float_t, ndim=1] inHigh',
    'const double inLow[]' : 'np.ndarray[np.float_t, ndim=1] inLow',
    'const double inClose[]' : 'np.ndarray[np.float_t, ndim=1] inClose',
    'const double inVolume[]' : 'np.ndarray[np.float_t, ndim=1] inVolume',
    'const double inPeriods[]' : 'np.ndarray[np.float_t, ndim=1] inPeriods',
    'int optInFastPeriod' : 'optInFastPeriod=1',
    'int optInSlowPeriod' : 'optInSlowPeriod=1',
    'int optInTimePeriod' : 'optInTimePeriod=1',
    'int optInFastK_Period' : 'optInFastK_Period=1',
    'int optInFastD_Period' : 'optInFastD_Period=1',
    'int optInSlowK_Period' : 'optInSlowK_Period=1',
    'int optInSlowD_Period' : 'optInSlowD_Period=1',
    'int optInTimePeriod1' : 'optInTimePeriod1',
    'int optInTimePeriod2' : 'optInTimePeriod2',
    'int optInTimePeriod3' : 'optInTimePeriod3',
    'double optInNbDev' : 'optInNbDev=1',
    'double optInNbDevUp' : 'optInNbDevUp=1',
    'double optInNbDevDn' : 'optInNbDevDn=1',
    'double optInPenetration' : 'optInPenetration',
    'double optInAcceleration' : 'optInAcceleration',
    'double optInAccelerationInitLong' : 'optInAccelerationInitLong',
    'double optInAccelerationInitShort' : 'optInAccelerationInitShort',
    'double optInAccelerationLong' : 'optInAccelerationLong',
    'double optInAccelerationShort' : 'optInAccelerationShort',
    'double optInAccelerationMaxLong' : 'optInAccelerationMaxLong',
    'double optInAccelerationMaxShort' : 'optInAccelerationMaxShort',
    'double optInMaximum' : 'optInMaximum',
    'double optInStartValue' : 'optInStartValue',
    'double optInOffsetOnReverse' : 'optInOffsetOnReverse',
    'double optInFastLimit' : 'optInFastLimit',
    'double optInSlowLimit' : 'optInSlowLimit',
    'double optInVFactor' : 'optInVFactor=1',
    'int optInSignalPeriod' : 'optInSignalPeriod=1',
    'int optInMinPeriod' : 'optInMinPeriod',
    'int optInMaxPeriod' : 'optInMaxPeriod',
    'TA_MAType optInMAType' : 'optInMAType=1', # fixme: default values
    'TA_MAType optInFastMAType' : 'optInFastMAType=1', # fixme: default values
    'TA_MAType optInSlowMAType' : 'optInSlowMAType=1', # fixme: default values
    'TA_MAType optInSignalMAType' : 'optInSignalMAType=1', # fixme: default values
    'TA_MAType optInFastK_MAType' : 'optInFastK_MAType=1', # fixme: default values
    'TA_MAType optInFastD_MAType' : 'optInFastD_MAType=1', # fixme: default values
    'TA_MAType optInSlowK_MAType' : 'optInSlowK_MAType=1', # fixme: default values
    'TA_MAType optInSlowD_MAType' : 'optInSlowD_MAType=1', # fixme: default values
}

outputs = {
    'int *outBegIdx' : 'cdef int outBegIdx',
    'int *outNBElement' : 'cdef int outNBElement',
    'int outInteger[]' : 'cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)',
    'int outMinIdx[]' : 'cdef np.ndarray[np.int_t, ndim=1] outMinIdx = numpy.zeros(allocationSize)',
    'int outMaxIdx[]' : 'cdef np.ndarray[np.int_t, ndim=1] outMaxIdx = numpy.zeros(allocationSize)',
    'double outReal[]' : 'cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)',
    'double outRealUpperBand[]' : 'cdef np.ndarray[np.float_t, ndim=1] outRealUpperBand = numpy.zeros(allocationSize)',
    'double outRealMiddleBand[]' : 'cdef np.ndarray[np.float_t, ndim=1] outRealMiddleBand = numpy.zeros(allocationSize)',
    'double outRealLowerBand[]' : 'cdef np.ndarray[np.float_t, ndim=1] outRealLowerBand = numpy.zeros(allocationSize)',
    'double outAroonDown[]' : 'cdef np.ndarray[np.float_t, ndim=1] outAroonDown = numpy.zeros(allocationSize)',
    'double outAroonUp[]' : 'cdef np.ndarray[np.float_t, ndim=1] outAroonUp = numpy.zeros(allocationSize)',
    'double outInPhase[]' : 'cdef np.ndarray[np.float_t, ndim=1] outInPhase = numpy.zeros(allocationSize)',
    'double outQuadrature[]' : 'cdef np.ndarray[np.float_t, ndim=1] outQuadrature = numpy.zeros(allocationSize)',
    'double outSine[]' : 'cdef np.ndarray[np.float_t, ndim=1] outSine = numpy.zeros(allocationSize)',
    'double outLeadSine[]' : 'cdef np.ndarray[np.float_t, ndim=1] outLeadSine = numpy.zeros(allocationSize)',
    'double outMACD[]' : 'cdef np.ndarray[np.float_t, ndim=1] outMACD = numpy.zeros(allocationSize)',
    'double outMACDSignal[]' : 'cdef np.ndarray[np.float_t, ndim=1] outMACDSignal = numpy.zeros(allocationSize)',
    'double outMACDHist[]' : 'cdef np.ndarray[np.float_t, ndim=1] outMACDHist = numpy.zeros(allocationSize)',
    'double outMAMA[]' : 'cdef np.ndarray[np.float_t, ndim=1] outMAMA = numpy.zeros(allocationSize)',
    'double outFAMA[]' : 'cdef np.ndarray[np.float_t, ndim=1] outFAMA = numpy.zeros(allocationSize)',
    'double outMax[]' : 'cdef np.ndarray[np.float_t, ndim=1] outMax = numpy.zeros(allocationSize)',
    'double outMin[]' : 'cdef np.ndarray[np.float_t, ndim=1] outMin = numpy.zeros(allocationSize)',
    'double outFastK[]' : 'cdef np.ndarray[np.float_t, ndim=1] outFastK = numpy.zeros(allocationSize)',
    'double outFastD[]' : 'cdef np.ndarray[np.float_t, ndim=1] outFastD = numpy.zeros(allocationSize)',
    'double outSlowK[]' : 'cdef np.ndarray[np.float_t, ndim=1] outSlowK = numpy.zeros(allocationSize)',
    'double outSlowD[]' : 'cdef np.ndarray[np.float_t, ndim=1] outSlowD = numpy.zeros(allocationSize)',
}

params = {
    'int startIdx' : 'startIdx',
    'int endIdx' : 'endIdx',
    'const double inReal[]' : '<double *>inReal.data',
    'const double inReal0[]' : '<double *>inReal0.data',
    'const double inReal1[]' : '<double *>inReal1.data',
    'const double inOpen[]' : '<double *>inOpen.data',
    'const double inHigh[]' : '<double *>inHigh.data',
    'const double inLow[]' : '<double *>inLow.data',
    'const double inClose[]' : '<double *>inClose.data',
    'const double inVolume[]' : '<double *>inVolume.data',
    'const double inPeriods[]' : '<double *>inPeriods.data',
    'int optInFastPeriod' : 'optInFastPeriod',
    'int optInSlowPeriod' : 'optInSlowPeriod',
    'int optInTimePeriod' : 'optInTimePeriod',
    'int optInFastK_Period' : 'optInFastK_Period',
    'int optInFastD_Period' : 'optInFastD_Period',
    'int optInSlowK_Period' : 'optInSlowK_Period',
    'int optInSlowD_Period' : 'optInSlowD_Period',
    'int optInTimePeriod1' : 'optInTimePeriod1',
    'int optInTimePeriod2' : 'optInTimePeriod2',
    'int optInTimePeriod3' : 'optInTimePeriod3',
    'double optInNbDev' : 'optInNbDev',
    'double optInNbDevUp' : 'optInNbDevUp',
    'double optInNbDevDn' : 'optInNbDevDn',
    'double optInPenetration' : 'optInPenetration',
    'double optInAcceleration' : 'optInAcceleration',
    'double optInAccelerationInitLong' : 'optInAccelerationInitLong',
    'double optInAccelerationInitShort' : 'optInAccelerationInitShort',
    'double optInAccelerationLong' : 'optInAccelerationLong',
    'double optInAccelerationShort' : 'optInAccelerationShort',
    'double optInAccelerationMaxLong' : 'optInAccelerationMaxLong',
    'double optInAccelerationMaxShort' : 'optInAccelerationMaxShort',
    'double optInMaximum' : 'optInMaximum',
    'double optInStartValue' : 'optInStartValue',
    'double optInOffsetOnReverse' : 'optInOffsetOnReverse',
    'double optInFastLimit' : 'optInFastLimit',
    'double optInSlowLimit' : 'optInSlowLimit',
    'double optInVFactor' : 'optInVFactor',
    'int optInSignalPeriod' : 'optInSignalPeriod',
    'int optInMinPeriod' : 'optInMinPeriod',
    'int optInMaxPeriod' : 'optInMaxPeriod',
    'TA_MAType optInMAType' : 'optInMAType',
    'TA_MAType optInFastMAType' : 'optInFastMAType',
    'TA_MAType optInSlowMAType' : 'optInSlowMAType',
    'TA_MAType optInSignalMAType' : 'optInSignalMAType',
    'TA_MAType optInFastK_MAType' : 'optInFastK_MAType',
    'TA_MAType optInFastD_MAType' : 'optInFastD_MAType',
    'TA_MAType optInSlowK_MAType' : 'optInSlowK_MAType',
    'TA_MAType optInSlowD_MAType' : 'optInSlowD_MAType',
    'int *outBegIdx' : '&outBegIdx',
    'int *outNBElement' : '&outNBElement',
    'int outInteger[]' : '<int *>outInteger.data',
    'int outMinIdx[]' : '<int *>outMinIdx.data',
    'int outMaxIdx[]' : '<int *>outMaxIdx.data',
    'double outReal[]' : '<double *>outReal.data',
    'double outRealUpperBand[]' : '<double *>outRealUpperBand.data',
    'double outRealMiddleBand[]' : '<double *>outRealMiddleBand.data',
    'double outRealLowerBand[]' : '<double *>outRealLowerBand.data',
    'double outAroonDown[]' : '<double *>outAroonDown.data',
    'double outAroonUp[]' : '<double *>outAroonUp.data',
    'double outInPhase[]' : '<double *>outInPhase.data',
    'double outQuadrature[]' : '<double *>outQuadrature.data',
    'double outSine[]' : '<double *>outSine.data',
    'double outLeadSine[]' : '<double *>outLeadSine.data',
    'double outMACD[]' : '<double *>outMACD.data',
    'double outMACDSignal[]' : '<double *>outMACDSignal.data',
    'double outMACDHist[]' : '<double *>outMACDHist.data',
    'double outMAMA[]' : '<double *>outMAMA.data',
    'double outFAMA[]' : '<double *>outFAMA.data',
    'double outMax[]' : '<double *>outMax.data',
    'double outMin[]' : '<double *>outMin.data',
    'double outFastK[]' : '<double *>outFastK.data',
    'double outFastD[]' : '<double *>outFastD.data',
    'double outSlowK[]' : '<double *>outSlowK.data',
    'double outSlowD[]' : '<double *>outSlowD.data',
}

returns = {
    'int *outBegIdx' : 'outBegIdx',
    'int *outNBElement' : 'outNBElement',
    'int outInteger[]' : 'outInteger',
    'int outMinIdx[]' : 'outMinIdx',
    'int outMaxIdx[]' : 'outMaxIdx',
    'double outReal[]' : 'outReal',
    'double outRealUpperBand[]' : 'outRealUpperBand',
    'double outRealMiddleBand[]' : 'outRealMiddleBand',
    'double outRealLowerBand[]' : 'outRealLowerBand',
    'double outAroonDown[]' : 'outAroonDown',
    'double outAroonUp[]' : 'outAroonUp',
    'double outInPhase[]' : 'outInPhase',
    'double outQuadrature[]' : 'outQuadrature',
    'double outSine[]' : 'outSine',
    'double outLeadSine[]' : 'outLeadSine',
    'double outMACD[]' : 'outMACD',
    'double outMACDSignal[]' : 'outMACDSignal',
    'double outMACDHist[]' : 'outMACDHist',
    'double outMAMA[]' : 'outMAMA',
    'double outFAMA[]' : 'outFAMA',
    'double outMax[]' : 'outMax',
    'double outMin[]' : 'outMin',
    'double outFastK[]' : 'outFastK',
    'double outFastD[]' : 'outFastD',
    'double outSlowK[]' : 'outSlowK',
    'double outSlowD[]' : 'outSlowD',
}

# print functions
for f in functions:
    if 'Lookback' in f: # skip lookback functions
        continue
    i = f.index('(')
    name = f[:i].split()[1]
    args = f[i:].split(',')
    args = [re.sub('[\(\);]', '', s).strip() for s in args]
    print "def %s(" % name[3:],
    i = 0
    for arg in args:
        s = inputs.get(arg)
        if s is not None:
            if i > 0:
                print ',',
            i += 1
            print s,
        else:
            assert re.match('.*(void|startIdx|endIdx|out)/*', arg), arg
    print '):'

    print '    cdef int startIdx = 0'
    for arg in args:
        if 'inReal1' in arg:
            print '    cdef int endIdx = inReal1.shape[0] - 1'
            break
        elif 'inReal0' in arg:
            print '    cdef int endIdx = inReal0.shape[0] - 1'
            break
        elif 'inReal' in arg:
            print '    cdef int endIdx = inReal.shape[0] - 1'
            break
        elif 'inHigh' in arg:
            print '    cdef int endIdx = inHigh.shape[0] - 1'
            break

    print '    cdef int lookback = %s_Lookback(' % name,
    opts = [arg for arg in args if 'opt' in arg]
    for i, opt in enumerate(opts):
        if i > 0:
            print ',',
        print opt.split()[-1],
    print ')'
    print '    cdef int temp = max(lookback, startIdx )'
    print '    cdef int allocationSize'
    print '    if ( temp > endIdx ):'
    print '        allocationSize = 0'
    print '    else:'
    print '        allocationSize = endIdx - temp + 1'

    for arg in args:
        s = outputs.get(arg)
        if s:
            print '    %s' % s
        else:
            assert re.match('.*(void|startIdx|endIdx|opt|in)/*', arg), arg

    print '    retCode = TA_Initialize()'
    print '    if retCode != TA_SUCCESS:'
    print '        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)'
    print '    else:'
    print '        retCode = %s(' % name,
    for i, arg in enumerate(args):
        if i > 0:
            print ',',
        print params[arg],
    print ')'
    print '    TA_Shutdown()'

    print '    return (',
    i = 0
    for arg in args:
        s = returns.get(arg)
        if s:
            if i > 0:
                print ',',
            i += 1
            print s,
        else:
            assert re.match('.*(void|startIdx|endIdx|opt|in)/*', arg), arg
    print ')'
    print

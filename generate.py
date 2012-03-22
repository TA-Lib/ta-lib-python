
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

# TA_MAType enums
Sma, Ema, Wma, Dema, Tema, Trima, Kama, Mama, T3 = range(9)

RetCodes = {
  0 : "Success",
  1 : "LibNotInitialize",
  2 : "BadParam",
  3 : "AllocErr",
  4 : "GroupNotFound",
  5 : "FuncNotFound",
  6 : "InvalidHandle",
  7 : "InvalidParamHolder",
  8 : "InvalidParamHolderType",
  9 : "InvalidParamFunction",
  10 : "InputNotAllInitialize",
  11 : "OutputNotAllInitialize",
  12 : "OutOfRangeStartIndex",
  13 : "OutOfRangeEndIndex",
  14 : "InvalidListType",
  15 : "BadObject",
  16 : "NotSupported",
  5000 : "InternalError",
  0xFFFF : "UnknownErr",
}

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

# print functions
names = []
for f in functions:

    if 'Lookback' in f: # skip lookback functions
        continue

    i = f.index('(')
    name = f[:i].split()[1]
    args = f[i:].split(',')
    args = [re.sub('[\(\);]', '', s).strip() for s in args]

    names.append(name[3:].lower())
    print "def %s(" % name[3:].lower(),
    i = 0
    for arg in args:
        var = arg.split()[-1]

        if var in ('startIdx', 'endIdx'):
            continue

        elif 'out' in var:
            break

        if i > 0:
            print ',',
        i += 1

        if var.endswith('[]'):
            assert arg.startswith('const double'), arg
            print 'np.ndarray[np.float_t, ndim=1] %s' % var[:-2],

        elif var.startswith('opt'):
            if arg.startswith('double'):
                print '%s=-4e37' % var,  # TA_REAL_DEFAULT
            elif arg.startswith('int'):
                print '%s=-2**31' % var, # TA_INTEGER_DEFAULT
            elif arg.startswith('TA_MAType'):
                print '%s=0' % var,      # TA_MAType_SMA
            else:
                assert False, arg

    print '):'

    print '    cdef int startIdx = 0'
    for arg in args:
        var = arg.split()[-1]
        if var in ('inReal0[]', 'inReal1[]', 'inReal[]', 'inHigh[]'):
            print '    cdef int endIdx = %s.shape[0] - 1' % var[:-2]
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
        var = arg.split()[-1]

        if 'out' not in var:
            continue

        if var.endswith('[]'):
            if 'double' in arg:
                vartype = 'np.float_t'
            elif 'int' in arg:
                vartype = 'np.int_t'
            else:
                assert False, args
            print '    cdef np.ndarray[%s, ndim=1] %s = numpy.zeros(allocationSize)' % (vartype, var[:-2])

        elif var.startswith('*'):
            print '    cdef int %s' % var[1:]

        else:
            assert False, arg

    print '    retCode = TA_Initialize()'
    print '    if retCode != TA_SUCCESS:'
    print '        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))'
    print '    retCode = %s(' % name,

    for i, arg in enumerate(args):
        if i > 0:
            print ',',
        var = arg.split()[-1]

        if var.endswith('[]'):
            if 'double' in arg:
                print '<double *>%s.data' % var[:-2],
            elif 'int' in arg:
                print '<int *>%s.data' % var[:-2],
            else:
                assert False, arg

        elif var.startswith('*'):
            print '&%s' % var[1:],

        else:
            print var,

    print ')'
    print '    if retCode != TA_SUCCESS:'
    print '        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))'
    print '    TA_Shutdown()'

    print '    return (',
    i = 0
    for arg in args:
        var = arg.split()[-1]
        if var.endswith('[]'):
            var = var[:-2]
        elif var.startswith('*'):
            var = var[1:]
        if var.startswith('out'):
            if i > 0:
                print ',',
            i += 1
            print var,
        else:
            assert re.match('.*(void|startIdx|endIdx|opt|in)/*', arg), arg
    print ')'
    print

print '__all__ = [%s]' % ','.join(names)

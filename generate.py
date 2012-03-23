
import re

# FIXME: initialize once, then shutdown at the end, rather than each call?
# FIXME: should we check retCode from initialize and shutdown?

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
MA_SMA, MA_EMA, MA_WMA, MA_DEMA, MA_TEMA, MA_TRIMA, MA_KAMA, MA_MAMA, MA_T3 = range(9)

# extract the needed part of ta_libc.h that I will use in the interface
cdef extern from "ta_libc.h":
    enum: TA_SUCCESS
    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()
    char *TA_GetVersionString()"""

# ! can't use const in function declaration (cython 0.12 restriction)
# just removing them does the trick
for f in functions:
    f = f.replace('const', '')
    f = f.replace(';', '')
    f = f.replace('void', '')
    f = f.strip()
    print '    %s' % f
print

print """
__version__ = TA_GetVersionString()
"""

# cleanup variable names to make them more pythonic
def cleanup(name):
    if name.startswith('in'):
        return name[2:].lower()
    elif name.startswith('optIn'):
        return name[5:].lower()
    else:
        return name.lower()

# print functions
names = []
for f in functions:

    if 'Lookback' in f: # skip lookback functions
        continue

    i = f.index('(')
    name = f[:i].split()[1]
    args = f[i:].split(',')
    args = [re.sub('[\(\);]', '', s).strip() for s in args]

    names.append(name[3:])
    print "def %s(" % name[3:],
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
            var = cleanup(var[:-2])
            assert arg.startswith('const double'), arg
            print 'np.ndarray[np.float_t, ndim=1] %s' % var,

        elif var.startswith('opt'):
            var = cleanup(var)
            if arg.startswith('double'):
                print '%s=-4e37' % var,  # TA_REAL_DEFAULT
            elif arg.startswith('int'):
                print '%s=-2**31' % var, # TA_INTEGER_DEFAULT
            elif arg.startswith('TA_MAType'):
                print '%s=0' % var,      # TA_MAType_SMA
            else:
                assert False, arg

    print '):'

    print '    cdef int startidx = 0'
    for arg in args:
        var = arg.split()[-1]
        if var in ('inReal0[]', 'inReal1[]', 'inReal[]', 'inHigh[]'):
            var = cleanup(var[:-2])
            print '    cdef int endidx = %s.shape[0] - 1' % var
            break

    print '    cdef int lookback = %s_Lookback(' % name,
    opts = [arg for arg in args if 'opt' in arg]
    for i, opt in enumerate(opts):
        if i > 0:
            print ',',
        print cleanup(opt.split()[-1]),
    print ')'
    print '    cdef int temp = max(lookback, startidx )'
    print '    cdef int allocation'
    print '    if ( temp > endidx ):'
    print '        allocation = 0'
    print '    else:'
    print '        allocation = endidx - temp + 1'

    for arg in args:
        var = arg.split()[-1]

        if 'out' not in var:
            continue

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                vartype = 'np.float_t'
            elif 'int' in arg:
                vartype = 'np.int_t'
            else:
                assert False, args
            print '    cdef np.ndarray[%s, ndim=1] %s = numpy.zeros(allocation)' % (vartype, var)

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print '    cdef int %s' % var

        else:
            assert False, arg

    print '    TA_Initialize()'
    print '    retCode = %s(' % name,

    for i, arg in enumerate(args):
        if i > 0:
            print ',',
        var = arg.split()[-1]

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print '<double *>%s.data' % var,
            elif 'int' in arg:
                print '<int *>%s.data' % var,
            else:
                assert False, arg

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print '&%s' % var,

        else:
            print cleanup(var),

    print ')'
    print '    if retCode != TA_SUCCESS:'
    print '        raise Exception("%d" % retCode)'
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
            print cleanup(var),
        else:
            assert re.match('.*(void|startIdx|endIdx|opt|in)/*', arg), arg
    print ')'
    print

print '__all__ = [%s]' % ','.join(names)

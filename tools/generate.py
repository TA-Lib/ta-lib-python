import os
import re
import sys

from talib import abstract

# FIXME: initialize once, then shutdown at the end, rather than each call?
# FIXME: should we pass startIdx and endIdx into function?
# FIXME: don't return number of elements since it always equals allocation?

functions = []
include_paths = ['/usr/include', '/usr/local/include']
ta_func_header = None
for path in include_paths:
    if os.path.exists(path + '/ta-lib/ta_func.h'):
        ta_func_header = path + '/ta-lib/ta_func.h'
        break
if not ta_func_header:
    print >> sys.stderr, 'Error: ta-lib/ta_func.h not found'
    sys.exit(1)
with open(ta_func_header) as f:
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
print """\
import talib # unused but we import anyway to make sure initialize and shutdown are handled correctly
cimport numpy as np
from numpy import nan
from cython import boundscheck, wraparound

from common_c import _ta_check_success

ctypedef np.double_t double_t
ctypedef np.int32_t int32_t

ctypedef int TA_RetCode
ctypedef int TA_MAType

cdef double NaN = nan

cdef extern from "math.h":
    bint isnan(double x)

cdef extern from "numpy/arrayobject.h":
    int PyArray_TYPE(np.ndarray)
    object PyArray_EMPTY(int, np.npy_intp*, int, int)
    int PyArray_FLAGS(np.ndarray)
    object PyArray_GETCONTIGUOUS(np.ndarray)

np.import_array() # Initialize the NumPy C API

# extract the needed part of ta_libc.h that I will use in the interface
cdef extern from "ta-lib/ta_libc.h":
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

    shortname = name[3:]
    names.append(shortname)
    func_info = abstract.Function(shortname).info
    defaults, documentation = abstract._get_defaults_and_docs(func_info)

    print '@wraparound(False)  # turn off relative indexing from end of lists'
    print '@boundscheck(False) # turn off bounds-checking for entire function'
    print 'def %s(' % shortname,
    docs = ['%s(' % shortname]
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
            print 'np.ndarray %s not None' % var,
            docs.append(var)
            docs.append(', ')

        elif var.startswith('opt'):
            var = cleanup(var)
            default_arg = arg.split()[-1][len('optIn'):] # chop off typedef and 'optIn'
            default_arg = default_arg[0].lower() + default_arg[1:] # lowercase first letter

            if arg.startswith('double'):
                if default_arg in defaults:
                    print 'double %s=%s' % (var, defaults[default_arg]),
                else:
                    print 'double %s=-4e37' % var, # TA_REAL_DEFAULT
            elif arg.startswith('int'):
                if default_arg in defaults:
                    print 'int %s=%s' % (var, defaults[default_arg]),
                else:
                    print 'int %s=-2**31' % var,   # TA_INTEGER_DEFAULT
            elif arg.startswith('TA_MAType'):
                print 'int %s=0' % var,        # TA_MAType_SMA
            else:
                assert False, arg
            if '[, ' not in docs:
                docs[-1] = ('[, ')
            docs.append('%s=?' % var)
            docs.append(', ')

    docs[-1] = '])' if '[, ' in docs else ')'
    if documentation:
        docs.append('\n\n')
        docs.append(documentation)
    print '):'
    print '    """%s"""' % ''.join(docs)
    print '    cdef:'
    print '        np.npy_intp length'
    print '        int begidx, endidx, lookback'
    for arg in args:
        var = arg.split()[-1]

        if 'out' in var:
            break

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print '        double* %s_data' % var
            elif 'int' in arg:
                print '        int* %s_data' % var
            else:
                assert False, args

    for arg in args:
        var = arg.split()[-1]

        if 'out' not in var:
            continue

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            print '        np.ndarray %s' % var
            if 'double' in arg:
                print '        double* %s_data' % var
            elif 'int' in arg:
                print '        int* %s_data' % var
            else:
                assert False, args

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print '        int %s' % var

        else:
            assert False, arg

    for arg in args:
        var = arg.split()[-1]
        if 'out' in var:
            break
        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                cast = '<double*>'
            elif 'int' in arg:
                cast = '<int*>'
            else:
                assert False, arg
            print '    assert PyArray_TYPE(%s) == np.NPY_DOUBLE, "%s is not double"' % (var, var)
            print '    assert %s.ndim == 1, "%s has wrong dimensions"' % (var, var)
            print '    if not (PyArray_FLAGS(%s) & np.NPY_C_CONTIGUOUS):' % var
            print '        %s = PyArray_GETCONTIGUOUS(%s)' % (var, var)
            print '    %s_data = %s%s.data' % (var, cast, var)

    for arg in args:
        var = arg.split()[-1]
        if var in ('inReal0[]', 'inReal1[]', 'inReal[]', 'inHigh[]'):
            var = cleanup(var[:-2])
            print '    length = %s.shape[0]' % var
            print '    begidx = 0'
            print '    for i from 0 <= i < length:'
            print '        if not isnan(%s_data[i]):' % var
            print '            begidx = i'
            print '            break'
            print '    else:'
            print '        raise Exception("inputs are all NaN")'
            print '    endidx = length - begidx - 1'
            break

    print '    lookback = begidx + %s_Lookback(' % name,
    opts = [arg for arg in args if 'opt' in arg]
    for i, opt in enumerate(opts):
        if i > 0:
            print ',',
        print cleanup(opt.split()[-1]),
    print ')'

    for arg in args:
        var = arg.split()[-1]

        if 'out' not in var:
            continue

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print '    %s = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)' % var
                print '    %s_data = <double*>%s.data' % (var, var)
                print '    for i from 0 <= i < min(lookback, length):'
                print '        %s_data[i] = NaN' % var
            elif 'int' in arg:
                print '    %s = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)' % var
                print '    %s_data = <int*>%s.data' % (var, var)
                print '    for i from 0 <= i < min(lookback, length):'
                print '        %s_data[i] = 0' % var
            else:
                assert False, args

    print '    retCode = %s(' % name,

    for i, arg in enumerate(args):
        if i > 0:
            print ',',
        var = arg.split()[-1]

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'out' in var:
                data = '(%s_data+lookback)' % var
            else:
                data = '(%s_data+begidx)' % var
            if 'double' in arg:
                print '<double *>%s' % data,
            elif 'int' in arg:
                print '<int *>%s' % data,
            else:
                assert False, arg

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print '&%s' % var,

        else:
            print cleanup(var) if var != 'startIdx' else '0',

    print ')'
    print '    _ta_check_success("%s", retCode)' % name
    print '    return',
    i = 0
    for arg in args:
        var = arg.split()[-1]
        if var.endswith('[]'):
            var = var[:-2]
        elif var.startswith('*'):
            var = var[1:]
        if var.startswith('out'):
            if var not in ("outNBElement", "outBegIdx"):
                if i > 0:
                    print ',',
                i += 1
                print cleanup(var),
        else:
            assert re.match('.*(void|startIdx|endIdx|opt|in)/*', arg), arg
    print
    print

print '__all__ = [%s]' % ','.join(['\"%s\"' % name for name in names])

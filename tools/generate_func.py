import os
import re
import sys

from talib import abstract

# FIXME: initialize once, then shutdown at the end, rather than each call?
# FIXME: should we pass startIdx and endIdx into function?
# FIXME: don't return number of elements since it always equals allocation?

functions = []
if sys.platform == 'win32':
    include_dirs = [
        r"c:\ta-lib\c\include",
        r"c:\Program Files\TA-Lib\include",
        r"c:\Program Files (x86)\TA-Lib\include",
    ]
else:
    include_dirs = [
        '/usr/include',
        '/usr/local/include',
        '/opt/include',
        '/opt/local/include',
        '/opt/homebrew/include',
        '/opt/homebrew/opt/ta-lib/include',
    ]

if 'TA_INCLUDE_PATH' in os.environ:
    include_dirs = os.environ['TA_INCLUDE_PATH'].split(os.pathsep)

header_found = False
for path in include_dirs:
    ta_func_header = os.path.join(path, 'ta-lib', 'ta_func.h')
    if os.path.exists(ta_func_header):
        header_found = True
        break
if not header_found:
    print('Error: ta-lib/ta_func.h not found', file=sys.stderr)
    sys.exit(1)
with open(ta_func_header) as f:
    tmp = []
    for line in f:
        if line.startswith('TA_LIB_API'):
            line = line[10:]
        line = line.strip()
        if tmp or \
            line.startswith('TA_RetCode TA_') or \
            line.startswith('int TA_'):
            line = re.sub(r'/\*[^\*]+\*/', '', line) # strip comments
            tmp.append(line)
            if not line:
                s = ' '.join(tmp)
                s = re.sub(r'\s+', ' ', s)
                functions.append(s)
                tmp = []

# strip "float" functions
functions = [s for s in functions if not s.startswith('TA_RetCode TA_S_')]

# strip non-indicators
functions = [s for s in functions if not s.startswith('TA_RetCode TA_Set')]
functions = [s for s in functions if not s.startswith('TA_RetCode TA_Restore')]

# print headers
print("""\
cimport numpy as np
from numpy import nan
from cython import boundscheck, wraparound

# _ta_check_success: defined in _common.pxi

cdef double NaN = nan

cdef extern from "numpy/arrayobject.h":
    int PyArray_TYPE(np.ndarray)
    np.ndarray PyArray_EMPTY(int, np.npy_intp*, int, int)
    int PyArray_FLAGS(np.ndarray)
    np.ndarray PyArray_GETCONTIGUOUS(np.ndarray)

np.import_array() # Initialize the NumPy C API

cimport _ta_lib as lib
from _ta_lib cimport TA_RetCode

cdef np.ndarray check_array(np.ndarray real):
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("input array type is not double")
    if real.ndim != 1:
        raise Exception("input array has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_ARRAY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    return real

cdef np.npy_intp check_length2(np.ndarray a1, np.ndarray a2) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_intp check_length3(np.ndarray a1, np.ndarray a2, np.ndarray a3) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_intp check_length4(np.ndarray a1, np.ndarray a2, np.ndarray a3, np.ndarray a4) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    if length != a4.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_int check_begidx1(np.npy_intp length, double* a1):
    cdef:
        double val
    for i from 0 <= i < length:
        val = a1[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx2(np.npy_intp length, double* a1, double* a2):
    cdef:
        double val
    for i from 0 <= i < length:
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx3(np.npy_intp length, double* a1, double* a2, double* a3):
    cdef:
        double val
    for i from 0 <= i < length:
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        val = a3[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx4(np.npy_intp length, double* a1, double* a2, double* a3, double* a4):
    cdef:
        double val
    for i from 0 <= i < length:
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        val = a3[i]
        if val != val:
            continue
        val = a4[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.ndarray make_double_array(np.npy_intp length, int lookback):
    cdef:
        np.ndarray outreal
        double* outreal_data
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_ARRAY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    return outreal

cdef np.ndarray make_int_array(np.npy_intp length, int lookback):
    cdef:
        np.ndarray outinteger
        int* outinteger_data
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_ARRAY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    return outinteger

""")

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
    args = [re.sub(r'[\(\);]', '', s).strip() for s in args]

    shortname = name[3:]
    names.append(shortname)
    try:
        func_info = abstract.Function(shortname).info
        defaults, documentation = abstract._get_defaults_and_docs(func_info)
    except:
        print("cannot find defaults and docs for", shortname, file=sys.stderr)
        defaults, documentation = {}, ""

    print('@wraparound(False)  # turn off relative indexing from end of lists')
    print('@boundscheck(False) # turn off bounds-checking for entire function')
    print('def %s(' % shortname, end=' ')
    docs = [' %s(' % shortname]
    i = 0
    for arg in args:
        var = arg.split()[-1]

        if var in ('startIdx', 'endIdx'):
            continue

        elif 'out' in var:
            break

        if i > 0:
            print(',', end=' ')
        i += 1

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            assert arg.startswith('const double'), arg
            print('np.ndarray %s not None' % var, end=' ')
            docs.append(var)
            docs.append(', ')

        elif var.startswith('opt'):
            var = cleanup(var)
            default_arg = arg.split()[-1][len('optIn'):] # chop off typedef and 'optIn'
            default_arg = default_arg[0].lower() + default_arg[1:] # lowercase first letter

            if arg.startswith('double'):
                if default_arg in defaults:
                    print('double %s=%s' % (var, defaults[default_arg]), end=' ')
                else:
                    print('double %s=-4e37' % var, end=' ') # TA_REAL_DEFAULT
            elif arg.startswith('int'):
                if default_arg in defaults:
                    print('int %s=%s' % (var, defaults[default_arg]), end=' ')
                else:
                    print('int %s=-2**31' % var, end=' ')   # TA_INTEGER_DEFAULT
            elif arg.startswith('TA_MAType'):
                print('int %s=%s' % (var, defaults.get('matype', 0)), end=' ') # TA_MAType_SMA
            else:
                assert False, arg
            if '[, ' not in docs:
                docs[-1] = ('[, ')
            docs.append('%s=?' % var)
            docs.append(', ')

    docs[-1] = '])' if '[, ' in docs else ')'
    if documentation:
        tmp_docs = []
        lower_case = False
        documentation = documentation.split('\n')[2:] # discard abstract calling definition
        for line in documentation:
            line = line.replace('Substraction', 'Subtraction')
            if 'prices' not in line and 'price' in line:
                line = line.replace('price', 'real')
            if not line or line.isspace():
                tmp_docs.append('')
            else:
                tmp_docs.append('    %s' % line) # add an indent of 4 spaces
        docs.append('\n\n')
        docs.append('\n'.join(tmp_docs))
        docs.append('\n    ')
    print('):')
    print('    """%s"""' % ''.join(docs))
    print('    cdef:')
    print('        np.npy_intp length')
    print('        int begidx, endidx, lookback')
    print('        TA_RetCode retCode')

    for arg in args:
        var = arg.split()[-1]
        if 'out' not in var:
            continue
        if var.endswith('[]'):
            var = cleanup(var[:-2])
            print('        np.ndarray %s' % var)
        elif var.startswith('*'):
            var = cleanup(var[1:])
            print('        int %s' % var)
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
            else:
                assert False, arg
            print('    %s = check_array(%s)' % (var, var))

    # check all input array lengths are the same
    inputs = []
    for arg in args:
        var = arg.split()[-1]
        if 'out' in var:
            break
        if var.endswith('[]'):
            var = cleanup(var[:-2])
            inputs.append(var)
    if len(inputs) == 1:
        print('    length = %s.shape[0]' % inputs[0])
    else:
        print('    length = check_length%s(%s)' % (len(inputs), ', '.join(inputs)))

    # check for all input values are non-NaN
    print('    begidx = check_begidx%s(length, %s)' % (len(inputs), ', '.join('<double*>(%s.data)' % s for s in inputs)))

    print('    endidx = <int>length - begidx - 1')
    print('    lookback = begidx + lib.%s_Lookback(' % name, end=' ')
    opts = [arg for arg in args if 'opt' in arg]
    for i, opt in enumerate(opts):
        if i > 0:
            print(',', end=' ')
        print(cleanup(opt.split()[-1]), end=' ')
    print(')')

    for arg in args:
        var = arg.split()[-1]

        if 'out' not in var:
            continue

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print('    %s = make_double_array(length, lookback)' % var)
            elif 'int' in arg:
                print('    %s = make_int_array(length, lookback)' % var)
            else:
                assert False, args

    print('    retCode = lib.%s(' % name, end=' ')

    for i, arg in enumerate(args):
        if i > 0:
            print(',', end=' ')
        var = arg.split()[-1]

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'out' in var:
                data = '(%s.data)+lookback' % var
            else:
                data = '(%s.data)+begidx' % var
            if 'double' in arg:
                print('<double *>%s' % data, end=' ')
            elif 'int' in arg:
                print('<int *>%s' % data, end=' ')
            else:
                assert False, arg

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print('&%s' % var, end=' ')

        else:
            cleaned = cleanup(var) if var != 'startIdx' else '0'
            print(cleaned, end=' ')

    print(')')
    print('    _ta_check_success("%s", retCode)' % name)
    if 'INDEX' in f:
        for arg in args:
            var = arg.split()[-1]

            if 'out' not in var:
                continue

            if var.endswith('[]'):
                var = cleanup(var[:-2])
                print('    %s_data = <int*>%s.data' % (var, var))
                print('    for i from lookback <= i < length:')
                print('        %s_data[i] += begidx' % var)
    print('    return ', end='')
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
                    print(',', end=' ')
                i += 1
                print(cleanup(var), end=' ')
        else:
            assert re.match('.*(void|startIdx|endIdx|opt|in)/*', arg), arg
    print('')
    print('')

print('__TA_FUNCTION_NAMES__ = [%s]' % ','.join(['\"%s\"' % name for name in names]))

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
from cython import boundscheck, wraparound
cimport _ta_lib as lib
from _ta_lib cimport TA_RetCode
# NOTE: _ta_check_success, NaN are defined in common.pxi

np.import_array() # Initialize the NumPy C API
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
    print('def stream_%s(' % shortname, end=' ')
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
    print('        TA_RetCode retCode')
    for arg in args:
        var = arg.split()[-1]
        if 'out' in var:
            break
        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print('        double* %s_data' % var)
            elif 'int' in arg:
                print('        int* %s_data' % var)
            else:
                assert False, args

    for arg in args:
        var = arg.split()[-1]
        if 'out' not in var:
            continue
        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print('        double %s' % var)
            elif 'int' in arg:
                print('        int %s' % var)
            else:
                assert False, args
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
            print('    %s_data = %s%s.data' % (var, cast, var))

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

    for arg in args:
        var = arg.split()[-1]

        if 'out' not in var:
            continue

        if var.endswith('[]'):
            var = cleanup(var[:-2])
            if 'double' in arg:
                print('    %s = NaN' % var)
            elif 'int' in arg:
                print('    %s = 0' % var)
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
                print('&%s' % var, end=' ')
            else:
                print('%s_data' % var, end=' ')

        elif var.startswith('*'):
            var = cleanup(var[1:])
            print('&%s' % var, end=' ')

        elif var in ('startIdx', 'endIdx'):
            print('<int>(length) - 1', end= ' ')

        else:
            cleaned = cleanup(var)
            print(cleaned, end=' ')

    print(')')
    print('    _ta_check_success("%s", retCode)' % name)
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


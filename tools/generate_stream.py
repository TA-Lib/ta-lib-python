import os
import re
import sys

from talib import abstract

VERBS = ('OpenAndFill', 'Open', 'Update', 'Peek', 'Value', 'Clone', 'Close',
         'OutRange', 'Advance')

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
    source = re.sub(r'/\*.*?\*/', '', f.read(), flags=re.S)

# One entry per TA_<NAME>_<verb> declaration; the argument lists never nest
# parentheses, so splitting on commas is enough.
declarations = {}
pattern = r'TA_LIB_API\s+TA_RetCode\s+TA_(\w+)_(%s)\s*\(([^;]*)\)\s*;' % '|'.join(VERBS)
for name, verb, args in re.findall(pattern, source):
    args = [re.sub(r'\s+', ' ', a).strip() for a in args.split(',')]
    declarations.setdefault(name, {})[verb] = args

if not declarations:
    print('Error: %s declares no streaming API; TA-Lib C 0.8.1 or later is required'
          % ta_func_header, file=sys.stderr)
    sys.exit(1)

for name, decls in declarations.items():
    missing = [verb for verb in VERBS if verb not in decls]
    if missing:
        print('Error: TA_%s is missing %s' % (name, ', '.join(missing)), file=sys.stderr)
        sys.exit(1)


def cleanup(name):
    if name.startswith('in'):
        return name[2:].lower()
    elif name.startswith('optIn'):
        return name[5:].lower()
    else:
        return name.lower()


def split_arg(arg):
    """'const double inReal[]' -> ('const double', 'inReal', 0, True)"""
    array = arg.endswith('[]')
    if array:
        arg = arg[:-2]
    ctype, _, var = arg.rpartition(' ')
    stars = len(var) - len(var.lstrip('*'))
    return ctype.strip(), var[stars:], stars, array


def declare(name, decls):
    yield '    ctypedef struct TA_%s_Stream:' % name
    yield '        pass'
    for verb in VERBS:
        args = []
        for arg in decls[verb]:
            ctype, var, stars, array = split_arg(arg)
            args.append('%s%s %s' % (ctype, '*' * (stars + array), var))
        yield '    TA_RetCode TA_%s_%s(%s)' % (name, verb, ', '.join(args))


def parse(name, decls, defaults, output_names):
    open_args = decls['Open']
    history = open_args.index('int historyLen')
    inputs, params, outputs = [], [], []
    for arg in open_args[1:history]:
        ctype, var, stars, array = split_arg(arg)
        assert (ctype, array) == ('const double', True), arg
        inputs.append(cleanup(var))
    for arg in open_args[history + 1:]:
        ctype, var, stars, array = split_arg(arg)
        if var.startswith('out'):
            spelled = (output_names[len(outputs)] if output_names
                       else cleanup(var)[len('out'):])
            outputs.append((ctype, 'out%s' % spelled))
            continue
        key = var[len('optIn'):]
        key = key[0].lower() + key[1:]
        if ctype == 'double':
            default = defaults.get(key, '-4e37')             # TA_REAL_DEFAULT
        elif ctype == 'int':
            default = defaults.get(key, '-2**31')            # TA_INTEGER_DEFAULT
        else:
            assert ctype == 'TA_MAType', arg
            # abstract lowercases the whole name, and a prefixed one (KDJ's
            # slowk_matype) is not spelled 'matype'.
            default = defaults.get(key.lower(), 11)          # TA_MAType_DEFAULT
        params.append(('double' if ctype == 'double' else 'int',
                       cleanup(var), default))
    assert output_names is None or len(outputs) == len(output_names), name
    return {'name': name, 'inputs': inputs, 'params': params, 'outputs': outputs}


PREAMBLE = '''\
cimport cython
cimport numpy as np
import numpy
from collections import namedtuple

cimport _ta_lib as lib
from _ta_lib cimport TA_RetCode, TA_MAType, TA_BAD_PARAM
# NOTE: _ta_check_success and InsufficientHistory come from _common.pxi,
# check_array / make_*_array from _func.pxi, and _PANDAS_SERIES /
# _POLARS_SERIES from _abstract.pxi.

np.import_array() # Initialize the NumPy C API
'''

ARRAYS = '''\
cdef np.npy_intp check_length(tuple arrays) except -1:
    cdef np.npy_intp length = (<np.ndarray>arrays[0]).shape[0]
    for other in arrays[1:]:
        if length != (<np.ndarray>other).shape[0]:
            raise Exception("input array lengths are different")
    return length


cdef int check_begidx(np.npy_intp length, tuple arrays) except -2:
    """The first bar that is not NaN in any input, as the batch tier reads it."""
    cdef double* data[8]
    cdef np.npy_intp i
    cdef int k, count = len(arrays)
    if count > 8:
        raise Exception("too many input arrays")
    for k in range(count):
        data[k] = <double*>(<np.ndarray>arrays[k]).data
    for i in range(length):
        for k in range(count):
            if data[k][i] != data[k][i]:
                break
        else:
            return <int>i
    return <int>length - 1


'''

HELPERS = '''\
OutRange = namedtuple("OutRange", "begidx nbelement", module=__name__)


cdef np.ndarray _stream_input(object values):
    """What the Function API accepts, through the same checks."""
    if isinstance(values, np.ndarray):
        return check_array(values)
    for series in (_PANDAS_SERIES, _POLARS_SERIES):
        if series is not None and isinstance(values, series):
            return check_array(values.to_numpy().astype(float))
    raise TypeError("input must be a numpy array or a pandas or polars Series, "
                    "not %s" % type(values).__name__)


cdef int _stream_history(np.npy_intp length, int begidx) except -1:
    if begidx < 0:
        raise InsufficientHistory("no history: the input array is empty")
    return <int>(length - begidx)


cdef _stream_open_failed(str function_name, TA_RetCode retCode, int historylen, int need):
    if retCode == 17:
        if need <= historylen:
            need = historylen + 1   # a seeding that wants more than the lookback
        raise InsufficientHistory(
            "%s: %d bars of history, at least %d needed (a leading bar that is "
            "NaN in any input is not history)" % (function_name, historylen, need))
    _ta_check_success(function_name, retCode)


cdef _stream_like(tuple sources, object result):
    pandas = [s for s in sources
              if _PANDAS_SERIES is not None and isinstance(s, _PANDAS_SERIES)]
    polars = [s for s in sources
              if _POLARS_SERIES is not None and isinstance(s, _POLARS_SERIES)]
    if pandas and polars:
        raise Exception("Cannot mix polars and pandas")
    if pandas:
        return _PANDAS_SERIES(result, index=pandas[0].index)
    if polars:
        return _POLARS_SERIES(result)
    return result


cdef class Stream:
    """Base class of every talib.stream handle.

    Carries what does not depend on the function: the opaque C handle, and the
    offset of the first history bar the batch tier would have used."""
    cdef void* _handle
    cdef int _begidx
    cdef object __weakref__

    def __cinit__(self):
        self._handle = NULL

    cdef TA_RetCode _out_range(self, int* outbegidx, int* outnbelement):
        return TA_BAD_PARAM

    cdef TA_RetCode _advance(self):
        return TA_BAD_PARAM

    @property
    def out_range(self):
        cdef int outbegidx
        cdef int outnbelement
        cdef TA_RetCode retCode = self._out_range(&outbegidx, &outnbelement)
        if retCode != 0:
            _ta_check_success("%s.out_range" % type(self).__name__, retCode)
        return OutRange(outbegidx + self._begidx, outnbelement)

    @cython.binding(False)
    def advance(self):
        cdef TA_RetCode retCode = self._advance()
        if retCode != 0:
            _ta_check_success("%s.advance" % type(self).__name__, retCode)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __reduce__(self):
        raise TypeError(
            "cannot pickle %s: a stream handle points into the TA-Lib C library, "
            "and does not cross a process boundary" % type(self).__name__)
'''


def emit(func, docstring):
    name, inputs = func['name'], func['inputs']
    params, outputs = func['params'], func['outputs']
    handle, cls = 'TA_%s_Stream' % name, '%s_Stream' % name
    arguments = inputs + ['%s %s=%s' % p for p in params]
    out_ptrs = ['&%s' % py for _, py in outputs]
    # An index output counts from the first bar the stream opened on; the batch
    # tier reports it in the caller's coordinates, so shift it back the same way.
    shift = ' + self._begidx' if 'INDEX' in name else ''
    value = (outputs[0][1] + shift if len(outputs) == 1
             else '(%s)' % ', '.join(py + shift for _, py in outputs))
    live = '<%s*>self._handle' % handle
    lookback_args = ', '.join(py for _, py, _ in params)
    out = []

    def call(verb, *args):
        return 'TA_%s_%s(%s)' % (name, verb, ', '.join(args))

    def emit_call(verb, *args):
        out.append('        cdef TA_RetCode retCode = %s' % call(verb, *args))
        out.append('        if retCode != 0:')
        out.append('            _ta_check_success("TA_%s_%s", retCode)' % (name, verb))

    def emit_history():
        """Read the inputs as the batch tier does: leading bars that are NaN in
        any input are not history, and the stream opens past them."""
        for py in inputs:
            out.append('        cdef np.ndarray a_%s = _stream_input(%s)' % (py, py))
        out.append('        cdef tuple arrays = (%s,)'
                   % ', '.join('a_%s' % py for py in inputs))
        out.append('        cdef np.npy_intp length = check_length(arrays)')
        out.append('        cdef int begidx = check_begidx(length, arrays)')
        out.append('        cdef int historylen = _stream_history(length, begidx)')

    def opened(*tail):
        return (['&handle'] + ['<double*>a_%s.data + begidx' % py for py in inputs]
                + ['historylen'] + [py for _, py, _ in params] + list(tail))

    out.append('cdef class %s(Stream):' % cls)
    out.append('    """%s"""' % docstring)
    out.append('')
    out.append('    def __dealloc__(self):')
    out.append('        if self._handle is not NULL:')
    out.append('            %s' % call('Close', live))
    out.append('            self._handle = NULL')
    out.append('')
    out.append('    def __init__(self, %s):' % ', '.join(arguments))
    emit_history()
    out.append('        cdef %s* handle = NULL' % handle)
    out.extend('        cdef %s %s' % o for o in outputs)
    out.append('        cdef TA_RetCode retCode = %s' % call('Open', *opened(*out_ptrs)))
    out.append('        if retCode != 0:')
    out.append('            _stream_open_failed("TA_%s_Open", retCode, historylen, '
               'lib.TA_%s_Lookback(%s) + 1)' % (name, name, lookback_args))
    out.append('        if self._handle is not NULL:')
    out.append('            %s' % call('Close', live))
    out.append('        self._handle = <void*>handle')
    out.append('        self._begidx = begidx')
    out.append('')
    out.append('    @staticmethod')
    out.append('    def open_and_fill(%s):' % ', '.join(arguments))
    emit_history()
    out.append('        cdef int lookback = begidx + lib.TA_%s_Lookback(%s)'
               % (name, lookback_args))
    out.extend('        cdef np.ndarray %s = make_%s_array(length, lookback)' % (py, ctype)
               for ctype, py in outputs)
    out.append('        cdef int outbegidx')
    out.append('        cdef int outnbelement')
    out.append('        cdef %s* handle = NULL' % handle)
    out.append('        cdef TA_RetCode retCode = %s' % call('OpenAndFill', *opened(
        '&outbegidx', '&outnbelement',
        *['<%s*>%s.data + lookback' % o for o in outputs])))
    out.append('        if retCode != 0:')
    out.append('            _stream_open_failed("TA_%s_OpenAndFill", retCode, historylen, '
               'lookback - begidx + 1)' % name)
    if shift:
        out.append('        cdef np.npy_intp i')
        for ctype, py in outputs:
            out.append('        cdef %s* %s_data = <%s*>%s.data' % (ctype, py, ctype, py))
            out.append('        for i in range(lookback, length):')
            out.append('            %s_data[i] += begidx' % py)
    out.append('        cdef %s stream = %s.__new__(%s)' % (cls, cls, cls))
    out.append('        stream._handle = <void*>handle')
    out.append('        stream._begidx = begidx')
    filled = ['_stream_like((%s,), %s)' % (', '.join(inputs), py) for _, py in outputs]
    out.append('        return stream, %s' % (
        filled[0] if len(outputs) == 1 else '(%s)' % ', '.join(filled)))
    for verb in ('Update', 'Peek'):
        out.append('')
        out.append('    @cython.binding(False)')
        out.append('    def %s(self, %s):'
                   % (verb.lower(), ', '.join('double %s' % py for py in inputs)))
        out.extend('        cdef %s %s' % o for o in outputs)
        emit_call(verb, *([live] + inputs + out_ptrs))
        out.append('        return %s' % value)
    out.append('')
    out.append('    @property')
    out.append('    def value(self):')
    out.extend('        cdef %s %s' % o for o in outputs)
    emit_call('Value', *([live] + out_ptrs))
    out.append('        return %s' % value)
    out.append('')
    out.append('    cdef TA_RetCode _out_range(self, int* outbegidx, int* outnbelement):')
    out.append('        return %s' % call('OutRange', live, 'outbegidx', 'outnbelement'))
    out.append('')
    out.append('    cdef TA_RetCode _advance(self):')
    out.append('        return %s' % call('Advance', live))
    out.append('')
    out.append('    @cython.binding(False)')
    out.append('    def copy(self):')
    out.append('        cdef %s* clone = NULL' % handle)
    out.append('        cdef TA_RetCode retCode = %s' % call('Clone', live, '&clone'))
    out.append('        if retCode != 0:')
    out.append('            _ta_check_success("TA_%s_Clone", retCode)' % name)
    out.append('        cdef %s stream = %s.__new__(type(self))' % (cls, cls))
    out.append('        stream._handle = <void*>clone')
    out.append('        stream._begidx = self._begidx')
    out.append('        return stream')
    return '\n'.join(out)


STUB = '''\
from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray


class OutRange(NamedTuple):
    begidx: int
    nbelement: int


class Stream:
    @property
    def out_range(self) -> OutRange: ...
    def advance(self) -> None: ...
'''


def emit_stub(func, documented):
    """The talib/stream.pyi entry, defaults spelled as the documentation does."""
    name, inputs, params, outputs = (func['name'], func['inputs'],
                                     func['params'], func['outputs'])
    args = ', '.join(['%s: NDArray[np.float64]' % py for py in inputs]
                     + ['%s: %s = %r' % (py, 'float' if ctype == 'double' else 'int',
                                         documented[py]) for ctype, py, _ in params])
    scalars = ['float' if ctype == 'double' else 'int' for ctype, _ in outputs]
    arrays = ['NDArray[np.float64]' if ctype == 'double' else 'NDArray[np.int32]'
              for ctype, _ in outputs]
    out = []
    if len(outputs) > 1:
        value = 'Tuple[%s]' % ', '.join(scalars)
        filled = 'Tuple[%s]' % ', '.join(arrays)
    else:
        value, filled = scalars[0], arrays[0]
    bars = ', '.join('%s: float' % py for py in inputs)
    out.append('class %s(Stream):' % name)
    out.append('    def __init__(self, %s) -> None: ...' % args)
    out.append('    @staticmethod')
    out.append('    def open_and_fill(%s) -> Tuple["%s", %s]: ...' % (args, name, filled))
    out.append('    def update(self, %s) -> %s: ...' % (bars, value))
    out.append('    def peek(self, %s) -> %s: ...' % (bars, value))
    out.append('    @property')
    out.append('    def value(self) -> %s: ...' % value)
    out.append('    def copy(self) -> "%s": ...' % name)
    return '\n'.join(out)


def docstring_for(func, documentation):
    """The batch function's docstring, verbatim: same call, same defaults."""
    docs = [' %s(' % func['name']]
    for py in func['inputs']:
        docs.append(py)
        docs.append(', ')
    for _, py, _ in func['params']:
        if '[, ' not in docs:
            docs[-1] = '[, '
        docs.append('%s=?' % py)
        docs.append(', ')
    docs[-1] = '])' if '[, ' in docs else ')'
    if documentation:
        lines = []
        for line in documentation.split('\n')[2:]:  # discard the calling definition
            line = line.replace('Substraction', 'Subtraction')
            if 'prices' not in line and 'price' in line:
                line = line.replace('price', 'real')
            lines.append('' if not line or line.isspace() else '    %s' % line)
        docs.append('\n\n')
        docs.append('\n'.join(lines))
        docs.append('\n    ')
    return ''.join(docs)


stub = '--stub' in sys.argv

if not stub:
    print(PREAMBLE)
    print('cdef extern from "ta-lib/ta_func.h":')
    for name in sorted(declarations):
        print()
        for line in declare(name, declarations[name]):
            print(line)
    print()
    print(ARRAYS)
    print(HELPERS)
else:
    print(STUB)

for name in sorted(declarations):
    try:
        info = abstract.Function(name).info
        defaults, documentation = abstract._get_defaults_and_docs(info)
    except Exception:
        print("cannot find defaults and docs for", name, file=sys.stderr)
        info = {'output_names': None, 'parameters': {}}
        defaults, documentation = {}, ""
    func = parse(name, declarations[name], defaults, info['output_names'])
    print()
    if stub:
        print(emit_stub(func, info['parameters']))
    else:
        print(emit(func, docstring_for(func, documentation)))
        print()

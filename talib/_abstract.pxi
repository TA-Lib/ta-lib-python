'''
This file Copyright (c) 2013 Brian A Cappello <briancappello at gmail>
'''
import math
import threading
try:
    from collections import OrderedDict
except ImportError: # handle python 2.6 and earlier
    from ordereddict import OrderedDict
from cython.operator cimport dereference as deref
import numpy
import sys

cimport numpy as np
cimport _ta_lib as lib
# NOTE: _ta_check_success, MA_Type is defined in _common.pxi

np.import_array() # Initialize the NumPy C API

# lookup for TALIB input parameters which don't define expected price series inputs
__INPUT_PRICE_SERIES_DEFAULTS = {'price':   'close',
                                 'price0':  'high',
                                 'price1':  'low',
                                 'periods': 'periods', # only used by MAVP; not a price series!
                                 }

__INPUT_ARRAYS_TYPES = [dict]
__ARRAY_TYPES = [np.ndarray]

# allow use of pandas.DataFrame for input arrays
try:
    import pandas
    __INPUT_ARRAYS_TYPES.append(pandas.DataFrame)
    __ARRAY_TYPES.append(pandas.Series)
    __PANDAS_DATAFRAME = pandas.DataFrame
    __PANDAS_SERIES = pandas.Series
except ImportError as import_error:
    try:
        if not isinstance(import_error, ModuleNotFoundError) or import_error.name != 'pandas':
            # Propagate the error when the module exists but failed to be imported.
            raise import_error
    # `ModuleNotFoundError` was introduced in Python 3.6.
    except NameError:
        pass

    __PANDAS_DATAFRAME = None
    __PANDAS_SERIES = None

# allow use of polars.DataFrame for input arrays
try:
    import polars
    __INPUT_ARRAYS_TYPES.append(polars.DataFrame)
    __ARRAY_TYPES.append(polars.Series)
    __POLARS_DATAFRAME = polars.DataFrame
    __POLARS_SERIES = polars.Series
except ImportError as import_error:
    try:
        if not isinstance(import_error, ModuleNotFoundError) or import_error.name != 'polars':
            # Propagate the error when the module exists but failed to be imported.
            raise import_error
    # `ModuleNotFoundError` was introduced in Python 3.6.
    except NameError:
        pass

    __POLARS_DATAFRAME = None
    __POLARS_SERIES = None

__INPUT_ARRAYS_TYPES = tuple(__INPUT_ARRAYS_TYPES)
__ARRAY_TYPES = tuple(__ARRAY_TYPES)


if sys.version >= '3':

    def str2bytes(s):
        return bytes(s, 'ascii')

    def bytes2str(b):
        return b.decode('ascii')

else:

    def str2bytes(s):
        return s

    def bytes2str(b):
        return b

class Function(object):
    """
    This is a pythonic wrapper around TALIB's abstract interface. It is
    intended to simplify using individual TALIB functions by providing a
    unified interface for setting/controlling input data, setting function
    parameters and retrieving results. Input data consists of a ``dict`` of
    ``numpy`` arrays (or a ``pandas.DataFrame`` or ``polars.DataFrame``), one
    array for each of open, high, low, close and volume. This can be set with
    the set_input_arrays() method. Which keyed array(s) are used as inputs when
    calling the function is controlled using the input_names property.

    This class gets initialized with a TALIB function name and optionally an
    input_arrays object. It provides the following primary functions for
    setting inputs and retrieving results:

    ---- input_array/TA-function-parameter set-only functions -----
    - set_input_arrays(input_arrays)
    - set_function_args([input_arrays,] [param_args_andor_kwargs])

    Documentation for param_args_andor_kwargs can be seen by printing the
    Function instance or programmatically via the info, input_names and
    parameters properties.

    ----- result-returning functions -----
    - the outputs property wraps a method which ensures results are always valid
    - run([input_arrays]) # calls set_input_arrays and returns self.outputs
    - FunctionInstance([input_arrays,] [param_args_andor_kwargs]) # calls set_function_args and returns self.outputs
    """

    def __init__(self, function_name, func_object, *args, **kwargs):
        # make sure the function_name is valid and define all of our variables
        self.__name = function_name.upper()
        self.__namestr = self.__name
        self.__name = str2bytes(self.__name)

        # thread-local storage
        self.__localdata = threading.local()

        # finish initializing: query the TALIB abstract interface and set arguments
        self.set_function_args(*args, **kwargs)
        self.func_object = func_object

    @property
    def __local(self):
        local = self.__localdata
        if not hasattr(local, 'info'):
            local.info = None
            local.input_arrays = {}

            # dictionaries of function args. keys are input/opt_input/output parameter names
            local.input_names = OrderedDict()
            local.opt_inputs = OrderedDict()
            local.outputs = OrderedDict()
            local.outputs_valid = False

            # function info
            local.info = _ta_getFuncInfo(self.__name)

            # inputs (price series names)
            for i in xrange(local.info.pop('num_inputs')):
                info = _ta_getInputParameterInfo(self.__name, i)
                input_name = info['name']
                if info['price_series'] is None:
                    info['price_series'] = __INPUT_PRICE_SERIES_DEFAULTS[input_name]
                local.input_names[input_name] = info
            local.info['input_names'] = self.input_names

            # optional inputs (function parameters)
            for i in xrange(local.info.pop('num_opt_inputs')):
                info = _ta_getOptInputParameterInfo(self.__name, i)
                param_name = info['name']
                local.opt_inputs[param_name] = info
            local.info['parameters'] = self.parameters

            # outputs
            local.info['output_flags'] = OrderedDict()
            for i in xrange(local.info.pop('num_outputs')):
                info = _ta_getOutputParameterInfo(self.__name, i)
                output_name = info['name']
                local.info['output_flags'][output_name] = info['flags']
                local.outputs[output_name] = None
            local.info['output_names'] = self.output_names
        return local

    @property
    def info(self):
        """
        Returns a copy of the function's info dict.
        """
        return self.__local.info.copy()

    @property
    def function_flags(self):
        """
        Returns any function flags defined for this indicator function.
        """
        return self.__local.info['function_flags']

    @property
    def output_flags(self):
        """
        Returns the flags for each output for this indicator function.
        """
        return self.__local.info['output_flags'].copy()

    def get_input_names(self):
        """
        Returns the dict of input price series names that specifies which
        of the ndarrays in input_arrays will be used to calculate the function.
        """
        local = self.__local
        ret = OrderedDict()
        for input_name in local.input_names:
            ret[input_name] = local.input_names[input_name]['price_series']
        return ret

    def set_input_names(self, input_names):
        """
        Sets the input price series names to use.
        """
        local = self.__local
        for input_name, price_series in input_names.items():
            local.input_names[input_name]['price_series'] = price_series
            local.info['input_names'][input_name] = price_series
        local.outputs_valid = False

    input_names = property(get_input_names, set_input_names)

    def get_input_arrays(self):
        """
        Returns a copy of the dict of input arrays in use.
        """
        local = self.__local
        if __POLARS_DATAFRAME is not None \
            and isinstance(local.input_arrays, __POLARS_DATAFRAME):
            return local.input_arrays.clone()
        else:
            return local.input_arrays.copy()

    def set_input_arrays(self, input_arrays):
        """
        Sets the dict of input_arrays to use. Returns True/False for
        subclasses:

        If input_arrays is a dict with the keys open, high, low, close and
        volume, it is assigned as the input_array to use and this function
        returns True, returning False otherwise. If you implement your own
        data type and wish to subclass Function, you should wrap this function
        with an if-statement:

        class CustomFunction(Function):
            def __init__(self, function_name):
                Function.__init__(self, function_name)

            def set_input_arrays(self, input_data):
                if Function.set_input_arrays(self, input_data):
                    return True
                elif isinstance(input_data, some_module.CustomDataType):
                    input_arrays = Function.get_input_arrays(self)
                    # convert input_data to input_arrays and then call the super
                    Function.set_input_arrays(self, input_arrays)
                    return True
                return False
        """
        local = self.__local
        if isinstance(input_arrays, __INPUT_ARRAYS_TYPES):
            missing_keys = []
            for key in self.__input_price_series_names():
                if __POLARS_DATAFRAME is not None \
                    and isinstance(input_arrays, __POLARS_DATAFRAME):
                    missing = key not in input_arrays.columns
                else:
                    missing = key not in input_arrays
                if missing:
                    missing_keys.append(key)
            if len(missing_keys) == 0:
                local.input_arrays = input_arrays
                local.outputs_valid = False
                return True
            else:
                raise Exception('input_arrays parameter missing required data '\
                                'key%s: %s' % ('s' if len(missing_keys) > 1 \
                                                    else '',
                                                ', '.join(missing_keys)))
        return False

    input_arrays = property(get_input_arrays, set_input_arrays)

    def get_parameters(self):
        """
        Returns the function's optional parameters and their default values.
        """
        local = self.__local
        ret = OrderedDict()
        for opt_input in local.opt_inputs:
            ret[opt_input] = self.__get_opt_input_value(opt_input)
        return ret

    def set_parameters(self, parameters=None, **kwargs):
        """
        Sets the function parameter values.
        """
        local = self.__local
        parameters = parameters or {}
        parameters.update(kwargs)
        for param, value in parameters.items():
            if self.__check_opt_input_value(param, value):
                local.opt_inputs[param]['value'] = value
        local.outputs_valid = False
        local.info['parameters'] = self.parameters

    parameters = property(get_parameters, set_parameters)

    def set_function_args(self, *args, **kwargs):
        """
        optional args:[input_arrays,] [parameter_args,] [input_price_series_kwargs,] [parameter_kwargs]
        """
        local = self.__local
        update_info = False

        for key in kwargs:
            if key in local.opt_inputs:
                value = kwargs[key]
                if self.__check_opt_input_value(key, value):
                    local.opt_inputs[key]['value'] = kwargs[key]
                    update_info = True
            elif key in local.input_names:
                local.input_names[key]['price_series'] = kwargs[key]
                local.info['input_names'][key] = kwargs[key]

        if args:
            skip_first = 0
            if self.set_input_arrays(args[0]):
                skip_first = 1
            if len(args) > skip_first:
                for i, param_name in enumerate(local.opt_inputs):
                    i += skip_first
                    if i < len(args):
                        value = args[i]
                        if self.__check_opt_input_value(param_name, value):
                            local.opt_inputs[param_name]['value'] = value
                            update_info = True

        if args or kwargs:
            if update_info:
                local.info['parameters'] = self.parameters
            local.outputs_valid = False

    @property
    def lookback(self):
        """
        Returns the lookback window size for the function with the parameter
        values that are currently set.
        """
        local = self.__local
        cdef lib.TA_ParamHolder *holder
        holder = __ta_paramHolderAlloc(self.__name)
        for i, opt_input in enumerate(local.opt_inputs):
            value = self.__get_opt_input_value(opt_input)
            type_ = local.opt_inputs[opt_input]['type']
            if type_ == lib.TA_OptInput_RealRange or type_ == lib.TA_OptInput_RealList:
                __ta_setOptInputParamReal(holder, i, value)
            elif type_ == lib.TA_OptInput_IntegerRange or type_ == lib.TA_OptInput_IntegerList:
                __ta_setOptInputParamInteger(holder, i, value)

        lookback = __ta_getLookback(holder)
        __ta_paramHolderFree(holder)
        return lookback

    @property
    def output_names(self):
        """
        Returns a list of the output names returned by this function.
        """
        ret = self.__local.outputs.keys()
        if not isinstance(ret, list):
            ret = list(ret)
        return ret

    @property
    def outputs(self):
        """
        Returns the TA function values for the currently set input_arrays and
        parameters. Returned values are a ndarray if there is only one output
        or a list of ndarrays for more than one output.
        """
        local = self.__local
        if not local.outputs_valid:
            self.__call_function()
        ret = local.outputs.values()
        if not isinstance(ret, list):
            ret = list(ret)
        if __PANDAS_DATAFRAME is not None and \
                isinstance(local.input_arrays, __PANDAS_DATAFRAME):
            index = local.input_arrays.index
            if len(ret) == 1:
                return __PANDAS_SERIES(ret[0], index=index)
            else:
                return __PANDAS_DATAFRAME(numpy.column_stack(ret),
                                          index=index,
                                          columns=self.output_names)
        elif __POLARS_DATAFRAME is not None and \
                isinstance(local.input_arrays, __POLARS_DATAFRAME):
            if len(ret) == 1:
                return __POLARS_SERIES(ret[0])
            else:
                return __POLARS_DATAFRAME(numpy.column_stack(ret),
                                          schema=self.output_names)
        else:
            return ret[0] if len(ret) == 1 else ret

    def run(self, input_arrays=None):
        """
        run([input_arrays=None])

        This is a shortcut to the outputs property that also allows setting
        the input_arrays dict.
        """
        if input_arrays:
            self.set_input_arrays(input_arrays)
        self.__call_function()
        return self.outputs

    def __call__(self, *args, **kwargs):
        """
        func_instance([input_arrays,] [parameter_args,] [input_price_series_kwargs,] [parameter_kwargs])

        This is a shortcut to the outputs property that also allows setting
        the input_arrays dict and function parameters.
        """
        local = self.__local
        # do not cache ta-func parameters passed to __call__
        opt_input_values = [(param_name, local.opt_inputs[param_name]['value'])
                            for param_name in local.opt_inputs.keys()]
        price_series_name_values = [(n, local.input_names[n]['price_series'])
                                    for n in local.input_names]

        # allow calling with same signature as talib.func module functions
        args = list(args)
        input_arrays = {}
        input_price_series_names = self.__input_price_series_names()
        if args and not isinstance(args[0], __INPUT_ARRAYS_TYPES):
            for i, arg in enumerate(args):
                if not isinstance(arg, __ARRAY_TYPES):
                    break

                try:
                    input_arrays[input_price_series_names[i]] = arg
                except IndexError:
                    msg = 'Too many price arguments: expected %d (%s)' % (
                        len(input_price_series_names),
                        ', '.join(input_price_series_names))
                    raise TypeError(msg)

        if __PANDAS_DATAFRAME is not None \
                and isinstance(local.input_arrays, __PANDAS_DATAFRAME):
            no_existing_input_arrays = local.input_arrays.empty
        elif __POLARS_DATAFRAME is not None \
                and isinstance(local.input_arrays, __POLARS_DATAFRAME):
            no_existing_input_arrays = local.input_arrays.is_empty()
        else:
            no_existing_input_arrays = not bool(local.input_arrays)

        if len(input_arrays) == len(input_price_series_names):
            self.set_input_arrays(input_arrays)
            args = args[len(input_arrays):]
        elif len(input_arrays) or (no_existing_input_arrays and (
                not len(args) or not isinstance(args[0], __INPUT_ARRAYS_TYPES))):
            msg = 'Not enough price arguments: expected %d (%s)' % (
                len(input_price_series_names),
                ', '.join(input_price_series_names))
            raise TypeError(msg)

        self.set_function_args(*args, **kwargs)
        self.__call_function()

        # restore opt_input values to as they were before this call
        for param_name, value in opt_input_values:
            local.opt_inputs[param_name]['value'] = value
        local.info['parameters'] = self.parameters

        # restore input names values to as they were before this call
        for input_name, value in price_series_name_values:
            local.input_names[input_name]['price_series'] = value
            local.info['input_names'][input_name] = value

        return self.outputs

    # figure out which price series names we're using for inputs
    def __input_price_series_names(self):
        local = self.__local
        input_price_series_names = []
        for input_name in local.input_names:
            price_series = local.input_names[input_name]['price_series']
            if isinstance(price_series, list): # TALIB-supplied input names
                for name in price_series:
                    input_price_series_names.append(name)
            else: # name came from __INPUT_PRICE_SERIES_DEFAULTS
                input_price_series_names.append(price_series)
        return input_price_series_names

    def __call_function(self):
        local = self.__local
        input_price_series_names = self.__input_price_series_names()

        # populate the ordered args we'll call the function with
        args = []
        for price_series in input_price_series_names:
            series = local.input_arrays[price_series]
            if __PANDAS_SERIES is not None and \
                    isinstance(series, __PANDAS_SERIES):
                series = series.values.astype(float)
            elif __POLARS_SERIES is not None and \
                    isinstance(series, __POLARS_SERIES):
                series = series.to_numpy().astype(float)
            args.append(series)
        for opt_input in local.opt_inputs:
            value = self.__get_opt_input_value(opt_input)
            args.append(value)

        # Use the func module to actually call the function.
        results = self.func_object(*args)
        if isinstance(results, np.ndarray):
            keys = local.outputs.keys()
            if not isinstance(keys, list):
                keys = list(keys)
            local.outputs[keys[0]] = results
        else:
            for i, output in enumerate(local.outputs):
                local.outputs[output] = results[i]
        local.outputs_valid = True

    def __check_opt_input_value(self, input_name, value):
        type_ = self.__local.opt_inputs[input_name]['type']
        if type_ in {lib.TA_OptInput_IntegerList, lib.TA_OptInput_IntegerRange}:
            type_ = int
        elif type_ in {lib.TA_OptInput_RealList, lib.TA_OptInput_RealRange}:
            type_ = float

        if isinstance(value, type_):
           return True
        elif value is not None:
            raise TypeError(
                'Invalid parameter value for %s (expected %s, got %s)' % (
                    input_name, type_.__name__, type(value).__name__))
        return False

    def __get_opt_input_value(self, input_name):
        """
        Returns the user-set value if there is one, otherwise the default.
        """
        local = self.__local
        value = local.opt_inputs[input_name]['value']
        if value is None:
            value = local.opt_inputs[input_name]['default_value']
        return value

    def __repr__(self):
        return '%s' % self.info

    def __unicode__(self):
        return unicode(self.__str__())

    def __str__(self):
        return _get_defaults_and_docs(self.info)[1] # docstring includes defaults


######################  INTERNAL python-level functions  #######################
# These map 1-1 with native C TALIB abstract interface calls. Their names
# are the same except for having the leading 4 characters lowercased (and
# the Alloc/Free function pairs which have been combined into single get
# functions)
#
# These are TA function information-discovery calls. The Function class
# encapsulates these functions into an easy-to-use, pythonic interface. It's
# therefore recommended over using these functions directly.

def _ta_getGroupTable():
    """
    Returns the list of available TALIB function group names. *slow*
    """
    cdef lib.TA_StringTable *table
    _ta_check_success('TA_GroupTableAlloc', lib.TA_GroupTableAlloc(&table))
    groups = []
    for i in xrange(table.size):
        groups.append(deref(&table.string[i]))
    _ta_check_success('TA_GroupTableFree', lib.TA_GroupTableFree(table))
    return groups

def _ta_getFuncTable(char *group):
    """
    Returns a list of the functions for the specified group name. *slow*
    """
    cdef lib.TA_StringTable *table
    _ta_check_success('TA_FuncTableAlloc', lib.TA_FuncTableAlloc(group, &table))
    functions = []
    for i in xrange(table.size):
        functions.append(deref(&table.string[i]))
    _ta_check_success('TA_FuncTableFree', lib.TA_FuncTableFree(table))
    return functions

def __get_flags(int flag, dict flags_lookup_dict):
    """
    TA-LIB provides hints for multiple flags as a bitwise-ORed int.
    This function returns the flags from flag found in the provided
    flags_lookup_dict.
    """
    value_range = flags_lookup_dict.keys()
    if not isinstance(value_range, list):
        value_range = list(value_range)
    min_int = int(math.log(min(value_range), 2))
    max_int = int(math.log(max(value_range), 2))

    # if the flag we got is out-of-range, it just means no extra info provided
    if flag < 1 or flag > 2**max_int:
        return None

    # In this loop, i is essentially the bit-position, which represents an
    # input from flags_lookup_dict. We loop through as many flags_lookup_dict
    # bit-positions as we need to check, bitwise-ANDing each with flag for a hit.
    ret = []
    for i in xrange(min_int, max_int+1):
        if 2**i & flag:
            ret.append(flags_lookup_dict[2**i])
    return ret

TA_FUNC_FLAGS = {
    16777216: 'Output scale same as input',
    67108864: 'Output is over volume',
    134217728: 'Function has an unstable period',
    268435456: 'Output is a candlestick'
}

# when flag is 0, the function (should) work on any reasonable input ndarray
TA_INPUT_FLAGS = {
    1: 'open',
    2: 'high',
    4: 'low',
    8: 'close',
    16: 'volume',
    32: 'openInterest',
    64: 'timeStamp'
}

TA_OUTPUT_FLAGS = {
    1: 'Line',
    2: 'Dotted Line',
    4: 'Dashed Line',
    8: 'Dot',
    16: 'Histogram',
    32: 'Pattern (Bool)',
    64: 'Bull/Bear Pattern (Bearish < 0, Neutral = 0, Bullish > 0)',
    128: 'Strength Pattern ([-200..-100] = Bearish, [-100..0] = Getting Bearish, 0 = Neutral, [0..100] = Getting Bullish, [100-200] = Bullish)',
    256: 'Output can be positive',
    512: 'Output can be negative',
    1024: 'Output can be zero',
    2048: 'Values represent an upper limit',
    4096: 'Values represent a lower limit'
}

def _ta_getFuncInfo(char *function_name):
    """
    Returns the info dict for the function. It has the following keys: name,
    group, help, flags, num_inputs, num_opt_inputs and num_outputs.
    """
    cdef const lib.TA_FuncInfo *info
    retCode = lib.TA_GetFuncInfo(__ta_getFuncHandle(function_name), &info)
    _ta_check_success('TA_GetFuncInfo', retCode)

    return {
        'name': bytes2str(info.name),
        'group': bytes2str(info.group),
        'display_name': bytes2str(info.hint),
        'function_flags': __get_flags(info.flags, TA_FUNC_FLAGS),
        'num_inputs': int(info.nbInput),
        'num_opt_inputs': int(info.nbOptInput),
        'num_outputs': int(info.nbOutput)
    }

def _ta_getInputParameterInfo(char *function_name, int idx):
    """
    Returns the function's input info dict for the given index. It has two
    keys: name and flags.
    """
    cdef const lib.TA_InputParameterInfo *info
    retCode = lib.TA_GetInputParameterInfo(__ta_getFuncHandle(function_name), idx, &info)
    _ta_check_success('TA_GetInputParameterInfo', retCode)

    name = bytes2str(info.paramName)
    name = name[len('in'):].lower()
    if 'real' in name:
        name = name.replace('real', 'price')
    elif 'price' in name:
        name = 'prices'

    return {
        'name': name,
        'price_series': __get_flags(info.flags, TA_INPUT_FLAGS)
    }

def _ta_getOptInputParameterInfo(char *function_name, int idx):
    """
    Returns the function's opt_input info dict for the given index. It has the
    following keys: name, display_name, type, help, default_value and value.
    """
    cdef const lib.TA_OptInputParameterInfo *info
    retCode = lib.TA_GetOptInputParameterInfo(__ta_getFuncHandle(function_name), idx, &info)
    _ta_check_success('TA_GetOptInputParameterInfo', retCode)

    name = bytes2str(info.paramName)
    name = name[len('optIn'):].lower()
    default_value = int(info.defaultValue) if info.type > 1 else info.defaultValue

    return {
        'name': name,
        'display_name': bytes2str(info.displayName),
        'type': info.type,
        'help': bytes2str(info.hint),
        'default_value': default_value,
        'value': None
    }

def _ta_getOutputParameterInfo(char *function_name, int idx):
    """
    Returns the function's output info dict for the given index. It has two
    keys: name and flags.
    """
    cdef const lib.TA_OutputParameterInfo *info
    retCode = lib.TA_GetOutputParameterInfo(__ta_getFuncHandle(function_name), idx, &info)
    _ta_check_success('TA_GetOutputParameterInfo', retCode)

    name = bytes2str(info.paramName)
    name = name[len('out'):].lower()
    # chop off leading 'real' if a descriptive name follows
    if 'real' in name and name not in ['real', 'real0', 'real1']:
        name = name[len('real'):]

    return {
        'name': name,
        'flags': __get_flags(info.flags, TA_OUTPUT_FLAGS)
    }

def _get_defaults_and_docs(func_info):
    """
    Returns a tuple with two outputs: defaults, a dict of parameter defaults,
    and documentation, a formatted docstring for the function.
    .. Note: func_info should come from Function.info, *not* _ta_getFuncInfo.
    """
    defaults = {}
    func_line = [func_info['name'], '(']
    func_args = ['[input_arrays]']
    docs = []
    docs.append('%(display_name)s (%(group)s)\n' % func_info)

    input_names = func_info['input_names']
    docs.append('Inputs:')
    for input_name in input_names:
        value = input_names[input_name]
        if not isinstance(value, list):
            value = '(any ndarray)'
        docs.append('    %s: %s' % (input_name, value))

    params = func_info['parameters']
    if params:
        docs.append('Parameters:')
    for param in params:
        docs.append('    %s: %s' % (param, params[param]))
        func_args.append('[%s=%s]' % (param, params[param]))
        defaults[param] = params[param]
        if param == 'matype':
            docs[-1] = ' '.join([docs[-1], '(%s)' % MA_Type[params[param]]])

    outputs = func_info['output_names']
    docs.append('Outputs:')
    for output in outputs:
        if output == 'integer':
            output = 'integer (values are -100, 0 or 100)'
        docs.append('    %s' % output)

    func_line.append(', '.join(func_args))
    func_line.append(')\n')
    docs.insert(0, ''.join(func_line))
    documentation = '\n'.join(docs)
    return defaults, documentation


###############    PRIVATE C-level-only functions    ###########################
# These map 1-1 with native C TALIB abstract interface calls. Their names are the
# same except for having the leading 4 characters lowercased.

# These functions are for:
# - Getting TALIB handle and paramholder pointers
# - Setting TALIB paramholder optInput values and calling the lookback function

cdef const lib.TA_FuncHandle*  __ta_getFuncHandle(char *function_name):
    """
    Returns a pointer to a function handle for the given function name
    """
    cdef const lib.TA_FuncHandle *handle
    _ta_check_success('TA_GetFuncHandle', lib.TA_GetFuncHandle(function_name, &handle))
    return handle

cdef lib.TA_ParamHolder* __ta_paramHolderAlloc(char *function_name):
    """
    Returns a pointer to a parameter holder for the given function name
    """
    cdef lib.TA_ParamHolder *holder
    retCode = lib.TA_ParamHolderAlloc(__ta_getFuncHandle(function_name), &holder)
    _ta_check_success('TA_ParamHolderAlloc', retCode)
    return holder

cdef int __ta_paramHolderFree(lib.TA_ParamHolder *params):
    """
    Frees the memory allocated by __ta_paramHolderAlloc (call when done with the parameter holder)
    WARNING: Not properly calling this function will cause memory leaks!
    """
    _ta_check_success('TA_ParamHolderFree', lib.TA_ParamHolderFree(params))

cdef int __ta_setOptInputParamInteger(lib.TA_ParamHolder *holder, int idx, int value):
    retCode = lib.TA_SetOptInputParamInteger(holder, idx, value)
    _ta_check_success('TA_SetOptInputParamInteger', retCode)

cdef int __ta_setOptInputParamReal(lib.TA_ParamHolder *holder, int idx, double value):
    retCode = lib.TA_SetOptInputParamReal(holder, idx, value)
    _ta_check_success('TA_SetOptInputParamReal', retCode)

cdef int __ta_getLookback(lib.TA_ParamHolder *holder):
    cdef int lookback
    retCode = lib.TA_GetLookback(holder, &lookback)
    _ta_check_success('TA_GetLookback', retCode)
    return lookback

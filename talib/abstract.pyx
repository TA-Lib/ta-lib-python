'''
This file Copyright (c) 2013 Brian A Cappello <briancappello at gmail>
'''

from talib import utils
from talib import func as ta_func
from collections import OrderedDict

cimport numpy as np
cimport abstract_h as abstract
from cython.operator cimport dereference as deref


#################    Public API for external python apps    ####################

def get_functions():
    ''' Returns a list of all the functions supported by TALIB
    '''
    ret = []
    for group in _ta_getGroupTable():
        ret.extend(_ta_getFuncTable(group))
    return ret

def get_groups_of_functions():
    ''' Returns a dict with kyes of function-group names and values of lists
    of function names ie {'group_names': ['function_names']}
    '''
    d = {}
    for group in _ta_getGroupTable():
        d[group] = _ta_getFuncTable(group)
    return d


class FuncHandle(object):
    ''' This is a pythonic wrapper around TALIB's abstract interface. It is
    intended to simplify using individual TALIB functions by providing a unified
    interface for setting/controlling input data, setting function parameters
    and retrieving results. Input data consists of a dict of numpy arrays, one
    array for each of open, high, low, close and volume. This can be set with
    set_input_arrays(input_dict). Which keyed array(s) are used as inputs when
    calling the function is controlled with get/set_inputs().

    This class gets initialized with a TALIB function name and optionally an
    input_arrays dict. It provides the following primary functions for setting
    inputs and retrieving results:

    ---- input_array/TA-function-parameter set-only functions -----
    - set_input_arrays(input_arrays)
    - set_function_parameters([input_arrays,] [param_args_andor_kwargs])

    Documentation for param_args_andor_kwargs can be printed with print_help()
    or programatically via get_info() / get_inputs() / get_parameters().

    ----- result-returning functions -----
    - get_outputs()
    - run([input_arrays])
    - FuncHandleInstance([input_arrays,] [param_args_andor_kwargs])
    '''
    def __init__(self, function_name, input_arrays=None):
        # make sure the function_name is valid and define all of our variables
        self.__name = function_name.upper()
        if self.__name not in get_functions():
            raise Exception('%s not supported by TA-LIB.' % self.__name)
        self.__info = None
        self.__input_arrays = { 'open': None,
                               'high': None,
                               'low': None,
                               'close': None,
                               'volume': None }

        # dictionaries of function args. keys are input/opt_input/output parameter names
        self.__inputs = OrderedDict()
        self.__opt_inputs = OrderedDict()
        self.__outputs = OrderedDict()
        self.__outputs_valid = False

        # lookup for TALIB input parameters which don't define expected price series inputs
        self.__input_price_series_defaults = { 'real': 'close',
                                               'real0': 'high',
                                               'real1': 'low',
                                               'periods': None }

        # finally query the TALIB abstract interface for the details of our function
        self.__initialize_private_variables()
        if input_arrays:
            self.set_input_arrays(input_arrays)

    def __initialize_private_variables(self):
        # function info
        self.__info = _ta_getFuncInfo(self.__name)

        # inputs (price series names)
        for i in xrange(self.__info.pop('num_inputs')):
            info = _ta_getInputParameterInfo(self.__name, i)
            input_name = info['name']
            if info['price_series'] == None:
                info['price_series'] = self.__input_price_series_defaults[input_name]
            self.__inputs[input_name] = info
        self.__info['inputs'] = self.get_inputs()

        # optional inputs (function parameters)
        for i in xrange(self.__info.pop('num_opt_inputs')):
            info = _ta_getOptInputParameterInfo(self.__name, i)
            param_name = info['name']
            self.__opt_inputs[param_name] = info
        self.__info['parameters'] = self.get_parameters()

        # outputs
        for i in xrange(self.__info.pop('num_outputs')):
            info = _ta_getOutputParameterInfo(self.__name, i)
            output_name = info['name']
            self.__outputs[output_name] = None
        self.__info['outputs'] = self.__outputs.keys()

    def print_help(self):
        ''' Prints the function parameter options and their values
        '''
        # get and format the function parameter names/defaults
        args = None
        func_params = []
        for input_name in self.__opt_inputs:
            value = self.__get_opt_input_value(input_name)
            func_params.append('%s=%i' % (input_name, value))
        if func_params:
            args = ', '.join(func_params)

        # get and format the function data series input names
        kwargs = None
        inputs = OrderedDict()
        for input_name in self.__inputs:
            if self.__inputs[input_name]['price_series'] == None:
                inputs[input_name] = self.__input_price_series_defaults[input_name]
        if inputs:
            kwargs = ', '.join(['%s="%s"' % (k, v) for k, v in inputs.items()])

        # print the docstring results
        if args or kwargs:
            print '%s.set_function_parameters(*args, **kwargs)' % self.__name
            if args: print 'args: %s' % args
            if kwargs: print 'kwargs: %s' % kwargs

    def get_info(self):
        ''' Returns a copy of the function's info dict.
        '''
        return self.__info.copy()

    def get_inputs(self):
        ''' Returns the dict of input price series names that specifies which
        of the ndarrays in input_arrays will be used to calculate the function.
        '''
        ret = OrderedDict()
        for input_ in self.__inputs:
            ret[input_] = self.__inputs[input_]['price_series']
        return ret

    def set_inputs(self, inputs):
        ''' Sets the input price series names to use.
        '''
        for input_, price_series in inputs.items():
            self.__inputs[input_]['price_series'] = price_series

    def get_input_arrays(self):
        ''' Returns a copy of the dict of input arrays in use.
        '''
        return self.__input_arrays.copy()

    def set_input_arrays(self, input_arrays):
        ''' Sets the dict of input_arrays to use. Returns True/False for subclasses:

        If input_arrays is a dict with the keys open, high, low, close and volume,
        it is assigned as the input_array to use. This function then returns True,
        returning False otherwise. This is meant so you can optionally wrap this
        function in an if-statement if you implement your own data type eg:

        class CustomFuncHandle(talib_abstract.FuncHandle):
            def __init__(self, function_name):
                talib_abstract.FuncHandle.__init__(self, function_name)

            def set_input_arrays(self, input_data):
                if talib_abstract.FuncHandle.set_input_arrays(self, input_data):
                    return
                elif isinstance(input_data, some_module.CustomDataType):
                    input_arrays = talib_abstract.FuncHandle.get_input_arrays(self)
                    # convert input_data to input_arrays and then call the super
                    talib_abstract.FuncHandle.set_input_arrays(self, input_arrays)
        '''
        if isinstance(input_arrays, dict) \
          and sorted(input_arrays.keys()) == ['close', 'high', 'low', 'open', 'volume']:
            self.__input_arrays = input_arrays
            return True
        return False

    def get_parameters(self):
        ''' Returns the function's optional parameters and their default values.
        '''
        ret = OrderedDict()
        for input_ in self.__opt_inputs:
            ret[input_] = self.__get_opt_input_value(input_)
        return ret

    def __get_opt_input_value(self, input_name):
        ''' Returns the user-set value if there is one, otherwise the default.
        '''
        value = self.__opt_inputs[input_name]['value']
        if not value:
            value = self.__opt_inputs[input_name]['default_value']
        return value

    def set_parameters(self, params):
        ''' Sets the function parameter values.
        '''
        for param, value in params.items():
            self.__opt_inputs[param]['value'] = value
        self.__outputs_valid = False

    def set_function_parameters(self, *args, **kwargs):
        ''' optionl args:[input_arrays,] [parameter_args,] [input_price_series_kwargs,] [parameter_kwargs]
        '''
        args = [arg for arg in args]
        if args or kwargs:
            self.__outputs_valid = False
        if args:
            first = args.pop(0)
            if not self.set_input_arrays(first):
                args.insert(0, first)
        for i, param_name in enumerate(self.__opt_inputs):
            if i < len(args):
                value = args[i]
                self.__opt_inputs[param_name]['value'] = value
        for key in kwargs:
            if key in self.__opt_inputs:
                self.__opt_inputs[key]['value'] = kwargs[key]
            elif key in self.__inputs:
                self.__inputs[key]['price_series'] = kwargs[key]

    def get_lookback(self):
        ''' Returns the lookback window size for the function with the parameter
        values that are currently set.
        '''
        return _ta_getLookback(self.__name, self.__opt_inputs)

    def get_output_names(self):
        ''' Returns a list of the output names returned by this function.
        '''
        return self.__outputs.keys()

    def get_outputs(self):
        ''' Returns an OrderedDict of the calculated function values.
        '''
        if not self.__outputs_valid:
            self.__call_function()
        return self.__outputs.copy()

    def run(self, input_arrays=None):
        ''' A shortcut to get_outputs() that also allows setting the input_arrays
        dict.
        '''
        if input_arrays:
            self.set_function_parameters(input_arrays)
        self.__call_function()
        return self.get_outputs()

    def __call__(self, *args, **kwargs):
        ''' A shortcut to get_outputs() that also allows setting the input_arrays
        dict and function parameters.
        '''
        self.set_function_parameters(*args, **kwargs)
        self.__call_function()
        return self.get_outputs()

    def __call_function(self):
        # figure out which price series names we're using for inputs
        input_price_series_names = []
        for input_name in self.__inputs:
            price_series = self.__inputs[input_name]['price_series']
            if isinstance(price_series, list): # TALIB-supplied input names
                for name in price_series:
                    input_price_series_names.append(name)
            else: # name came from self.__input_price_series_defaults
                input_price_series_names.append(price_series)

        # populate the ordered args we'll call the function with
        args = []
        for price_series in input_price_series_names:
            args.append( self.__input_arrays[price_series] )
        for opt_input in self.__opt_inputs:
            value = self.__get_opt_input_value(opt_input)
            args.append(value)

        # I use the talib module to actually call the function. It should be
        # possible to use the abstract interface as well, but I'm not sure what
        # practical benefit this might provide. See the  _ta_getLookback() for
        # the general idea of how to do it. The rest of the boiler-plate code is
        # already written (see all the __ta_set* functions).
        results = ta_func.__getattribute__(self.__name)(*args)
        if isinstance(results, np.ndarray):
            self.__outputs[self.__outputs.keys()[0]] = results
        else:
            for i, output in enumerate(self.__outputs):
                self.__outputs[output] = results[i]
        self.__outputs_valid = True


######################  INTERNAL python-level functions  #######################
'''
These map 1-1 with native C TALIB abstract interface calls. Their names are the
same except for having the leading 4 characters lowercased (and the Alloc/Free
function pairs which have been combined into single get functions)

These are TA function information-discovery calls. The FuncHandle class encapsulates
these functions into an easy-to-use, pythonic interface. It's therefore recommended
over using these functions directly.
'''

def _ta_getGroupTable():
    ''' Returns the list of available TALIB function group names.
    '''
    cdef abstract.TA_StringTable *table
    utils._check_success('TA_GroupTableAlloc', abstract.TA_GroupTableAlloc(&table))
    groups = []
    for i in xrange(table.size):
        groups.append(deref(&table.string[i]))
    utils._check_success('TA_GroupTableFree', abstract.TA_GroupTableFree(table))
    return groups

def _ta_getFuncTable(char *group):
    ''' Returns a list of the functions for the specified group name.
    '''
    cdef abstract.TA_StringTable *table
    utils._check_success('TA_FuncTableAlloc', abstract.TA_FuncTableAlloc(group, &table))
    functions = []
    for i in xrange(table.size):
        functions.append(deref(&table.string[i]))
    utils._check_success('TA_FuncTableFree', abstract.TA_FuncTableFree(table))
    return functions

def __get_flags(flag, flags_lookup_dict):
    ''' TA-LIB provides hints for multiple flags as a bitwise-ORed int. This
    function returns the flags from flag found in the provided flags_lookup_dict.
    '''
    # if the flag we got is out-of-range, it just means no extra info provided
    if flag < 1 or flag > 2**len(flags_lookup_dict)-1:
        return None

    # In this loop, i is essentially the bit-position, which represents an
    # input from flags_lookup_dict. We loop through as many flags_lookup_dict
    # bit-positions as we need to check, bitwise-ANDing each with flag for a hit.
    ret = []
    for i in xrange(len(flags_lookup_dict)):
        if 2**i & flag:
            ret.append(flags_lookup_dict[2**i])
    return ret

def _ta_getFuncInfo(char *function_name):
    ''' Returns the info dict for the function. It has the following keys: name,
    group, help, flags, num_inputs, num_opt_inputs and num_outputs.
    '''
    cdef abstract.TA_FuncInfo *info
    retCode = abstract.TA_GetFuncInfo(__ta_getFuncHandle(function_name), &info)
    utils._check_success('TA_GetFuncInfo', retCode)

    ta_func_flags = { 16777216: 'Output scale same as input',
                      67108864: 'Output is over volume',
                      134217728: 'Function has an unstable period',
                      268435456: 'Output is a candlestick' }

    ret = { 'name': info.name,
            'group': info.group,
            'display_name': info.hint,
            'flags': __get_flags(info.flags, ta_func_flags),
            'num_inputs': int(info.nbInput),
            'num_opt_inputs': int(info.nbOptInput),
            'num_outputs': int(info.nbOutput) }
    return ret

def _ta_getInputParameterInfo(char *function_name, int idx):
    ''' Returns the function's input info dict for the given index. It has two
    keys: name and flags.
    '''
    cdef abstract.TA_InputParameterInfo *info
    retCode = abstract.TA_GetInputParameterInfo(__ta_getFuncHandle(function_name), idx, &info)
    utils._check_success('TA_GetInputParameterInfo', retCode)

    # when flag is 0, the function (should) work on any reasonable input ndarray
    ta_input_flags = { 1: 'open',
                       2: 'high',
                       4: 'low',
                       8: 'close',
                       16: 'volumne',
                       32: 'openInterest',
                       64: 'timeStamp' }

    name = info.paramName
    name = name[len('in'):] # chop off leading 'in'
    name = ''.join([name[0].lower(), name[1:]]) # lowercase the first letter

    ret = { 'name': name,
            #'type': info.type,
            'price_series': __get_flags(info.flags, ta_input_flags) }
    return ret

def _ta_getOptInputParameterInfo(char *function_name, int idx):
    ''' Returns the function's opt_input info dict for the given index. It has
    the following keys: name, display_name, type, help, default_value and value.
    '''
    cdef abstract.TA_OptInputParameterInfo *info
    retCode = abstract.TA_GetOptInputParameterInfo(__ta_getFuncHandle(function_name), idx, &info)
    utils._check_success('TA_GetOptInputParameterInfo', retCode)

    name = info.paramName
    name = name[len('optIn'):] # chop off leading 'optIn'
    if not name.startswith('MA'):
        name = ''.join([name[0].lower(), name[1:]]) # lowercase the first letter
    default_value = info.defaultValue
    if default_value % 1 == 0:
        default_value = int(default_value)

    ret = { 'name': name,
            'display_name': info.displayName,
            'type': info.type,
            'help': info.hint,
            'default_value': default_value,
            'value': None }
    return ret

def _ta_getOutputParameterInfo(char *function_name, int idx):
    ''' Returns the function's output info dict for the given index. It has two
    keys: name and flags.
    '''
    cdef abstract.TA_OutputParameterInfo *info
    retCode = abstract.TA_GetOutputParameterInfo(__ta_getFuncHandle(function_name), idx, &info)
    utils._check_success('TA_GetOutputParameterInfo', retCode)

    name = info.paramName
    name = name[len('out'):] # chop off leading 'out'
    if 'Real' in name and name not in ['Real', 'Real0', 'Real1', 'Real2']:
        name = name[len('Real'):] # chop off leading 'Real' if a descriptive name follows
    name = ''.join([name[0].lower(), name[1:]]) # lowercase the first letter

    output_flag_64 = ', '.join([ 'Bearish < 0', 'Neutral = 0', 'Bullish > 0' ])
    output_flag_128 = ', '.join([ '[-200..-100] = Bearish',
                                    '[-100..0] = Getting Bearish',
                                    '0 = Neutral',
                                    '[0..100] = Getting Bullish',
                                    '[100-200] = Bullish' ])
    ta_output_flags = { 1: 'Line',
                        2: 'Dotted Line',
                        4: 'Dashed Line',
                        8: 'Dot',
                        16: 'Histogram',
                        32: 'Pattern (Bool)',
                        64: 'Bull/Bear Pattern (%s)' % output_flag_64,
                        128: 'Strength Pattern (%s)' % output_flag_128,
                        256: 'Output can be positive',
                        512: 'Output can be negative',
                        1024: 'Output can be zero',
                        2048: 'Values represent an upper limit',
                        4096: 'Values represent a lower limit' }

    ret = { 'name': name,
            #'type': info.type,
            'description': __get_flags(info.flags, ta_output_flags) }
    return ret

def _ta_getLookback(function, opt_inputs):
    cdef abstract.TA_ParamHolder *holder = __ta_paramHolderAlloc(function)
    for i, opt_input in enumerate(opt_inputs):
        value = opt_inputs[opt_input]['value']
        if not value:
            value = opt_inputs[opt_input]['default_value']

        type_ = opt_inputs[opt_input]['type']
        if type_ == abstract.TA_OptInput_RealRange or type_ == abstract.TA_OptInput_RealList:
            __ta_setOptInputParamReal(holder, i, value)
        elif type_ == abstract.TA_OptInput_IntegerRange or type_ == abstract.TA_OptInput_IntegerList:
            __ta_setOptInputParamInteger(holder, i, value)

    lookback = __ta_getLookback(holder)
    __ta_paramHolderFree(holder)
    return lookback


###############    PRIVATE C-level-only functions    ###########################
# These map 1-1 with native C TALIB abstract interface calls. Their names are the
# same except for having the leading 4 characters lowercased.

# These functions are for:
# - Gettinig TALIB handle and paramholder pointers
# - Setting TALIB paramholder input/output pointers and optInput values
# - Finally for calling the actual TALIB function (or its lookback) with all of
#   the function's data pointers and optInputs defined in the paramholder

# ---------- get TALIB func handle -------------------
cdef abstract.TA_FuncHandle*  __ta_getFuncHandle(char *function_name):
    ''' Returns a pointer to a function handle for the given function name
    '''
    cdef abstract.TA_FuncHandle *handle
    utils._check_success('TA_GetFuncHandle', abstract.TA_GetFuncHandle(function_name, &handle))
    return handle

# --------- get param holder (alloc/free) -------------
cdef abstract.TA_ParamHolder* __ta_paramHolderAlloc(char *function_name):
    ''' Returns a pointer to a parameter holder for the given function handle
    '''
    cdef abstract.TA_ParamHolder *holder
    retCode = abstract.TA_ParamHolderAlloc(__ta_getFuncHandle(function_name), &holder)
    utils._check_success('TA_ParamHolderAlloc', retCode)
    return holder

cdef int __ta_paramHolderFree(abstract.TA_ParamHolder *params):
    ''' Frees the memory allocated by __ta_paramHolderAlloc (call when done with the parameter holder)
    WARNING: Not properly calling this function will cause memory leaks!
    '''
    utils._check_success('TA_ParamHolderFree', abstract.TA_ParamHolderFree(params))

# --------- set input data pointers ----------------
cdef int __ta_setInputParamIntegerPtr(abstract.TA_ParamHolder *holder, int idx, int *in_ptr):
    retCode = abstract.TA_SetInputParamIntegerPtr(holder, idx, in_ptr)
    utils._check_success('TA_SetInputParamIntegerPtr', retCode)
    return retCode

cdef int __ta_setInputParamRealPtr(abstract.TA_ParamHolder *holder, int idx, abstract.TA_Real *in_ptr):
    retCode = abstract.TA_SetInputParamRealPtr(holder, idx, in_ptr)
    utils._check_success('TA_SetInputParamRealPtr', retCode)
    return retCode

cdef int __ta_setInputParamPricePtr(abstract.TA_ParamHolder *holder, int idx,
    abstract.TA_Real *open_,
    abstract.TA_Real *high,
    abstract.TA_Real *low,
    abstract.TA_Real *close,
    abstract.TA_Real *volume,
    abstract.TA_Real *openInterest
):
    retCode = abstract.TA_SetInputParamPricePtr(holder, idx,
        open_, high, low, close, volume, openInterest)
    utils._check_success('TA_SetInputParamPricePtr', retCode)
    return retCode

# ---------- set opt input parameter values ----------------
cdef int __ta_setOptInputParamInteger(abstract.TA_ParamHolder *holder, int idx, int value):
    retCode = abstract.TA_SetOptInputParamInteger(holder, idx, value)
    utils._check_success('TA_SetOptInputParamInteger', retCode)

cdef int __ta_setOptInputParamReal(abstract.TA_ParamHolder *holder, int idx, int value):
    retCode = abstract.TA_SetOptInputParamReal(holder, idx, value)
    utils._check_success('TA_SetOptInputParamReal', retCode)

# --------- set output data pointers ----------------
cdef int __ta_setOutputParamIntegerPtr(abstract.TA_ParamHolder *holder, int idx, int *out_ptr):
    retCode = abstract.TA_SetOutputParamIntegerPtr(holder, idx, out_ptr)
    utils._check_success('TA_SetOutputParamIntegerPtr', retCode)
    return retCode

cdef int __ta_setOutputParamRealPtr(abstract.TA_ParamHolder *holder, int idx, abstract.TA_Real *out_ptr):
    retCode = abstract.TA_SetOutputParamRealPtr(holder, idx, out_ptr)
    utils._check_success('TA_SetOutputParamRealPtr', retCode)
    return retCode

# ----------- get lookback ---------------
cdef int __ta_getLookback(abstract.TA_ParamHolder *holder):
    cdef int lookback
    retCode = abstract.TA_GetLookback(holder, &lookback)
    utils._check_success('TA_GetLookback', retCode)
    return lookback

# ----------- call TALIB function -------------
cdef __ta_callFunc(abstract.TA_ParamHolder *holder, int startIdx=0, int endIdx=0):
    cdef int outBegIdx
    cdef int outNbElement
    retCode = abstract.TA_CallFunc( holder,
                           startIdx,
                           endIdx,
                           &outBegIdx,
                           &outNbElement )
    utils._check_success('TA_CallFunc', retCode)
    return (retCode, outBegIdx, outNbElement)

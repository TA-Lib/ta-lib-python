
cimport libta_lib as lib
from libta_lib cimport TA_RetCode, TA_FuncUnstId

__ta_version__ = lib.TA_GetVersionString()

cpdef _ta_check_success(str function_name, TA_RetCode ret_code):
    if ret_code == lib.TA_SUCCESS:
        return True
    ta_errors = {
        0: 'Success',
        1: 'Library Not Initialized (TA_LIB_NOT_INITIALIZE)',
        2: 'Bad Parameter (TA_BAD_PARAM)',
        3: 'Allocation Error (TA_ALLOC_ERR)',
        4: 'Group Not Found (TA_GROUP_NOT_FOUND)',
        5: 'Function Not Found (TA_FUNC_NOT_FOUND)',
        6: 'Invalid Handle (TA_INVALID_HANDLE)',
        7: 'Invalid Parameter Holder (TA_INVALID_PARAM_HOLDER)',
        8: 'Invalid Parameter Holder Type (TA_INVALID_PARAM_HOLDER_TYPE)',
        9: 'Invalid Parameter Function (TA_INVALID_PARAM_FUNCTION)',
        10: 'Input Not All Initialized (TA_INPUT_NOT_ALL_INITIALIZE)',
        11: 'Output Not All Initialized (TA_OUTPUT_NOT_ALL_INITIALIZE)',
        12: 'Out-of-Range Start Index (TA_OUT_OF_RANGE_START_INDEX)',
        13: 'Out-of-Range End Index (TA_OUT_OF_RANGE_END_INDEX)',
        14: 'Invalid List Type (TA_INVALID_LIST_TYPE)',
        15: 'Bad Object (TA_BAD_OBJECT)',
        16: 'Not Supported (TA_NOT_SUPPORTED)',
        5000: 'Internal Error (TA_INTERNAL_ERROR)',
        65535: 'Unknown Error (TA_UNKNOWN_ERR)',
        }
    raise Exception('%s function failed with error code %s: %s' % (
        function_name, ret_code, ta_errors[ret_code]))

def _ta_initialize():
    cdef TA_RetCode ret_code
    ret_code = lib.TA_Initialize()
    _ta_check_success('TA_Initialize', ret_code)

def _ta_shutdown():
    cdef TA_RetCode ret_code
    ret_code = lib.TA_Shutdown()
    _ta_check_success('TA_Shutdown', ret_code)

class MA_Type(object):
    SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3 = range(9)

    def __init__(self):
        self._lookup = {
            MA_Type.SMA: 'Simple Moving Average',
            MA_Type.EMA: 'Exponential Moving Average',
            MA_Type.WMA: 'Weighted Moving Average',
            MA_Type.DEMA: 'Double Exponential Moving Average',
            MA_Type.TEMA: 'Triple Exponential Moving Average',
            MA_Type.TRIMA: 'Triangular Moving Average',
            MA_Type.KAMA: 'Kaufman Adaptive Moving Average',
            MA_Type.MAMA: 'MESA Adaptive Moving Average',
            MA_Type.T3: 'Triple Generalized Double Exponential Moving Average',
            }

    def __getitem__(self, type_):
        return self._lookup[type_]

MA_Type = MA_Type()

_ta_func_unst_ids = {'NONE': -1}
for i, name in enumerate([
            'ADX', 'ADXR', 'ATR', 'CMO', 'DX', 'EMA', 'HT_DCPERIOD',
            'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDLINE',
            'HT_TRENDMODE', 'KAMA', 'MAMA', 'MFI', 'MINUS_DI', 'MINUS_DM',
            'NATR', 'PLUS_DI', 'PLUS_DM', 'RSI', 'STOCHRSI', 'T3', 'ALL'
        ]):
    _ta_func_unst_ids[name] = i

def _ta_set_unstable_period(name, period):
    cdef TA_RetCode ret_code
    cdef TA_FuncUnstId id = _ta_func_unst_ids[name]
    ret_code = lib.TA_SetUnstablePeriod(id, period)
    _ta_check_success('TA_SetUnstablePeriod', ret_code)

def _ta_get_unstable_period(name):
    cdef unsigned int period
    cdef TA_FuncUnstId id = _ta_func_unst_ids[name]
    period = lib.TA_GetUnstablePeriod(id)
    return period

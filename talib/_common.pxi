cimport _ta_lib as lib
from _ta_lib cimport TA_RetCode, TA_FuncUnstId

__ta_version__ = lib.TA_GetVersionString()

cpdef _ta_check_success(str function_name, TA_RetCode ret_code):
    if ret_code == 0:
        return True
    elif ret_code == 1:
        description = 'Library Not Initialized (TA_LIB_NOT_INITIALIZE)'
    elif ret_code == 2:
        description = 'Bad Parameter (TA_BAD_PARAM)'
    elif ret_code == 3:
        description = 'Allocation Error (TA_ALLOC_ERR)'
    elif ret_code == 4:
        description = 'Group Not Found (TA_GROUP_NOT_FOUND)'
    elif ret_code == 5:
        description = 'Function Not Found (TA_FUNC_NOT_FOUND)'
    elif ret_code == 6:
        description = 'Invalid Handle (TA_INVALID_HANDLE)'
    elif ret_code == 7:
        description = 'Invalid Parameter Holder (TA_INVALID_PARAM_HOLDER)'
    elif ret_code == 8:
        description = 'Invalid Parameter Holder Type (TA_INVALID_PARAM_HOLDER_TYPE)'
    elif ret_code == 9:
        description = 'Invalid Parameter Function (TA_INVALID_PARAM_FUNCTION)'
    elif ret_code == 10:
        description = 'Input Not All Initialized (TA_INPUT_NOT_ALL_INITIALIZE)'
    elif ret_code == 11:
        description = 'Output Not All Initialized (TA_OUTPUT_NOT_ALL_INITIALIZE)'
    elif ret_code == 12:
        description = 'Out-of-Range Start Index (TA_OUT_OF_RANGE_START_INDEX)'
    elif ret_code == 13:
        description = 'Out-of-Range End Index (TA_OUT_OF_RANGE_END_INDEX)'
    elif ret_code == 14:
        description = 'Invalid List Type (TA_INVALID_LIST_TYPE)'
    elif ret_code == 15:
        description = 'Bad Object (TA_BAD_OBJECT)'
    elif ret_code == 16:
        description = 'Not Supported (TA_NOT_SUPPORTED)'
    elif ret_code == 5000:
        description = 'Internal Error (TA_INTERNAL_ERROR)'
    elif ret_code == 65535:
        description = 'Unknown Error (TA_UNKNOWN_ERR)'
    else:
        description = 'Unknown Error'
    raise Exception('%s function failed with error code %s: %s' % (
        function_name, ret_code, description))

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

def _ta_set_compatibility(value):
    cdef TA_RetCode ret_code
    ret_code = lib.TA_SetCompatibility(value)
    _ta_check_success('TA_SetCompatibility', ret_code)

def _ta_get_compatibility():
    cdef int value
    value = lib.TA_GetCompatibility()
    return value

class CandleSettingType(object):
    BodyLong, BodyVeryLong, BodyShort, BodyDoji, ShadowLong, ShadowVeryLong, \
    ShadowShort, ShadowVeryShort, Near, Far, Equal, AllCandleSettings = \
    range(12)

CandleSettingType = CandleSettingType()

class RangeType(object):
    RealBody, HighLow, Shadows = range(3)

RangeType = RangeType()

def _ta_set_candle_settings(settingtype, rangetype, avgperiod, factor):
    cdef TA_RetCode ret_code
    ret_code = lib.TA_SetCandleSettings(settingtype, rangetype, avgperiod, factor)
    _ta_check_success('TA_SetCandleSettings', ret_code)

def _ta_restore_candle_default_settings(settingtype):
    cdef TA_RetCode ret_code
    ret_code = lib.TA_RestoreCandleDefaultSettings(settingtype)
    _ta_check_success('TA_RestoreCandleDefaultSettings', ret_code)

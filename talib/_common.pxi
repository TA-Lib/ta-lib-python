import warnings

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

# Take every id straight from the C header instead of hardcoding a list.
# Cython resolves each `lib.TA_FUNC_UNST_*` at build time from the ta_defs.h
# it compiles against, so this one table is correct for every TA-Lib release.
_ta_func_unst_ids = {
    # Not sourced from the header: TA_FUNC_UNST_NONE does not exist in newer
    # ta-lib releases, and -1 is a Python-side sentinel anyway.
    'NONE': -1,
    'ADX': lib.TA_FUNC_UNST_ADX,
    'ATR': lib.TA_FUNC_UNST_ATR,
    'CMO': lib.TA_FUNC_UNST_CMO,
    'DX': lib.TA_FUNC_UNST_DX,
    'EMA': lib.TA_FUNC_UNST_EMA,
    'HT_DCPERIOD': lib.TA_FUNC_UNST_HT_DCPERIOD,
    'HT_DCPHASE': lib.TA_FUNC_UNST_HT_DCPHASE,
    'HT_PHASOR': lib.TA_FUNC_UNST_HT_PHASOR,
    'HT_SINE': lib.TA_FUNC_UNST_HT_SINE,
    'HT_TRENDLINE': lib.TA_FUNC_UNST_HT_TRENDLINE,
    'HT_TRENDMODE': lib.TA_FUNC_UNST_HT_TRENDMODE,
    'KAMA': lib.TA_FUNC_UNST_KAMA,
    'MAMA': lib.TA_FUNC_UNST_MAMA,
    'MINUS_DI': lib.TA_FUNC_UNST_MINUS_DI,
    'MINUS_DM': lib.TA_FUNC_UNST_MINUS_DM,
    'NATR': lib.TA_FUNC_UNST_NATR,
    'PLUS_DI': lib.TA_FUNC_UNST_PLUS_DI,
    'PLUS_DM': lib.TA_FUNC_UNST_PLUS_DM,
    'RSI': lib.TA_FUNC_UNST_RSI,
    'T3': lib.TA_FUNC_UNST_T3,
    'ALL': lib.TA_FUNC_UNST_ALL,
}

# Still accepted so existing code keeps running, but they are no longer knobs
# in TA-Lib C: their enumerators are now TA_FUNC_UNST_UNUSED_*.  They are not
# aliased to their inner indicator on purpose -- turning a long-standing no-op
# into a call that changes ADX or RSI for every other function would be a
# worse surprise than doing nothing.
_ta_func_unst_retired = {
    'ADXR': "ADXR has no unstable period of its own; it follows the inner ADX. Set 'ADX' instead.",
    'MFI': "MFI's unstable period was retired in TA-Lib C.",
    'STOCHRSI': "STOCHRSI has no unstable period of its own; it follows the inner RSI. Set 'RSI' instead.",
}

def _ta_func_unst_id(name):
    if name in _ta_func_unst_retired:
        warnings.warn(
            "unstable period '%s' is deprecated and has no effect: %s" % (
                name, _ta_func_unst_retired[name]),
            DeprecationWarning, stacklevel=3)
        return None
    return _ta_func_unst_ids[name]

def _ta_set_unstable_period(name, period):
    cdef TA_RetCode ret_code
    cdef TA_FuncUnstId id
    unst_id = _ta_func_unst_id(name)
    if unst_id is None:
        return
    id = unst_id
    ret_code = lib.TA_SetUnstablePeriod(id, period)
    _ta_check_success('TA_SetUnstablePeriod', ret_code)

def _ta_get_unstable_period(name):
    cdef unsigned int period
    cdef TA_FuncUnstId id
    unst_id = _ta_func_unst_id(name)
    if unst_id is None:
        return 0
    id = unst_id
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

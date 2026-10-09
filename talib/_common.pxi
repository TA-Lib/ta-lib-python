import warnings
cimport cython

cimport _ta_lib as lib
from _ta_lib cimport TA_RetCode, TA_FuncUnstId

__ta_version__ = lib.TA_GetVersionString()

# The gate protects bookkeeping only: independent readers compute concurrently.
# A settings change fails rather than waiting for a live stream or a reader
# (which could itself be invoking the setter from an allocation callback).
cdef cython.pymutex _ta_gate
cdef unsigned long _ta_readers = 0
cdef unsigned long _ta_live_streams = 0
cdef bint _ta_writing = False
cdef bint _ta_initialized = False

cdef bint _ta_begin_read() noexcept nogil:
    global _ta_readers
    with _ta_gate:
        if _ta_writing or not _ta_initialized:
            return False
        _ta_readers += 1
        return True

cdef void _ta_end_read() noexcept nogil:
    global _ta_readers
    with _ta_gate:
        _ta_readers -= 1

cdef bint _ta_begin_write() noexcept nogil:
    global _ta_writing
    with _ta_gate:
        if _ta_writing or _ta_readers or _ta_live_streams:
            return False
        _ta_writing = True
        return True

cdef void _ta_end_write() noexcept nogil:
    global _ta_writing
    with _ta_gate:
        _ta_writing = False

cdef void _ta_stream_registered() noexcept nogil:
    global _ta_live_streams
    with _ta_gate:
        _ta_live_streams += 1

cdef void _ta_stream_closed() noexcept nogil:
    global _ta_live_streams
    with _ta_gate:
        _ta_live_streams -= 1

cdef void _ta_acquire_read() except *:
    cdef bint acquired
    with nogil:
        acquired = _ta_begin_read()
    if not acquired:
        raise RuntimeError("TA-Lib is not initialized or a settings change is in progress")

cdef class _TA_ReadLease:
    def __enter__(self):
        _ta_acquire_read()

    def __exit__(self, *args):
        with nogil:
            _ta_end_read()

cdef class _TA_WriteLease:
    def __enter__(self):
        cdef bint acquired
        with nogil:
            acquired = _ta_begin_write()
        if not acquired:
            raise RuntimeError("cannot change TA-Lib settings or lifecycle while calls or streams are active")

    def __exit__(self, *args):
        with nogil:
            _ta_end_write()


class InsufficientHistory(Exception):
    """Not enough history to open a stream (TA_INSUFFICIENT_HISTORY).

    Recoverable: collect more bars and try again."""


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
    elif ret_code == 17:
        description = 'Insufficient History (TA_INSUFFICIENT_HISTORY)'
    elif ret_code == 5000:
        description = 'Internal Error (TA_INTERNAL_ERROR)'
    elif ret_code == 65535:
        description = 'Unknown Error (TA_UNKNOWN_ERR)'
    else:
        description = 'Unknown Error'
    error = InsufficientHistory if ret_code == 17 else Exception
    raise error('%s function failed with error code %s: %s' % (
        function_name, ret_code, description))

def _ta_initialize():
    global _ta_initialized
    cdef TA_RetCode ret_code
    with _TA_WriteLease():
        if not _ta_initialized:
            with nogil:
                ret_code = lib.TA_Initialize()
            _ta_check_success('TA_Initialize', ret_code)
            _ta_initialized = True

def _ta_shutdown(bint at_exit=False):
    global _ta_initialized
    cdef TA_RetCode ret_code
    cdef bint acquired
    with nogil:
        acquired = _ta_begin_write()
    if not acquired:
        if at_exit:
            # Workers and module globals may still own handles at interpreter
            # exit. Leave the C library intact for their finalizers.
            return
        raise RuntimeError("cannot shut down TA-Lib while calls or streams are active")
    try:
        if _ta_initialized:
            with nogil:
                ret_code = lib.TA_Shutdown()
            _ta_check_success('TA_Shutdown', ret_code)
            _ta_initialized = False
    finally:
        with nogil:
            _ta_end_write()

class MA_Type(object):
    SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3, HMA, DISABLED, DEFAULT, \
        ZLEMA, RMA = range(14)

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
            MA_Type.HMA: 'Hull Moving Average',
            MA_Type.DISABLED: 'No Moving Average (identity)',
            MA_Type.DEFAULT: "The Function's Own Default Moving Average",
            MA_Type.ZLEMA: 'Zero-Lag Exponential Moving Average',
            MA_Type.RMA: "Wilder's Smoothed Moving Average",
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
    'RMA': lib.TA_FUNC_UNST_RMA,
    'HA': lib.TA_FUNC_UNST_HA,
    'RVI': lib.TA_FUNC_UNST_RVI,
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
    cdef unsigned int c_period
    unst_id = _ta_func_unst_id(name)
    if unst_id is None:
        return
    id = unst_id
    c_period = period
    with _TA_WriteLease():
        with nogil:
            ret_code = lib.TA_SetUnstablePeriod(id, c_period)
    _ta_check_success('TA_SetUnstablePeriod', ret_code)

def _ta_get_unstable_period(name):
    cdef unsigned int period
    cdef TA_FuncUnstId id
    unst_id = _ta_func_unst_id(name)
    if unst_id is None:
        return 0
    id = unst_id
    with _TA_ReadLease():
        with nogil:
            period = lib.TA_GetUnstablePeriod(id)
    return period

def _ta_set_compatibility(value):
    cdef TA_RetCode ret_code
    cdef int c_value = value
    with _TA_WriteLease():
        with nogil:
            ret_code = lib.TA_SetCompatibility(c_value)
    _ta_check_success('TA_SetCompatibility', ret_code)

def _ta_get_compatibility():
    cdef int value
    with _TA_ReadLease():
        with nogil:
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

def _ta_set_candle_settings(int settingtype, int rangetype, int avgperiod, double factor):
    cdef TA_RetCode ret_code
    with _TA_WriteLease():
        with nogil:
            ret_code = lib.TA_SetCandleSettings(settingtype, rangetype, avgperiod, factor)
    _ta_check_success('TA_SetCandleSettings', ret_code)

def _ta_restore_candle_default_settings(int settingtype):
    cdef TA_RetCode ret_code
    with _TA_WriteLease():
        with nogil:
            ret_code = lib.TA_RestoreCandleDefaultSettings(settingtype)
    _ta_check_success('TA_RestoreCandleDefaultSettings', ret_code)

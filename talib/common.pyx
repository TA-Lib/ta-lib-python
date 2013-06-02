
cimport libc as lib

ctypedef int TA_RetCode

__ta_version__ = lib.TA_GetVersionString()

cpdef _ta_check_success(str function_name, int ret_code):
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
    return ret_code

def _ta_shutdown():
    cdef TA_RetCode ret_code
    ret_code = lib.TA_Shutdown()
    _ta_check_success('TA_Shutdown', ret_code)
    return ret_code

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

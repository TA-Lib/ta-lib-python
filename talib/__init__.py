
__version__ = '0.4.3-git'

class MA(object):
    SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3 = range(9)

    def __init__(self):
        self._lookup = {
            MA.SMA: 'Simple Moving Average',
            MA.EMA: 'Exponential Moving Average',
            MA.WMA: 'Weighted Moving Average',
            MA.DEMA: 'Double Exponential Moving Average',
            MA.TEMA: 'Triple Exponential Moving Average',
            MA.TRIMA: 'Triangular Moving Average',
            MA.KAMA: 'Kaufman Adaptive Moving Average',
            MA.MAMA: 'MESA Adaptive Moving Average',
            MA.T3: 'Triple Generalized Double Exponential Moving Average',
            }

    def __getitem__(self, type_):
        return self._lookup[type_]

MA = MA()


def _check_success(function_name, ret_code):
    if ret_code == 0: # TA_SUCCESS
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
    if not isinstance(ret_code, int):
        ret_code = int(ret_code, 16)
    raise Exception('%s function failed with error code %s: %s' % (
        ta_function_name, ret_code, ta_errors[ret_code]))

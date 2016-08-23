from .c_ta_lib import __TA_FUNCTION_NAMES__


_ta_func = __import__(
    "c_ta_lib", globals(), locals(),
    __TA_FUNCTION_NAMES__, level=1
)
globals().update({
    func_name: getattr(_ta_func, func_name)
    for func_name in __TA_FUNCTION_NAMES__
})


__all__ = __TA_FUNCTION_NAMES__

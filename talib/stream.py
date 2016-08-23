from .c_ta_lib import __TA_FUNCTION_NAMES__


_ta_stream_func = __import__(
    "c_ta_lib", globals(), locals(),
    __TA_FUNCTION_NAMES__, level=1
)

for func_name in __TA_FUNCTION_NAMES__:
    globals()[func_name] = getattr(_ta_stream_func, "stream_%s" % func_name)

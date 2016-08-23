from .c_ta_lib import Function as _Function, __TA_FUNCTION_NAMES__


_ta_func = __import__(
    "c_ta_lib", globals(), locals(),
    __TA_FUNCTION_NAMES__, level=1
)

_func_obj_mapping = {
    func_name: getattr(_ta_func, func_name)
    for func_name in __TA_FUNCTION_NAMES__
}


def Function(function_name, *args, **kwargs):
    func_name = function_name.upper()
    if func_name not in _func_obj_mapping:
        raise Exception('%s not supported by TA-LIB.' % func_name)

    return _Function(
        func_name, _func_obj_mapping[func_name], *args, **kwargs
    )


for func_name in __TA_FUNCTION_NAMES__:
    globals()[func_name] = Function(func_name)


__all__ = ["Function"] + __TA_FUNCTION_NAMES__

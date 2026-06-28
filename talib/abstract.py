import talib._ta_lib as _ta_lib
from ._ta_lib import Function as _Function, __TA_FUNCTION_NAMES__, _get_defaults_and_docs

# add some backwards compat for backtrader
from ._ta_lib import TA_FUNC_FLAGS, TA_INPUT_FLAGS, TA_OUTPUT_FLAGS

_func_obj_mapping = {
    func_name: getattr(_ta_lib, func_name)
    for func_name in __TA_FUNCTION_NAMES__
}


class _FunctionProxy:
    def __init__(self, func_name, func_obj):
        object.__setattr__(self, '_func_name', func_name)
        object.__setattr__(self, '_func_obj', func_obj)

    def __call__(self, *args, **kwargs):
        if self._func_name in ('MACD', 'MACDFIX') and kwargs.get('signalperiod') == 1:
            raise ValueError(
                f"signalperiod=1 is not supported for {self._func_name} because the underlying TA-Lib "
                "implementation can produce look-ahead affected results; use signalperiod >= 2 instead."
            )
        return self._func_obj(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._func_obj, item)

    def __setattr__(self, key, value):
        if key in {'_func_name', '_func_obj'}:
            object.__setattr__(self, key, value)
            return
        setattr(self._func_obj, key, value)


def Function(function_name, *args, **kwargs):
    func_name = function_name.upper()
    if func_name not in _func_obj_mapping:
        raise Exception('%s not supported by TA-LIB.' % func_name)

    return _FunctionProxy(
        func_name,
        _Function(func_name, _func_obj_mapping[func_name], *args, **kwargs)
    )


for func_name in __TA_FUNCTION_NAMES__:
    globals()[func_name] = Function(func_name)


__all__ = ["Function", "_get_defaults_and_docs"] + __TA_FUNCTION_NAMES__

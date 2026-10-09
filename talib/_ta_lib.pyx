#cython: embedsignature=True, emit_code_comments=False, freethreading_compatible=True

include "_common.pxi"
include "_func.pxi"
include "_abstract.pxi"
include "_stream.pxi"

__all__ = __TA_FUNCTION_NAMES__

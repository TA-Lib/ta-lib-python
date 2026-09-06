"""Streaming API: one handle per indicator, O(1) per closed bar.

A handle is opened on history, then carried forward one bar at a time. Every
value it produces is bit-identical to what the Function API reports for the
same bar.

    from talib import stream

    s = stream.RSI(history, timeperiod=14)   # same arguments as talib.RSI
    s.value                                  # value at the last history bar

    for bar in feed:
        emit(s.update(bar))                  # one closed bar in, its value out

    s.peek(forming)                          # what update would return; commits
                                             # nothing, so call it as often as
                                             # the forming bar is revised
    fork = s.copy()                          # independent handle at the same bar
    s.advance()                              # count a bar you did not feed
    s.out_range                              # (begidx, nbelement), as batch reports

Opening needs at least ``lookback + 1`` bars, which ``abstract`` knows -- and a
little more where a function's seeding does, so treat a short history as "not
yet" rather than computing the number:

    while handle is None:
        history.append(next(feed))
        try:
            handle = stream.RSI(history, timeperiod=14)
        except talib.InsufficientHistory:
            pass

Leading bars that are NaN in any input are not history: they are skipped, as the
Function API skips them, and do not count toward the warm-up. A NaN or an
infinity anywhere else in the history is undefined behaviour in TA-Lib C, and
for a few window functions a handle and the Function API do then disagree.

A bar that is not finite is likewise rejected: ``update`` raises and the handle
is left exactly as it was, neither its value nor its range moved. For a bar you
mean to skip rather than re-feed, say so with ``advance()``, or two handles on
one feed drift a bar apart.

``open_and_fill`` is the alternate constructor for when you want the batch
series over the history as well -- one pass gives both:

    s, rsi = stream.RSI.open_and_fill(history, timeperiod=14)

A multi-output function answers with a named tuple whose fields are the output
names in the function's docstring; a single-output one with a bare float (or
int). Handles cannot be pickled.
"""
import talib._ta_lib as _ta_lib
from talib._ta_lib import OutRange, Stream, __TA_FUNCTION_NAMES__

__all__ = ['Stream', 'OutRange']

for func_name in __TA_FUNCTION_NAMES__:
    globals()[func_name] = getattr(_ta_lib, '%s_Stream' % func_name)
    __all__.append(func_name)
    # the named tuple a multi-output handle answers with
    value_name = '%s_Value' % func_name
    value_type = getattr(_ta_lib, value_name, None)
    if value_type is not None:
        globals()[value_name] = value_type
        __all__.append(value_name)

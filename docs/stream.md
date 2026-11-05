# Streaming API

The Streaming API keeps a handle per indicator instead of recomputing from the
whole array. Opening one costs a pass over the history; each closed bar after
that takes O(1) work. Values agree with the Function API for the same bars.
On Apple Silicon macOS, functions using transcendental math (such as SIN,
EXP and LN) may differ in the last bits.

## Open a handle and feed closed bars

The constructor takes the same input arrays and parameters as the corresponding
Function API call. Its `value` property reports the result at the last history
bar. After opening, pass scalar inputs for each new bar to `update()`:

```python
import numpy as np
import talib
from talib import abstract, stream

close = np.arange(1.0, 101.0)
s = stream.SMA(close, timeperiod=30)
assert s.value == talib.SMA(close, timeperiod=30)[-1]

for price in (101.0, 102.0, 103.0):
    latest = s.update(price)
    assert latest == s.value
```

A single-output function returns a float, or an int for an integer output.
A multi-output function returns a plain tuple in the same output order as the
Function API. Supply multiple scalar inputs in the constructor's input order:

```python
m = stream.MACD(close)
macd, macdsignal, macdhist = m.update(101.0)
assert m.value[2] == macdhist
assert abstract.Function('MACD').output_names == ['macd', 'macdsignal', 'macdhist']

high, low = close + 1.0, close - 1.0
atr = stream.ATR(high, low, close, timeperiod=14)
latest_atr = atr.update(102.0, 100.0, 101.0)
```

See [All Functions](funcs.md) for indicator inputs, parameters and output order.

## Preview and copy

Use `peek()` for a forming bar that may still change. It returns what `update()`
would produce without committing the bar or changing the handle's value or
range. Call `update()` once when that bar closes. `copy()` creates an independent
handle at the same position:

```python
before = s.value
preview = s.peek(104.0)
assert s.value == before
fork = s.copy()
assert fork.update(104.0) == preview
assert s.update(104.0) == preview
```

## Warm-up and missing bars

Opening requires at least `lookback + 1` bars, and some indicators need more
history for seeding. Catch `talib.InsufficientHistory` and collect more bars
before trying again:

```python
need = abstract.Function('RSI', timeperiod=14).lookback + 1
history = close[:5]
try:
    rsi = stream.RSI(history, timeperiod=14)
except talib.InsufficientHistory:
    pass  # Collect more history before opening again.
```

Leading bars containing NaN in any input are skipped and do not count toward
warm-up. NaN or infinity elsewhere in the opening history has undefined
behavior in TA-Lib C; for some window functions, streaming and batch results
can disagree. Supply finite history after the leading NaNs.

`update()` rejects a nonfinite input and leaves both the value and range
unchanged. If you intend to skip that bar rather than retry it, call `advance()`
to count it. This keeps handles following the same feed aligned.
`advance()` moves the range while holding the current value; it does not feed
a replacement price into the indicator.

`out_range` is an `OutRange(begidx, nbelement)` named tuple describing the output
range in input-series coordinates:

```python
output_range = s.out_range
first_output_bar = output_range.begidx
output_count = output_range.nbelement
s.advance()
```

## Get history outputs and a handle together

`open_and_fill()` returns the handle and the Function API's output series for
the opening history in one pass. Multiple outputs are returned as a tuple of
arrays:

```python
rsi, values = stream.RSI.open_and_fill(close, timeperiod=14)
np.testing.assert_array_equal(values, talib.RSI(close, timeperiod=14))
assert rsi.value == values[-1]
next_rsi = rsi.update(101.0)
```

## Lifetime and threads

Handles cannot be pickled or shared with another process. Keep the history and
reopen a handle in the other process instead.

Native stream operations release the GIL on regular CPython. Independent
handles can run concurrently; operations on one handle use a per-handle mutex.
Do not modify or resize input arrays during a call.

Unstable periods, compatibility and candle settings are process-wide. Set them
before opening streams. Changing settings while a calculation is active or
any stream handle (including a copy) is alive raises `RuntimeError`. Release
all handles before changing those settings.

[Documentation Index](doc_index.md)

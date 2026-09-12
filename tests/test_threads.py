from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.testing import assert_array_equal

import talib
from talib import abstract, func, stream

THREADS = 8
ROUNDS = 50


def _calls(series, ford_2012):
    o, h, lo, c = (ford_2012[k] for k in ('open', 'high', 'low', 'close'))
    return [
        lambda: func.EMA(series, timeperiod=30),
        lambda: func.RSI(series),
        lambda: func.MACD(series),
        lambda: func.CDLDOJI(o, h, lo, c),
        lambda: func.ATR(h, lo, c),
        lambda: stream.EMA(series, timeperiod=30),
        lambda: abstract.Function('BBANDS')(series),
    ]


def _flatten(result):
    if isinstance(result, (tuple, list)):
        return [np.asarray(r) for r in result]
    return [np.asarray(result)]


def test_functions_are_correct_when_called_concurrently(series, ford_2012):
    # The C call runs with the GIL released, so several threads may be inside
    # TA-Lib at once. Every result must still equal the single-threaded one.
    calls = _calls(series, ford_2012)
    expected = [_flatten(call()) for call in calls]

    def worker(i):
        call = calls[i % len(calls)]
        return i % len(calls), _flatten(call())

    with ThreadPoolExecutor(THREADS) as pool:
        for which, got in pool.map(worker, range(THREADS * ROUNDS)):
            for g, e in zip(got, expected[which]):
                assert_array_equal(g, e)


def test_settings_still_work_between_concurrent_calls(series):
    # Global settings stay under the GIL. Changing one between batches of
    # concurrent calls must apply to every later call.
    talib.set_unstable_period('EMA', 0)
    with ThreadPoolExecutor(THREADS) as pool:
        base = list(pool.map(lambda _: func.EMA(series, timeperiod=30), range(THREADS)))
    talib.set_unstable_period('EMA', 10)
    try:
        with ThreadPoolExecutor(THREADS) as pool:
            shifted = list(pool.map(lambda _: func.EMA(series, timeperiod=30), range(THREADS)))
    finally:
        talib.set_unstable_period('EMA', 0)
    for b, s in zip(base, shifted):
        assert_array_equal(b, base[0])
        assert_array_equal(s, shifted[0])
    assert np.isnan(shifted[0]).sum() == np.isnan(base[0]).sum() + 10

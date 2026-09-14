import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import talib
from talib import abstract, func, stream

THREADS = 8
ROUNDS = 10


def _while_running(call, meanwhile):
    # The switch interval must be out of reach: a thread holding the GIL is then
    # never made to yield it, so `meanwhile` runs inside `call` only if `call`
    # releases it. With the default interval this passes on a build that holds it.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(100.0)
    try:
        worker = threading.Thread(target=call)
        worker.start()
        try:
            return meanwhile()
        finally:
            worker.join()
    finally:
        sys.setswitchinterval(interval)


def _closes():
    rng = np.random.default_rng(0)
    for size in (1 << 14, 1 << 16, 1 << 18, 1 << 20):
        yield 100 + np.cumsum(rng.standard_normal(size))


def _releases_the_gil(make_call):
    # A first call can import, and an import releases the GIL.
    make_call(np.linspace(1.0, 2.0, 1000))()
    for close in _closes():
        done = threading.Event()
        call = make_call(close)
        if _while_running(lambda: (call(), done.set()), lambda: not done.is_set()):
            return True
    return False


@pytest.mark.parametrize('make_call', [
    lambda close: lambda: func.HT_DCPHASE(close),
    lambda close: lambda: abstract.Function('HT_DCPHASE')(close),
], ids=['func', 'abstract'])
def test_indicator_call_releases_the_gil(make_call):
    assert _releases_the_gil(make_call)


@pytest.mark.parametrize('make_call', [
    lambda close: lambda: stream.HT_DCPHASE(close),
    lambda close: lambda: stream.HT_DCPHASE.open_and_fill(close),
], ids=['open', 'open_and_fill'])
def test_stream_open_releases_the_gil(make_call):
    assert _releases_the_gil(make_call)


def _calls(prices):
    o, h, lo, c = (prices[k] for k in ('open', 'high', 'low', 'close'))
    return [
        lambda: func.EMA(c, timeperiod=30),
        lambda: func.RSI(c),
        lambda: func.MACD(c),
        lambda: func.CDLDOJI(o, h, lo, c),
        lambda: func.ATR(h, lo, c),
        lambda: abstract.Function('BBANDS')(c),
    ]


def _flatten(result):
    if isinstance(result, (tuple, list)):
        return [np.asarray(r) for r in result]
    return [np.asarray(result)]


def _per_thread_prices(ford_2012, repeat):
    # Threads computing identical numbers cannot show one wrote into another's buffer.
    return [{k: np.tile(np.roll(v, 13 * t), repeat) for k, v in ford_2012.items()}
            for t in range(THREADS)]


def test_functions_are_correct_when_called_concurrently(ford_2012):
    calls = [_calls(prices) for prices in _per_thread_prices(ford_2012, 200)]
    expected = [[_flatten(call()) for call in mine] for mine in calls]
    start = threading.Barrier(THREADS)

    def worker(t):
        start.wait()
        n = len(calls[t])
        return t, [(i % n, _flatten(calls[t][i % n]())) for i in range(ROUNDS * n)]

    with ThreadPoolExecutor(THREADS) as pool:
        for t, results in pool.map(worker, range(THREADS)):
            for which, got in results:
                assert len(got) == len(expected[t][which])
                for g, e in zip(got, expected[t][which]):
                    assert_array_equal(g, e)


def test_settings_still_work_between_concurrent_calls(series):
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


def test_one_stream_handle_per_thread_is_correct_concurrently(ford_2012):
    warmup = 60
    closes = [prices['close'] for prices in _per_thread_prices(ford_2012, 20)]
    handles = [stream.MACD(close[:warmup]) for close in closes]
    start = threading.Barrier(THREADS)

    def worker(t):
        start.wait()
        return t, [handles[t].update(bar) for bar in closes[t][warmup:]]

    with ThreadPoolExecutor(THREADS) as pool:
        for t, got in pool.map(worker, range(THREADS)):
            assert_array_equal(np.array(got), np.column_stack(func.MACD(closes[t]))[warmup:])

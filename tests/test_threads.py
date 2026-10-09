import gc
import subprocess
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import talib
from talib import abstract, stream
from talib import _ta_lib


@pytest.mark.parametrize('length', [100, 40000])
def test_concurrent_batch_calls(length):
    def calculate(i):
        close = 100 + np.cumsum(np.random.default_rng(i).normal(size=length))
        high, low = close + 1, close - 1
        return (talib.RSI(close), talib.MACD(close),
                talib.CDLDOJI(close, high, low, close),
                abstract.Function('BBANDS')(close))

    expected = [calculate(i) for i in range(4)]
    start = threading.Barrier(4)

    def worker(i):
        start.wait()
        for _ in range(3):
            for got, want in zip(calculate(i), expected[i]):
                assert_array_equal(got, want)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, range(4)))


def test_shared_stream_updates_and_reads():
    handle = stream.EMA(np.full(100, 100.0), timeperiod=10)
    original = handle.out_range
    start = threading.Barrier(4)

    def worker(_):
        start.wait()
        for _ in range(200):
            assert handle.update(100.0) == 100.0
            assert handle.peek(100.0) == 100.0
            assert handle.value == 100.0
            clone = handle.copy()
            assert clone.value == 100.0
            assert clone.out_range.begidx == original.begidx

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, range(4)))
    assert handle.out_range.nbelement == original.nbelement + 800


def test_shared_stream_reinitialization():
    history = np.full(100, 100.0)
    handle = stream.EMA(history, timeperiod=10)
    start = threading.Barrier(2)

    def reopen():
        start.wait()
        for _ in range(100):
            handle.__init__(history, timeperiod=10)

    def update():
        start.wait()
        for _ in range(500):
            assert handle.update(100.0) == 100.0
            assert handle.copy().value == 100.0

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(reopen), pool.submit(update)]
        for future in futures:
            future.result()


@pytest.mark.parametrize('change', [
    lambda: talib.set_unstable_period('EMA', 0),
    lambda: talib.set_compatibility(0),
    lambda: _ta_lib._ta_set_candle_settings(0, 0, 10, 1.0),
    lambda: _ta_lib._ta_restore_candle_default_settings(0),
    _ta_lib._ta_shutdown,
])
def test_settings_and_shutdown_reject_live_streams(change):
    handle, _ = stream.EMA.open_and_fill(np.arange(100, dtype=float))
    clone = handle.copy()
    # Replacing a handle must neither double-count it nor forget it.
    handle.__init__(np.arange(100, dtype=float))
    with pytest.raises(RuntimeError):
        change()
    del handle
    gc.collect()
    with pytest.raises(RuntimeError):
        change()
    del clone
    gc.collect()
    change()
    _ta_lib._ta_initialize()


def test_failed_stream_open_does_not_block_settings():
    with pytest.raises(talib.InsufficientHistory):
        stream.EMA(np.array([1.0]))
    gc.collect()
    talib.set_unstable_period('EMA', 0)


def test_settings_exclusion_and_reader_cleanup():
    entered = threading.Event()
    release = threading.Event()

    def reader():
        try:
            with _ta_lib._TA_ReadLease():
                entered.set()
                assert release.wait(10)
                raise ValueError('exercise exception cleanup')
        except ValueError:
            pass

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(reader)
        try:
            assert entered.wait(10)
            with pytest.raises(RuntimeError):
                talib.set_unstable_period('EMA', 0)
            # Readers must remain concurrent while settings changes are excluded.
            assert talib.EMA(np.arange(100, dtype=float)).shape == (100,)
        finally:
            release.set()
        future.result()
    talib.set_unstable_period('EMA', 0)


def test_settings_change_applies_to_later_calls():
    values = np.arange(100, dtype=float)
    original = talib.get_unstable_period('EMA')
    try:
        talib.set_unstable_period('EMA', 0)
        before = talib.EMA(values)
        talib.set_unstable_period('EMA', 10)
        after = talib.EMA(values)
        assert np.isnan(after).sum() == np.isnan(before).sum() + 10
    finally:
        talib.set_unstable_period('EMA', original)


@pytest.mark.skipif(not sysconfig.get_config_var('Py_GIL_DISABLED'),
                    reason='requires a free-threaded interpreter')
def test_import_keeps_gil_disabled():
    subprocess.run([sys.executable, '-c',
                    'import sys; assert not sys._is_gil_enabled(); '
                    'import talib, talib.abstract, talib.stream; '
                    'assert not sys._is_gil_enabled()'], check=True)


def test_shutdown_and_reinitialize():
    subprocess.run([sys.executable, '-c', '''
import numpy as np
import talib
from talib import _ta_lib
_ta_lib._ta_shutdown()
try:
    talib.EMA(np.arange(100, dtype=float))
except RuntimeError:
    pass
else:
    raise AssertionError('a call after shutdown succeeded')
_ta_lib._ta_initialize()
assert talib.EMA(np.arange(100, dtype=float)).shape == (100,)
'''], check=True)

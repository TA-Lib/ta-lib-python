import ast
import copy
import os
import pickle
from functools import lru_cache

import numpy as np
import pytest

import talib
from talib import abstract, stream


def _flagged(flag):
    return {name for name in talib.__TA_FUNCTION_NAMES__
            if flag in abstract.Function(name).function_flags}


# The corpus is what the C library itself says streams, not the hand-maintained
# group dict: a function the flag knows and the dict does not would otherwise be
# silently absent from every parametrised test below.
FUNCTIONS = sorted(_flagged('Function has a streaming API'))
CANDLESTICKS = _flagged('Output is a candlestick')
NAN_BARS = 5

with open(os.path.join(os.path.dirname(talib.__file__), 'stream.pyi')) as _stub:
    STUB = _stub.read()


@lru_cache(maxsize=None)
def _input_names(name):
    return abstract.Function(name).info['input_names']


def series(name, data):
    """The arrays talib.<name> and stream.<name> take, in order."""
    args = []
    for role, price in _input_names(name).items():
        if isinstance(price, list):
            args.extend(data[p] for p in price)
        else:
            args.append(data[price if price in data else role])
    return args


def lookback(name):
    # Not cached: the unstable period moves it.
    return abstract.Function(name).lookback


def outputs(result):
    return result if isinstance(result, tuple) else (result,)


def paired(got, want):
    """zip() stops at the shorter side; a handle short of an output would
    otherwise compare nothing."""
    assert len(got) == len(want), (len(got), len(want))
    return zip(got, want)


def batch(name, data):
    return [np.asarray(a) for a in outputs(getattr(talib, name)(*series(name, data)))]


def opened(name, args, warmup):
    """A handle over the first `warmup` bars, or over as few more as its seeding
    needs: lookback + 1 is a floor, not a promise."""
    for size in range(warmup, warmup + 4):
        try:
            return size, getattr(stream, name)(*[a[:size] for a in args])
        except talib.InsufficientHistory:
            continue
    pytest.fail('%s did not open on %d bars' % (name, warmup + 3))


def carried(name, data, warmup):
    """Open on a prefix, then update through the rest, against the batch series."""
    args = series(name, data)
    expected = batch(name, data)
    warmup, handle = opened(name, args, warmup)
    for got, want in paired(outputs(handle.value), expected):
        assert repr(got) == repr(want[warmup - 1].item()), name
    updated = 0
    for bar in range(warmup, len(data['close'])):
        produced = handle.update(*[a[bar] for a in args])
        for got, want in paired(outputs(produced), expected):
            assert repr(got) == repr(want[bar].item()), (name, bar)
        updated += 1
    assert updated == len(data['close']) - warmup
    return handle


def identical(got, want):
    """Bit-exact for arrays: NaN equals NaN, and -0.0 does not equal 0.0."""
    got, want = np.asarray(got).astype(float), np.asarray(want).astype(float)
    if not np.array_equal(got, want, equal_nan=True):
        return False
    real = ~np.isnan(got)
    return np.array_equal(np.signbit(got[real]), np.signbit(want[real]))


def parameters(doc):
    """Names and defaults from the signature Cython embeds on the first line."""
    head = doc.split('\n')[0]
    body = head[head.index('(') + 1:head.rindex(')')]
    named = []
    for argument in body.split(','):
        argument = argument.strip()
        if argument and argument != 'self':
            name, _, default = argument.partition('=')
            named.append((name.split()[-1], default))
    return named


def varies(out, warmup):
    out = np.asarray(out[warmup:], dtype=float)
    return len(np.unique(out[~np.isnan(out)])) > 1


@pytest.fixture(scope='session')
def datasets(ford_2012):
    """The real prices, and the same prices mapped into [-1, 1] so that the
    domain-limited transforms produce something other than NaN."""
    ford = {k: np.asarray(v, dtype=float) for k, v in ford_2012.items()}
    prices = np.concatenate([ford[k] for k in ('open', 'high', 'low', 'close')])
    lo, hi = prices.min(), prices.max()
    volume = ford['volume']
    scaled = {k: (v - lo) / (hi - lo) * 2 - 1 for k, v in ford.items() if k != 'volume'}
    scaled['volume'] = (volume - volume.min()) / (volume.max() - volume.min()) * 0.9 + 0.1
    # And one with degenerate bars, which real prices never have: a bar with no
    # range at all and a bar with no trades are the inputs that reach the
    # zero-denominator guards.
    flat = {k: v.copy() for k, v in ford.items()}
    for k in ('open', 'high', 'low'):
        flat[k][::7] = flat['close'][::7]
    flat['volume'][::11] = 0.0
    for data in (ford, scaled, flat):
        data['periods'] = np.linspace(2, 20, len(data['close']))
    return ford, scaled, flat


@pytest.fixture(scope='session')
def nan_prefixed(datasets):
    """The real prices behind NAN_BARS bars of NaN -- what a shifted or resampled
    column looks like. The Function API opens past those bars; so does a handle."""
    return {k: np.concatenate([np.full(NAN_BARS, np.nan), v])
            for k, v in datasets[0].items()}


@pytest.mark.parametrize('name', FUNCTIONS)
def test_matches_batch(name, datasets):
    """Two warm-up prefixes: an open-side seeding bug hides at a single one."""
    for data in datasets:
        for warmup in (lookback(name) + 1, len(data['close']) // 2):
            carried(name, data, max(warmup, lookback(name) + 1))


@pytest.mark.parametrize('name', FUNCTIONS)
def test_leading_nan_bars_are_not_history(name, nan_prefixed):
    """They are not counted as warm-up, and an index output still comes back in
    the caller's coordinates, exactly as the Function API reports it."""
    handle = carried(name, nan_prefixed, NAN_BARS + lookback(name) + 1)
    bars = len(nan_prefixed['close']) - NAN_BARS - lookback(name)
    assert handle.out_range == (NAN_BARS + lookback(name), bars)

    args = series(name, nan_prefixed)
    filled = outputs(getattr(stream, name).open_and_fill(*args)[1])
    for got, want in paired(filled, batch(name, nan_prefixed)):
        assert identical(got, want)

    if lookback(name):
        with pytest.raises(talib.InsufficientHistory):
            getattr(stream, name)(*[a[:NAN_BARS + lookback(name)] for a in args])


def test_the_data_discriminates(datasets):
    """A stream-equals-batch comparison proves nothing about an output whose
    batch series never varies. Only candlesticks are flat here, and data rich
    enough to fire every pattern is the C library's own gate."""
    flat, checked = set(), 0
    for name in FUNCTIONS:
        warmup = lookback(name) + 1
        seen = [varies(out, warmup) for out in batch(name, datasets[0])]
        for data in datasets[1:]:
            seen = [was or varies(out, warmup)
                    for was, out in zip(seen, batch(name, data))]
        checked += 1
        if not all(seen):
            flat.add(name)
    assert checked == len(FUNCTIONS)
    assert flat <= CANDLESTICKS


@pytest.mark.parametrize('name', FUNCTIONS)
def test_open_and_fill_matches_batch(name, datasets):
    data = datasets[0]
    expected = batch(name, data)
    handle, filled = getattr(stream, name).open_and_fill(*series(name, data))
    for got, want in paired(outputs(filled), expected):
        assert got.dtype == want.dtype
        assert identical(got, want)
    for got, want in paired(outputs(handle.value), expected):
        assert repr(got) == repr(want[-1].item())
    assert handle.out_range == (lookback(name), len(data['close']) - lookback(name))
    if len(expected) > 1:
        assert handle.value._fields == tuple(abstract.Function(name).output_names)


@pytest.mark.parametrize('name', FUNCTIONS)
def test_peek_and_copy_commit_nothing(name, datasets):
    data = datasets[0]
    args = series(name, data)
    handle, _ = getattr(stream, name).open_and_fill(*args)
    fork = handle.copy()
    value, span = repr(handle.value), handle.out_range
    bar = [a[-2] for a in args]     # a bar other than the one it just consumed

    provisional = repr(handle.peek(*bar))
    assert repr(handle.peek(*bar)) == provisional
    assert (repr(handle.value), handle.out_range) == (value, span)
    assert repr(handle.update(*bar)) == provisional
    assert handle.out_range == (span.begidx, span.nbelement + 1)
    assert (repr(fork.value), fork.out_range) == (value, span)


@pytest.mark.parametrize('name', FUNCTIONS)
def test_advance_counts_a_skipped_bar(name, datasets):
    handle, _ = getattr(stream, name).open_and_fill(*series(name, datasets[0]))
    value, span = repr(handle.value), handle.out_range
    handle.advance()
    assert handle.out_range == (span.begidx, span.nbelement + 1)
    assert repr(handle.value) == value


@pytest.mark.parametrize('name', FUNCTIONS)
def test_the_open_boundary_is_sharp(name, datasets):
    """Opening needs at least lookback + 1 bars, and more where the seeding does,
    so what holds everywhere is that every shorter history is refused."""
    args = series(name, datasets[0])
    need = lookback(name) + 1
    for size in range(need + 3):
        try:
            getattr(stream, name)(*[a[:size] for a in args])
        except talib.InsufficientHistory:
            continue
        assert size >= need
        return
    pytest.fail('%s never opened' % name)


@pytest.mark.parametrize('name', FUNCTIONS)
def test_takes_the_arguments_the_function_takes(name):
    """Including the bar arguments' names and order, which no value comparison
    reaches: swapping high and low leaves MEDPRICE bit-identical."""
    handle, function = getattr(stream, name), getattr(talib, name)
    assert parameters(handle.__doc__) == parameters(function.__doc__)
    assert parameters(handle.open_and_fill.__doc__) == parameters(function.__doc__)
    bars = parameters(handle.update.__doc__)
    assert bars == parameters(handle.peek.__doc__)
    assert bars == parameters(function.__doc__)[:len(bars)]


@pytest.mark.parametrize('name', FUNCTIONS)
def test_the_stub_matches_the_handle(name):
    """talib/stream.pyi is generated from the same headers, but nothing in a
    plain test run regenerates it, so compare it to what actually shipped."""
    declared = {node.name: node for node in ast.parse(STUB).body
                if isinstance(node, ast.ClassDef)}
    for method, signature in (('__init__', getattr(stream, name).__doc__),
                              ('update', getattr(stream, name).update.__doc__)):
        stub = [argument.arg
                for node in declared[name].body
                if isinstance(node, ast.FunctionDef) and node.name == method
                for argument in node.args.args if argument.arg != 'self']
        assert stub == [n for n, _ in parameters(signature)], (name, method)


def test_the_corpus_is_what_the_library_says_streams():
    """FUNCTIONS is derived from TA_FUNC_FLG_STREAM. Pin it against the other two
    lists of the same thing, so a function added to one and not the others fails
    here rather than quietly dropping out of the tests above."""
    assert FUNCTIONS
    assert set(FUNCTIONS) == set(talib.__TA_FUNCTION_NAMES__)
    assert set(FUNCTIONS) == set(talib.get_functions())
    assert all(isinstance(getattr(stream, name), type) for name in FUNCTIONS)


def test_multi_output_is_a_named_tuple(datasets):
    handle = stream.MACD(datasets[0]['close'])
    macd, macdsignal, macdhist = handle.value
    assert (handle.value.macd, handle.value.macdsignal, handle.value.macdhist) \
        == (macd, macdsignal, macdhist)


def test_single_output_is_a_scalar(datasets):
    assert isinstance(stream.SMA(datasets[0]['close']).value, float)
    assert isinstance(stream.CDLDOJI(*series('CDLDOJI', datasets[0])).value, int)


def test_a_handle_takes_what_the_function_takes(datasets):
    close = datasets[0]['close']
    for rejected in (list(close), close.astype(int), close.reshape(-1, 1)):
        with pytest.raises((TypeError, Exception)):
            stream.SMA(rejected)


def test_a_result_pickles(datasets):
    import pandas as pd
    import polars as pl
    handle = stream.MACD(datasets[0]['close'])
    assert pickle.loads(pickle.dumps(handle.value)) == handle.value
    assert pickle.loads(pickle.dumps(handle.out_range)) == handle.out_range
    with pytest.raises(Exception, match='Cannot mix polars and pandas'):
        stream.MAVP.open_and_fill(pd.Series(datasets[0]['close']),
                                  pl.Series(datasets[0]['periods']))


def test_a_handle_does_not_pickle(datasets):
    handle = stream.SMA(datasets[0]['close'])
    with pytest.raises(TypeError):
        pickle.dumps(handle)
    assert copy.copy(handle).value == handle.value
    assert copy.deepcopy(handle).value == handle.value


def test_update_rejects_a_non_finite_bar(datasets):
    close = datasets[0]['close']
    handle = stream.SMA(close)
    span = handle.out_range
    for bar in (np.nan, np.inf, -np.inf):
        with pytest.raises(Exception, match='Bad Parameter'):
            handle.update(bar)
    assert handle.value == talib.SMA(close)[-1]
    assert handle.out_range == span


def test_insufficient_history_says_what_it_counted(datasets):
    with pytest.raises(talib.InsufficientHistory,
                       match='29 bars of history, at least 30 needed'):
        stream.SMA(datasets[0]['close'][:29], timeperiod=30)


def test_a_handle_takes_the_arguments_the_function_takes(datasets):
    close = datasets[0]['close']
    assert stream.SMA(close, timeperiod=10).value == talib.SMA(close, timeperiod=10)[-1]
    assert stream.SMA(close, 10).value == talib.SMA(close, 10)[-1]

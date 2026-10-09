"""Benchmark a fixed total number of RSI calls on either Python build.

PYTHONPATH=. python tools/threads_talib.py 4005 10000 --repeat 5
PYTHONPATH=. python tools/threads_talib.py 100000 2000 --api func
Without a length argument, calls alternate between 999 and 4005 bars.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import sys
import threading
import time

import numpy as np
import pandas as pd
import talib
from talib.abstract import RSI


def benchmark(inputs, calls, workers, repeat, api):
    samples = []
    with ThreadPoolExecutor(workers) as pool:
        for _ in range(repeat):
            ready = threading.Barrier(workers + 1)

            def worker(index):
                # Private inputs and thread-local initialization are untimed.
                frames = [pd.DataFrame(values.copy(), columns=[
                    'open', 'high', 'low', 'close', 'volume']) for values in inputs]
                arrays = [np.ascontiguousarray(frame['close'].to_numpy()) for frame in frames]
                if api == 'abstract':
                    for frame in frames:
                        frame['RSI'] = RSI(frame)
                else:
                    for array in arrays:
                        talib.RSI(array)
                ready.wait()
                first = calls * index // workers
                last = calls * (index + 1) // workers
                for i in range(first, last):
                    which = i % len(inputs)
                    if api == 'abstract':
                        frames[which]['RSI'] = RSI(frames[which])
                    else:
                        talib.RSI(arrays[which])
                return last - first

            futures = [pool.submit(worker, i) for i in range(workers)]
            # Wait until preparation is complete; surface failures before the barrier.
            while ready.n_waiting != workers:
                if any(f.done() for f in futures):
                    ready.abort()
                    for f in futures:
                        if f.done():
                            f.result()
                time.sleep(0.001)
            started = time.perf_counter()
            ready.wait()
            completed = sum(f.result() for f in futures)
            samples.append(time.perf_counter() - started)
            assert completed == calls
    median = statistics.median(samples)
    return {'threads': workers, 'seconds': median, 'calls_per_second': calls / median,
            'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('length', type=int, nargs='?')
    parser.add_argument('calls', type=int, nargs='?', default=10000)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--api', choices=['abstract', 'func'], default='abstract')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    sizes = [args.length] if args.length is not None else [999, 4005]
    if (min(sizes) < 30 or min(args.threads) < 1 or args.calls < max(args.threads)
            or args.repeat < 1):
        parser.error('use >=30 bars, positive threads/repeats, and calls >= threads')
    rng = np.random.default_rng(0)
    inputs = [rng.random((size, 5)) for size in sizes]
    results = [benchmark(inputs, args.calls, workers, args.repeat, args.api)
               for workers in args.threads]
    baseline = next((r for r in results if r['threads'] == 1), results[0])
    for result in results:
        result['speedup'] = baseline['seconds'] / result['seconds']
    report = {'python': sys.version.split()[0],
              'gil_enabled': sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else True,
              'api': args.api, 'lengths': sizes, 'calls': args.calls,
              'repeat': args.repeat, 'results': results}
    if args.json:
        print(json.dumps(report))
    else:
        print(f"Python {report['python']}, GIL enabled: {report['gil_enabled']}, "
              f"API: {args.api}, bars: {sizes}, total calls: {args.calls}")
        print('threads  median seconds  calls/second  speedup')
        for result in results:
            print(f"{result['threads']:7d}  {result['seconds']:14.6f}  "
                  f"{result['calls_per_second']:12.0f}  {result['speedup']:7.2f}x")


if __name__ == '__main__':
    main()

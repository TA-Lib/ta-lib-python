.PHONY: build build-bundled build-tsan test-tsan

build:
	python3 -m pip install --use-pep517 -e .

build-bundled:
	TA_FORCE_BUNDLED=1 python3 -m pip install --use-pep517 -e .


build-tsan: clean
	TA_TSAN=1 python3 -m pip install --use-pep517 -e .

TSAN_OPTIONS ?= halt_on_error=1 second_deadlock_stack=1 handle_segv=0
test-tsan: build-tsan
	TSAN_OPTIONS="$(TSAN_OPTIONS)" pytest tests/test_threads.py


install:
	python3 -m pip install --use-pep517 .

talib/_func.pxi: tools/generate_func.py
	python3 tools/generate_func.py > talib/_func.pxi

talib/_stream.pxi: tools/generate_stream.py
	python3 tools/generate_stream.py > talib/_stream.pxi

talib/stream.pyi: tools/generate_stream.py
	python3 tools/generate_stream.py --stub > talib/stream.pyi

talib/abstract.pyi: tools/generate_abstract_stub.py
	python3 tools/generate_abstract_stub.py > talib/abstract.pyi

generate: talib/_func.pxi talib/_stream.pxi talib/stream.pyi talib/abstract.pyi

cython:
	cython talib/_ta_lib.pyx

annotate:
	cython -a talib/_ta_lib.pyx

clean:
	rm -rf build talib/_ta_lib.so talib/*.pyc

perf:
	python3 tools/perf_talib.py

test: build
	pytest tests/

sdist:
	python3 -m build --sdist

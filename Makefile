.PHONY: build

build:
	python3 -m pip install --use-pep517 -e .

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

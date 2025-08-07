.PHONY: build

build:
	python3 -m pip install --use-pep517 -e .

install:
	python3 -m pip install --use-pep517 .

talib/_func.pxi: tools/generate_func.py
	python3 tools/generate_func.py > talib/_func.pxi

talib/_stream.pxi: tools/generate_stream.py
	python3 tools/generate_stream.py > talib/_stream.pxi

generate: talib/_func.pxi talib/_stream.pxi

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

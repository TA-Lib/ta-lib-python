.PHONY: build

build:
	python3 setup.py build_ext --inplace

build_static:
	TA_LINK_STATIC=1 TA_INCLUDE_PATH=c-ta-lib/include TA_LIBRARY_PATH=c-ta-lib/lib python3 setup.py build_ext --inplace

install:
	python3 setup.py install

install_static:
	TA_LINK_STATIC=1 TA_INCLUDE_PATH=c-ta-lib/include TA_LIBRARY_PATH=c-ta-lib/lib python3 setup.py install

talib/_func.pxi: tools/generate_func.py
	python3 tools/generate_func.py > talib/_func.pxi

talib/_stream.pxi: tools/generate_stream.py
	python3 tools/generate_stream.py > talib/_stream.pxi

generate: talib/_func.pxi talib/_stream.pxi

cython:
	cython --directive emit_code_comments=False talib/_ta_lib.pyx

clean:
	rm -rf build talib/_ta_lib.so talib/*.pyc

perf:
	python3 tools/perf_talib.py

test: build
	LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} pytest

sdist:
	python3 setup.py sdist --formats=gztar,zip

sdist_static:
	TA_LINK_STATIC=1 TA_INCLUDE_PATH=c-ta-lib/include TA_LIBRARY_PATH=c-ta-lib/lib python3 setup.py sdist --formats=gztar,zip

talib_static:
	tools/build_talib_from_source.bash ${PWD}/c-ta-lib

wheel_static:
	TA_LINK_STATIC=1 TA_INCLUDE_PATH=c-ta-lib/include TA_LIBRARY_PATH=c-ta-lib/lib python3 setup.py bdist_wheel

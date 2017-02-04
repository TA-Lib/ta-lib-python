.PHONY: build

build:
	python setup.py build_ext --inplace

install:
	python setup.py install

talib/_func.pxi: tools/generate_func.py
	python tools/generate_func.py > talib/_func.pxi

talib/_stream.pxi: tools/generate_stream.py
	python tools/generate_stream.py > talib/_stream.pxi

generate: talib/_func.pxi talib/_stream.pxi

clean:
	rm -rf build talib/func*.so talib/abstract*.so talib/common*.so talib/stream*.so talib/*.pyc

perf:
	python tools/perf_talib.py

test:
	LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} nosetests

sdist:
	python setup.py sdist --formats=gztar,zip

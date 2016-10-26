build:
	python setup.py build_ext --inplace

install:
	python setup.py install

generate:
	python tools/generate.py > talib/func.pyx
	python tools/generate_stream.py > talib/stream.pyx

clean:
	rm -rf build talib/func*.so talib/abstract*.so talib/common*.so talib/stream*.so talib/*.pyc

perf:
	python tools/perf_talib.py

test:
	LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} nosetests

sdist:
	python setup.py sdist --formats=gztar,zip

VERSION?=2.7

build:
	python$(VERSION) setup.py build_ext --inplace

install:
	python$(VERSION) setup.py install

generate:
	python$(VERSION) tools/generate.py > talib/func.pyx

clean:
	rm -rf build talib/func*.so talib/abstract*.so talib/common*.so talib/*.pyc

perf:
	python$(VERSION) tools/perf_talib.py

test:
	LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} nosetests-$(VERSION)

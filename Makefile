build:
	python2.7 setup.py build_ext --inplace

generate:
	python2.7 generate.py > talib.pyx

clean:
	rm -rf build talib.so talib.c

test:
	LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} nosetests


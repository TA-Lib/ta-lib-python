
build:
	python setup.py build_ext --inplace

clean:
	rm -rf build talib.so talib.c


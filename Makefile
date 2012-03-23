build:
	python setup.py build_ext --inplace

generate:
	python generate.py > talib.pyx

clean:
	rm -rf build talib.so talib.c


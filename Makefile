build:
	python2.7 setup.py build_ext --inplace

generate:
	python2.7 tools/generate.py > talib/func.pyx

clean:
	rm -rf build talib/func.so talib/abstract.so talib/common_c.so

perf:
	python2.7 tools/perf_talib.py

test:
	LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} python -m unittest discover -v -s tests -p "*_test.py"

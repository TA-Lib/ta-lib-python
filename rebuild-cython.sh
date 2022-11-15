#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  find -E src/talib tests -regex '.*\.(c|so)' -exec rm {} +
else
  find src/talib tests -regex '.*\.\(c\|so\)' -exec rm {} +
fi
python setup.py build_ext --inplace

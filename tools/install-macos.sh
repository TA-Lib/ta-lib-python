#!/bin/bash

brew install ta-lib

export TA_INCLUDE_PATH="$(brew --prefix ta-lib)/include"
export TA_LIBRARY_PATH="$(brew --prefix ta-lib)/lib"

pip3 install ta-lib

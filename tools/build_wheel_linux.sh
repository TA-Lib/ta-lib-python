#!/bin/bash

# TALIB_C_VER=0.6.2
# TALIB_PY_VER=0.5.2
CMAKE_GENERATOR="Unix Makefiles"
CMAKE_BUILD_TYPE=Release
CMAKE_CONFIGURATION_TYPES=Release

# Download TA-Lib C Library
curl -L -o talib-c.zip https://github.com/TA-Lib/ta-lib/archive/refs/tags/v${TALIB_C_VER}.zip
if [ $? -ne 0 ]; then
    echo "Failed to download TA-Lib C library"
    exit 1
fi

# Download TA-Lib Python
curl -L -o talib-python.zip https://github.com/TA-Lib/ta-lib-python/archive/refs/tags/TA_Lib-${TALIB_PY_VER}.zip
if [ $? -ne 0 ]; then
    echo "Failed to download TA-Lib Python library"
    exit 1
fi

# Unzip TA-Lib C
unzip -q talib-c.zip
if [ $? -ne 0 ]; then
    echo "Failed to extract TA-Lib C library"
    exit 1
fi

# Unzip TA-Lib Python
unzip -q talib-python.zip -d ta-lib-python
if [ $? -ne 0 ]; then
    echo "Failed to extract TA-Lib Python library"
    exit 1
fi

mv ta-lib-python/ta-lib-python-TA_Lib-${TALIB_PY_VER}/* .
rm -r ta-lib-python

# cd to TA-Lib C
cd ta-lib-${TALIB_C_VER}

# Copy TA-Lib C headers to TA-Lib Python
mkdir -p include/ta-lib/
cp include/*.h include/ta-lib/

# Create build directory
mkdir -p _build
cd _build

# Use CMake to configure the build
cmake -G "$CMAKE_GENERATOR" -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DCMAKE_INSTALL_PREFIX=../../ta-lib-install ..
if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

# Compile TA-Lib
make
if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

make install
if [ $? -ne 0 ]; then
    echo "Install failed"
    exit 1
fi

echo "TA-Lib build completed successfully!"

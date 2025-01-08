:: Download and build TA-Lib 
@echo on

:: set TALIB_C_VER=0.6.2
:: set TALIB_PY_VER=0.5.2

set CMAKE_GENERATOR=NMake Makefiles
set CMAKE_BUILD_TYPE=Release
set CMAKE_CONFIGURATION_TYPES=Release

curl -L -o talib-c.zip https://github.com/TA-Lib/ta-lib/archive/refs/tags/v%TALIB_C_VER%.zip
if errorlevel 1 exit /B 1

curl -L -o talib-python.zip https://github.com/TA-Lib/ta-lib-python/archive/refs/tags/TA_Lib-%TALIB_PY_VER%.zip
if errorlevel 1 exit /B 1

tar -xzvf talib-c.zip
if errorlevel 1 exit /B 1

tar -xf talib-python.zip --strip-components=1
if errorlevel 1 exit /B 1

:: git apply --verbose --binary talib.diff
:: if errorlevel 1 exit /B 1

:: set MSBUILDTREATHIGHERTOOLSVERSIONASCURRENT

setlocal
cd ta-lib-%TALIB_C_VER%

mkdir  include\ta-lib
copy /Y include\*.* include\ta-lib

md _build
cd _build

cmake.exe ..
if errorlevel 1 exit /B 1

nmake.exe /nologo all
if errorlevel 1 exit /B 1

copy /Y /B ta-lib-static.lib ta-lib.lib

endlocal

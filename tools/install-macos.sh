
brew install ta-lib

arch="$(uname -m)"
if [ "$arch" = "arm64" ]; then
  export TA_INCLUDE_PATH="$(brew --prefix ta-lib)/include"
  export TA_LIBRARY_PATH="$(brew --prefix ta-lib)/lib"
fi

pip3 install ta-lib

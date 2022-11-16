#!/bin/bash

curl -L -o ta-lib-0.4.0-src.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz &&
  tar xvfz ta-lib-0.4.0-src.tar.gz &&
  cd ta-lib &&
  ./configure &&
  make &&
  make install &&
  ldconfig &&
  yum install -y libtiff-devel libjpeg-devel openjpeg2-devel zlib-devel \
    freetype-devel lcms2-devel libwebp-devel tcl-devel tk-devel \
    harfbuzz-devel fribidi-devel libraqm-devel libimagequant-devel libxcb-devel

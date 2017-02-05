#!/bin/bash
set -e

if [[ -z $1 ]]; then
  echo "Usage: $0 deps_dir"
  exit 1
fi

DEPS_DIR=$1

TA_LIB_TGZ="ta-lib-0.4.0-src.tar.gz"
TA_LIB_URL="http://prdownloads.sourceforge.net/ta-lib/$TA_LIB_TGZ"

if [[ -d $DEPS_DIR/lib ]]; then
  echo "Already built"
  exit 0
fi
mkdir -p $DEPS_DIR/tmp
wget -O "$DEPS_DIR/tmp/$TA_LIB_TGZ" $TA_LIB_URL
pushd $DEPS_DIR/tmp
tar -zxvf $TA_LIB_TGZ
popd
pushd $DEPS_DIR/tmp/ta-lib
./configure --prefix=$DEPS_DIR
make install
popd

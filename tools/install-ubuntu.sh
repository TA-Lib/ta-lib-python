apt update
yes | apt-get install gcc build-essential python3-distutils python3-dev python3-pip

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

tar -zxvf ta-lib-0.4.0-src.tar.gz

rm ta-lib-0.4.0-src.tar.gz

cd ta-lib

if [[ `id -un` == "root" ]]
    then
        ./configure
        make
        make install
    else
        ./configure --prefix=/usr
        make
        make install
fi


yes | pip3 install setuptools
yes | pip3 install Ta-LIB

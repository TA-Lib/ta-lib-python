#Intiliaze wizards
echo "----------------------------------------------------"
echo "| Please wait while the wizard configures ta-lib py|"
echo "| C                                                |"
echo "|  (\.   \      ,/)                                |"
echo "|   \(   |\     )/                                 |"
echo "|   //\  | \   /\\                                 |"
echo "|  (/ /\_#oo#_/\ \)                                |"
echo "|   \/\  ####  /\/                                 |"
echo "|        '##'                                      |"
echo "----------------------------------------------------"

apt update
yes | sudo apt-get install gcc build-essential python3-distutils python3-dev
sudo pip3 install wheel		
sudo pip3 install regex	
sudo wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

sudo tar -zxvf ta-lib-0.4.0-src.tar.gz

sudo rm ta-lib-0.4.0-src.tar.gz

cd ta-lib

if [[ `id -un` == "root" ]]
    then
        ./configure
        sudo make
        sudo make install
    else
        ./configure --prefix=/usr
        sudo make
        sudo make install
fi

sudo bash -c "/usr/local/lib >> /etc/ld.so.conf"
sudo /sbin/ldconfig

yes | sudo -H pip3 install Ta-LIB

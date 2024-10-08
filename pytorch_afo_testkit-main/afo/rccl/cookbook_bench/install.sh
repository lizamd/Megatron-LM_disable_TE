#!/bin/bash

# install cookbook
pip install -r requirements.txt

# install rccl-test
cd rccl-tests
./install.sh
cd ..

# install rocmProfileData
apt-get update
apt-get install sqlite3 libsqlite3-dev
apt-get install libfmt-dev
git clone -b jpvillam/context_manager https://github.com/ROCm/rocmProfileData.git && cd rocmProfileData
make
make install 
cd ../

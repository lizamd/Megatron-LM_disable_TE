#!/bin/bash

# Install deps
apt-get update -y 
apt-get install sqlite3 libsqlite3-dev libfmt-dev -y

#Build RPD
git clone -b jpvillam/context_manager https://github.com/ROCm/rocmProfileData.git && cd rocmProfileData
make
make install 
cd ../


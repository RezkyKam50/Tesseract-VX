source ./scripts/setup/cuda_toolkit.sh
source ./scripts/setup/gcc_switcher.sh
 
aria2c -s 16 -x 16 https://www.python.org/ftp/python/3.14.0/Python-3.14.0.tar.xz
tar xvf Python-3.14.0.tar.xz
cd Python-3.14.0

./configure --prefix=/usr/local/python-3.14 \
            --enable-optimizations \
            --with-lto

make -j$(nproc --all)
sudo make altinstall

/usr/local/python-3.14/bin/python3.14 -m venv .venv
source .venv/bin/activate


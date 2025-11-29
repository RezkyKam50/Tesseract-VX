#!/bin/bash
set -e

# optional setup for ffmpeg with nvcuvid/nvenc support

echo "depending on the distro, if you have ffmpeg installed, check if its the non-patented version or else execute this script for cuda build and remove the non-patented version"
echo "To use this version of ffmpeg: "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH""

source ./scripts/cuda_toolkit.sh
source ./scripts/gcc_switcher.sh

cd thirdparty
cd nv-codec-headers && sudo make install 
cd ..
cd ffmpeg

./configure --enable-nonfree --enable-cuda-nvcc --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared

make -j$(nproc --all)

echo "WARN: this will install ffmpeg system-wide"
sudo make install


        

        

        
        

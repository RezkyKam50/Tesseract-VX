# CuDNN support
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.14.0.64_cuda13-archive.tar.xz
tar -xf cudnn-linux-x86_64-*-archive.tar.xz

sudo cp ./cudnn-linux-x86_64-*-archive/include/cudnn*.h /usr/local/cuda/include/
sudo cp ./cudnn-linux-x86_64-*-archive/lib/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

rm -rf ./cudnn-linux-x86_64-*-archive
rm -rf ./cudnn-linux-x86_64-*-archive.tar.xz
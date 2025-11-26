#!/bin/bash
set -e

# optional setup for NVIDIA GPU Codec SDK (NVENC/NVDEC)

if [ ! -d "./Video_Codec_SDK_*" ] && [ -z "$(find . -maxdepth 1 -type d -name 'Video_Codec_SDK_*' -print -quit)" ]; then
    echo "FATAL: SDK directory not found on $PWD"
    echo "Download on https://developer.nvidia.com/nvidia-video-codec-sdk/download"
    exit 1
fi

echo "Copying GPU Codec..."
sudo cp ./Video_Codec_SDK_*/Interface/nvEncodeAPI.h /usr/local/include/
sudo cp ./Video_Codec_SDK_*/Interface/* /usr/local/include/
sudo cp ./Video_Codec_SDK_*/Lib/linux/stubs/x86_64/libnvcuvid.so /usr/local/lib/
sudo cp ./Video_Codec_SDK_*/Lib/linux/stubs/x86_64/libnvidia-encode.so /usr/local/lib/

sudo ln -sf /usr/local/include/nvcuvid.h /usr/local/cuda-13.0/include/nvcuvid.h
sudo ln -sf /usr/local/include/nvEncodeAPI.h /usr/local/cuda-13.0/include/nvEncodeAPI.h
sudo ln -sf /usr/local/include/cuviddec.h /usr/local/cuda-13.0/include/cuviddec.h

sudo ldconfig
echo "Linking GPU Codec done..."
 
ls -la /usr/local/cuda-13.0/include/nvcuvid.h
ls -la /usr/local/cuda-13.0/include/nvEncodeAPI.h
ls -la /usr/local/cuda-13.0/include/cuviddec.h

ldconfig -p | grep nvcuvid
ldconfig -p | grep nvidia-encode
#!/bin/bash
set -e
 
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        OS_LIKE=$ID_LIKE
    else
        echo "Cannot detect OS type. Defaulting to /usr/local"
        OS="unknown"
    fi
}

setup_paths() {
    case $OS in
        fedora|rhel|centos)
            CUDA_PATH="/usr/local/cuda-13.0"
            INCLUDE_PATH="/usr/local/include"
            LIB_PATH="/usr/local/lib"
            ;;
        arch|manjaro)
            CUDA_PATH="/opt/cuda"
            INCLUDE_PATH="/usr/include"
            LIB_PATH="/usr/lib"
            ;;
        debian|ubuntu|linuxmint)
            CUDA_PATH="/usr/local/cuda-13.0"
            INCLUDE_PATH="/usr/local/include"
            LIB_PATH="/usr/local/lib"
            ;;
        *)
            CUDA_PATH="/usr/local/cuda-13.0"
            INCLUDE_PATH="/usr/local/include"
            LIB_PATH="/usr/local/lib"
            echo "Warning: Unknown OS detected. Using default paths."
            ;;
    esac
    
    echo "Detected OS: $OS"
    echo "Using CUDA path: $CUDA_PATH"
    echo "Using include path: $INCLUDE_PATH"
    echo "Using lib path: $LIB_PATH"
}
 
command_exists() {
    command -v "$1" >/dev/null 2>&1
}
 
ensure_dir() {
    if [ ! -d "$1" ]; then
        echo "Creating directory: $1"
        sudo mkdir -p "$1"
    fi
}
 
detect_os
setup_paths
 
ensure_dir "$INCLUDE_PATH"
ensure_dir "$LIB_PATH"
ensure_dir "$CUDA_PATH/include"
 
for zipfile in *.zip; do
    if [ -f "$zipfile" ]; then
        echo "Extracting $zipfile..."
        unzip -q "$zipfile"
    fi
done

echo "Copying GPU Codec..."
 
sudo cp ./Video_Codec_SDK_*/Interface/nvEncodeAPI.h "$INCLUDE_PATH/"
sudo cp ./Video_Codec_SDK_*/Interface/* "$INCLUDE_PATH/"
 
sudo cp ./Video_Codec_SDK_*/Lib/linux/stubs/x86_64/libnvcuvid.so "$LIB_PATH/"
sudo cp ./Video_Codec_SDK_*/Lib/linux/stubs/x86_64/libnvidia-encode.so "$LIB_PATH/"
 
echo "Creating symlinks..."
sudo ln -sf "$INCLUDE_PATH/nvcuvid.h" "$CUDA_PATH/include/nvcuvid.h"
sudo ln -sf "$INCLUDE_PATH/nvEncodeAPI.h" "$CUDA_PATH/include/nvEncodeAPI.h"
sudo ln -sf "$INCLUDE_PATH/cuviddec.h" "$CUDA_PATH/include/cuviddec.h"
 
echo "Updating library cache..."
if command_exists ldconfig; then
    sudo ldconfig
else
    echo "ldconfig not found, skipping library cache update"
fi

echo "Linking GPU Codec done..."
 
echo "Verifying installation..."
ls -la "$CUDA_PATH/include/nvcuvid.h" 2>/dev/null || echo "nvcuvid.h not found"
ls -la "$CUDA_PATH/include/nvEncodeAPI.h" 2>/dev/null || echo "nvEncodeAPI.h not found"
ls -la "$CUDA_PATH/include/cuviddec.h" 2>/dev/null || echo "cuviddec.h not found"
 
if command_exists ldconfig; then
    echo "Checking libraries in cache:"
    ldconfig -p | grep -E "(nvcuvid|nvidia-encode)" || echo "Libraries not found in cache"
fi

echo "GPU Codec installation completed successfully!"
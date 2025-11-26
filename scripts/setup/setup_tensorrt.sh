#!/bin/bash

# This script sets up NVIDIA TensorRT and ONNX **systemwide** and installs related Python packages. 
# We're going to detect the OS [Fedora, Ubuntu/Debian/ Arch] and install TensorRT accordingly as it depends on libnvinfer.so.
# This is required because PyPi packages alone do not provide the full TensorRT installation.
# After running this script, TensorRT libraries will be available systemwide, 
# and Python bindings will be installed in the current Python environment.

set -e

if [ -f "./scripts/cuda_toolkit.sh" ]; then
    source ./scripts/cuda_toolkit.sh
fi
 
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif command -v lsb_release >/dev/null 2>&1; then
        lsb_release -is | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}
 
download_file() {
    local url="$1"
    local filename="$2"
    
    if command -v aria2c >/dev/null 2>&1; then
        echo "Downloading with aria2c..."
        aria2c -s 16 -x 16 "$url" -o "$filename"
    elif command -v wget >/dev/null 2>&1; then
        echo "Downloading with wget..."
        wget "$url" -O "$filename"
    elif command -v curl >/dev/null 2>&1; then
        echo "Downloading with curl..."
        curl -L "$url" -o "$filename"
    else
        echo "Error: No download tool available (aria2c, wget, or curl)"
        exit 1
    fi
}
 
install_tensorrt() {
    local os="$1"

    
    case "$os" in
        ("fedora"|"rhel"|"centos")
            local filename="nv-tensorrt-local-repo-rhel9-10.13.3-cuda-13.0-1.0-1.x86_64.rpm"
            local url="https://developer.download.nvidia.com/compute/tensorrt/10.13.3/local_installers/$filename"

            echo "Installing TensorRT for Fedora/RHEL/CentOS..."
            download_file "$url" "$filename"
            
            if command -v dnf5 >/dev/null 2>&1; then
                sudo dnf5 install "./$filename"
                sudo dnf5 install tensorrt
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install "./$filename"
                sudo dnf install tensorrt
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install "./$filename"
                sudo yum install tensorrt
            else
                echo "Error: No package manager found (dnf5, dnf, or yum)"
                exit 1
            fi
            ;;
            
        ("ubuntu"|"debian")
            echo "Installing TensorRT for Ubuntu/Debian..."
            local deb_url="https://developer.download.nvidia.com/compute/tensorrt/10.13.3/local_installers/nv-tensorrt-local-repo-ubuntu2204-10.13.3-cuda-13.0_1.0-1_amd64.deb"
            local deb_filename="nv-tensorrt-local-repo-ubuntu2204-10.13.3-cuda-13.0_1.0-1_amd64.deb"
            
            download_file "$deb_url" "$deb_filename"
            sudo dpkg -i "./$deb_filename"
            sudo apt-get update
            sudo apt-get install tensorrt
            ;;
            
        ("arch")
        # Note that AUR packages are not officially supported by NVIDIA.
        # It is recommended to verify the integrity and authenticity of AUR packages before installation.
            echo "Installing TensorRT for Arch Linux..."
            if command -v yay >/dev/null 2>&1; then
                yay -S tensorrt
            elif command -v paru >/dev/null 2>&1; then
                paru -S tensorrt
            else
                echo "Please install tensorrt from AUR using your preferred AUR helper"
                echo "Or visit: https://developer.nvidia.com/tensorrt"
            fi
            ;;
            
        (*)
            echo "Unsupported OS: $os"
            echo "Please install TensorRT manually from: https://developer.nvidia.com/tensorrt"
            exit 1
            ;;
    esac
}
 
OS=$(detect_os)
echo "Detected OS: $OS"

install_tensorrt "$OS"
 
export NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=0

uv pip install tensorrt
uv pip install nvidia-tensorrt
uv pip install tensorrt-cu13-bindings
uv pip install polygraphy

uv pip install nvidia-pyindex
uv pip install onnx onnxruntime-gpu
uv pip install pycuda

echo "TensorRT and ONNX installation completed!"
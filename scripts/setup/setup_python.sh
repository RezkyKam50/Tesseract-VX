#!/bin/bash

set -e

source ./scripts/setup/cuda_toolkit.sh
source ./scripts/setup/gcc_switcher.sh

PYTHON_VERSION="3.14.0"
PYTHON_TAR="Python-${PYTHON_VERSION}.tar.xz"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TAR}"
INSTALL_PREFIX="$(pwd)/python-3.14" # using local installation to avoid requiring sudo
 
download_file() {
    local url="$1"
    local output="$2"
    
    if command -v aria2c &> /dev/null; then
        echo "Using aria2c for faster download..."
        aria2c -s 16 -x 16 "$url" -o "$output"
    elif command -v wget &> /dev/null; then
        echo "aria2c not found, using wget..."
        wget "$url" -O "$output"
    elif command -v curl &> /dev/null; then
        echo "Neither aria2c nor wget found, using curl..."
        curl -L "$url" -o "$output"
    else
        echo "Error: No download tool found. Please install aria2c, wget, or curl."
        exit 1
    fi
}
 
if [ ! -f "$PYTHON_TAR" ]; then
    echo "Downloading Python ${PYTHON_VERSION}..."
    download_file "$PYTHON_URL" "$PYTHON_TAR"
else
    echo "Python tarball already exists, skipping download."
fi
 
if [ ! -s "$PYTHON_TAR" ]; then
    echo "Error: Downloaded file is empty or doesn't exist."
    exit 1
fi

echo "Extracting Python source..."
tar xvf "$PYTHON_TAR"

cd "Python-${PYTHON_VERSION}"

echo "Configuring Python build..."
./configure --prefix="$INSTALL_PREFIX" \
            --enable-optimizations \
            --with-lto

echo "Building Python (this may take a while)..."
make -j$(nproc --all)

echo "Installing Python..."
sudo make altinstall

cd ..
 
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$INSTALL_PREFIX/bin/python3.14" -m venv .venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Python ${PYTHON_VERSION} installation completed successfully!"
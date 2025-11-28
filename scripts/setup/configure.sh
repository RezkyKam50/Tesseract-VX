#!/bin/bash
 
set -e

scripts_dir="./scripts/setup/"

echo "Checking mandatory dependencies..."
 
deps=(
    "ninja:ninja"
    "meson:meson"
    "aria2c:aria2"
    "gcc-14:gcc-14"
    "nvcc:cuda-toolkit-13-0"
)
 
check_dep() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 not found in PATH"
        echo "   Please install: $2"
        return 1
    else
        echo "$1 found"
        return 0
    fi
}
 
missing_deps=0
for dep in "${deps[@]}"; do
    IFS=':' read -r cmd package <<< "$dep"
    if ! check_dep "$cmd" "$package"; then
        missing_deps=$((missing_deps + 1))
    fi
done
 
if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    if [[ "$nvcc_version" != "13.0" ]]; then
        echo "ERROR: CUDA Toolkit version is $nvcc_version, but version 13.0 is required"
        missing_deps=$((missing_deps + 1))
    else
        echo "CUDA Toolkit 13.0 found"
    fi
fi
 
if [ $missing_deps -ne 0 ]; then
    echo ""
    echo "$missing_deps mandatory dependencies are missing."
    echo "Please install the missing dependencies and run this script again."
    exit 1
fi

echo ""
echo "All mandatory dependencies are available!"
 
echo ""
echo "Running setup scripts..."
 
echo "Running python setup..."
./${scripts_dir}/setup_python.sh

echo "Running TensorRT setup..."
./${scripts_dir}/setup_tensorrt.sh

echo "Running TensorRT Dependencies setup..."
./${scripts_dir}/setup_deps.sh
 

echo ""
echo "Setup completed successfully!"
#!/bin/bash
set -e

source .venv/bin/activate
source ./scripts/setup/gcc_switcher.sh 
source ./scripts/setup/cuda_toolkit.sh

export Python3_ROOT_DIR=/usr/local
export Python3_INCLUDE_DIR=/usr/local/include/python3.14
export Python3_LIBRARY=/usr/local/lib/libpython3.14.so
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export ONNX_ML=1

VIRTUAL_ENV=$(pwd)/.venv
 
ONNX_INCLUDE_DIR="/usr/local/include"
ONNX_LIBRARY="/usr/local/lib/libonnx.so"

echo "ONNX include dir: $ONNX_INCLUDE_DIR"
echo "ONNX library: $ONNX_LIBRARY"

cd ./thirdparty/onnxruntime
sudo rm -rf ./build 
git submodule update --init --recursive
 
rm -rf build/Linux/Release

# we dont need ML ops since were dealing with NN models only \

./build.sh \
    --config Release \
    --use_cuda \
    --cuda_home /opt/cuda \
    --cudnn_home /usr \
    --parallel=$(nproc) \
    --build_shared_lib \
    --use_full_protobuf \
    --disable_ml_ops \
    --skip_submodule_sync \
    --cmake_extra_defines \
        CMAKE_CUDA_ARCHITECTURES=89 \
        CMAKE_C_COMPILER_LAUNCHER=ccache \
        CMAKE_CXX_COMPILER_LAUNCHER=ccache \
        onnxruntime_BUILD_UNIT_TESTS=OFF \
        CMAKE_PREFIX_PATH="/usr/local" \
        onnxruntime_USE_PREINSTALLED_ONNX=ON \
        ONNX_NAMESPACE=onnx \
        ONNX_INCLUDE_DIR="${ONNX_INCLUDE_DIR}" \
        ONNX_LIBRARY="${ONNX_LIBRARY}" \
        Python3_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
        Python3_ROOT_DIR="${VIRTUAL_ENV}" \
        Python3_INCLUDE_DIR="${VIRTUAL_ENV}/include/python3.14" \
        Python3_LIBRARY="${VIRTUAL_ENV}/lib/libpython3.14.so"

uv pip install ./build/Linux/Release/*.whl
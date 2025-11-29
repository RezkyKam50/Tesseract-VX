#!/bin/bash
set -e   
 
source ./scripts/setup/gcc_switcher.sh 
source ./scripts/setup/cuda_toolkit.sh 
source .venv/bin/activate 
 
uv pip uninstall onnx 
sudo rm -rf /usr/local/include/onnx/
 
export Python3_ROOT_DIR=/usr/local
export Python3_INCLUDE_DIR=/usr/local/include/python3.14
export Python3_LIBRARY=/usr/local/lib/libpython3.14.so
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
 
cd ./thirdparty/onnx 

echo "Updating submodules..."
git submodule update --init --recursive

echo "Installing ONNX development requirements..."
uv pip install -r ./requirements-dev.txt

echo "Building ONNX C++ libraries from source..."

sudo rm -rf build_onnx
mkdir -p build_onnx 
cd build_onnx
 
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DONNX_ML=0 \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython3_ROOT_DIR="${Python3_ROOT_DIR}" \
  -DONNX_USE_LITE_PROTO=OFF \
  -DONNX_BUILD_TESTS=OFF 

make -j$(nproc)
sudo make install
 
sudo ldconfig

cd ..

echo "Installing typing_extensions..."
uv pip install typing_extensions
echo "Installing ONNX Python package..."
USE_NINJA=1 uv pip install .   

cd ..
 
echo "Verifying ONNX installation..."
python -c "import onnx; print('ONNX version: {}'.format(onnx.__version__)); print('ONNX location: {}'.format(onnx.__file__))"

echo "ONNX installation complete!"
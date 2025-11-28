cd onnxruntime
rm -rf ./build 
git submodule update --init --recursive
cd ..
source ./scripts/setup/gcc_switcher.sh 
uv pip install -r ./onnxruntime/requirements-dev.txt
./onnxruntime/build.sh --config Release --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF

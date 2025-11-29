# ./scripts/build_cv_cuda.sh
# run from parent directory
# currently opencv doesn't support Ninja build system
# known dependency hell : ffmpeg-free (non-patent ver.) remove && replace with the patented ffmpeg (check ./setup_ffmpeg.sh) <- necessary for performance

# Download NVIDIA GPU Accel. Codec for HW Accel. Video support:
# https://developer.nvidia.com/nvidia-video-codec-sdk/download

# build ffmpeg with CUDA Enabled
# sudo chmod +x ./build_ffpmeg.sh && ./build_ffpmeg.sh
# sudo chmod +x ./setup_nvcuvid.sh && ./setup_nvcuvid.sh
# Then, add this flag:
# -D WITH_NVCUVID=ON \
# ffmpeg specific
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
PY_ABI="3.13"

source ./scripts/setup/cuda_toolkit.sh
source ./scripts/setup/gcc_switcher.sh
source .venv/bin/activate
 
sudo rm -rf build_opencv
mkdir -p build_opencv
cd build_opencv


C_FLAGS="-O3 -march=native -mtune=native"
CXX_FLAGS="-O3 -march=native -mtune=native"

CUDA_FLAGS="\
-O3 \
--use_fast_math"

cmake ../thirdparty/opencv \
  -D CMAKE_BUILD_TYPE=Release \
  -D WITH_CUDA=ON \
  -D CMAKE_C_FLAGS="$C_FLAGS" \
  -D CMAKE_CXX_FLAGS="$CXX_FLAGS" \
  -D CMAKE_CUDA_FLAGS="$CUDA_FLAGS" \
  -D WITH_NVCUVID=ON \
  -D WITH_NVCUVENC=OFF \
  -D WITH_CUDNN=OFF \
  -D WITH_FFMPEG=ON \
  -D BUILD_opencv_videoio=ON \
  -D ENABLE_CCACHE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=../thirdparty/opencv_contrib/modules \
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_EXECUTABLE=$(which python3) \
  -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
  -D PYTHON3_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBPL') + '/libpython${PY_ABI}.so')") \
  -D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") 


echo "WARN: this will install opencv system-wide"
make -j$(nproc --all)
sudo make install

export PYTHONPATH=$PYTHONPATH:$HOME/Tesseract-VX/build_opencv/lib/python3


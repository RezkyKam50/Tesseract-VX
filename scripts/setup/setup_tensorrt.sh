source ./scripts/cuda_toolkit.sh

# for faster download, use aria2c instead of wget
# aria2c -s 16 -x 16

wget https://developer.download.nvidia.com/compute/tensorrt/10.13.3/local_installers/nv-tensorrt-local-repo-rhel9-10.13.3-cuda-13.0-1.0-1.x86_64.rpm
sudo dnf5 install ~/nv-tensorrt-local-repo-rhel9-10.13.3-cuda-13.0-1.0-1.x86_64.rpm
sudo dnf5 install tensorrt

export NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=0
uv pip install tensorrt
uv pip install nvidia-tensorrt
uv pip install tensorrt-cu13-bindings
uv pip install polygraphy

uv pip install nvidia-pyindex
uv pip install onnx onnxruntime-gpu
uv pip install pycuda
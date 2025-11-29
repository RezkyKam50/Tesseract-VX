source ./scripts/setup/cuda_toolkit.sh
source ./scripts/setup/gcc_switcher.sh

uv pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
python ./thirdparty/torch2trt/setup.py install
uv pip install numba cupy numpy 

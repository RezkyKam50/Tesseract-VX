source ./scripts/setup/cuda_toolkit.sh
source ./scripts/setup/gcc_switcher.sh

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
uv pip install numba cupy numpy cython_bbox cython tqdm thop tabulate pycocotools lap scipy nvtx PyNvVideoCodec

cd ./thirdparty/torch2trt 
uv pip install . --no-build-isolation
cd .. && cd onnxoptimizer
uv pip install . --no-build-isolation
cd .. && cd onnxsim
uv pip install . --no-build-isolation
cd .. && cd cupy
uv pip install . --no-build-isolation
cd .. && cd ..


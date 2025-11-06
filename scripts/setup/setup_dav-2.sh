# run this script from parents dir
source ./scripts/cuda_toolkit.sh

# pip install holoscan-cu13

git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
uv pip install -r requirements.txt # i reccommend to use uv for complex dependencies
uv pip install audioop-lts
uv pip install xformers
cd ..

cp -r ./Depth-Anything-V2/depth_anything_v2 ./depth_anything_v2 
cp -r ./Depth-Anything-V2/assets ./assets
rm -rf ./Depth-Anything-V2

mkdir checkpoints
cd checkpoints

git lfs install
git clone https://huggingface.co/depth-anything/Depth-Anything-V2-Base
git clone https://huggingface.co/depth-anything/Depth-Anything-V2-Large

cd ..





 
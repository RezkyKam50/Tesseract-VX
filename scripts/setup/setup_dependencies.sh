source ./scripts/setup/cuda_toolkit.sh
source ./scripts/setup/gcc_switcher.sh

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

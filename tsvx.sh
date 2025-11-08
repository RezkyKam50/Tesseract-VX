export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

source ./scripts/setup/cv_lib.sh

export PYTHONPATH="${PYTHONPATH}:${PWD}/src/bytetrack"
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \ 
     python src/tsvx.py --source ./src/bytetrack/videos/car.mp4 --save-video --output ./footage/output.mp4

#python src/spectralgraph.py --source 0  
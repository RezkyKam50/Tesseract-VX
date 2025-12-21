export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source .venv/bin/activate

echo "$(which python)"

sudo sysctl -w kernel.perf_event_paranoid=1
sudo sysctl -w kernel.kptr_restrict=0

nsys status -e

nsys profile \
    -o ./profiling/ \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --capture-range=nvtx \
    --nvtx-capture=all \
    --stop-on-exit=true \
    --kill=sigkill \
    ./tsvx.sh

nsys-ui ./profiling/*.nsys-rep
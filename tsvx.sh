#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/bytetrack"

source .venv/bin/activate
source ./scripts/setup/srclib.sh
 
taskset -c 0-8 python src/tsvx.py --source "0" --parallel --offload

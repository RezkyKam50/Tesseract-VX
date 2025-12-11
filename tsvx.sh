#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/bytetrack"

source .venv/bin/activate
source ./scripts/setup/srclib.sh
 
python src/tsvx.py --source ./footage/demo_footage2.mp4 --save-video --output ./footage/output.mp4 --parallel --optimize --gpu --no-display

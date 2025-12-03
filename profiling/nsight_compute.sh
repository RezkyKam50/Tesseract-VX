timeout 30 ncu -o ./profiling/nsight_prof -f tsvx.sh --profile-from-start off
ncu-ui ./profiling/*.ncu-rep
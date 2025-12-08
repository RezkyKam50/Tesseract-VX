sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
nsys profile -o ./profiling/trt_profile -f true --trace=cuda,nvtx --capture-range=nvtx tsvx.sh 
nsys-ui ./profiling/*.nsys-rep
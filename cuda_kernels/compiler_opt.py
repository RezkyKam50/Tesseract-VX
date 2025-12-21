# Compiler options for CUDA kernels
# Adjust these options to optimize performance

options=(
    '--maxrregcount=32',
    '-O1'
)

# using nvcc backend for cp.RawKernel may add start-up time but can yield better performance compared to nvrtc
backend = 'nvcc'   
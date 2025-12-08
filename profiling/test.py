#!/usr/bin/env python3
"""
NVTX Testing Script for NVIDIA Nsight Systems
This script demonstrates NVTX range marking for profiling Python code.
"""

import time
import random
import math
from numba import cuda
import nvtx

def initialize_array(size=1000000):
 
    with nvtx.annotate("initialize_array", color="red"):
        data = [random.random() for _ in range(size)]
        time.sleep(0.1)  
        return data

def process_data(data):
 
    with nvtx.annotate("process_data", color="green"):
        results = []
        for i in range(0, len(data), 1000):
            chunk = data[i:i+1000]
            chunk_sum = sum(chunk)
            chunk_avg = chunk_sum / len(chunk) if chunk else 0
            results.append(chunk_avg)
            time.sleep(0.01)
        return results

def gpu_computation():
 
    with nvtx.annotate("gpu_computation", color="blue"):
 
        data = cuda.to_device([i * 0.1 for i in range(10000)])
         
        @cuda.jit
        def square_kernel(arr):
            idx = cuda.grid(1)
            if idx < arr.size:
                arr[idx] = arr[idx] * arr[idx]
         
        threads_per_block = 256
        blocks_per_grid = (data.size + (threads_per_block - 1)) // threads_per_block
        square_kernel[blocks_per_grid, threads_per_block](data)
         
        result = data.copy_to_host()
        return result

def main():
    nvtx.mark("Application Start", color="yellow")
     
    nvtx.push_range("main_function", color="purple")
    
    try:
        print("Starting NVTX profiling test...")
         
        nvtx.push_range("data_initialization_phase", color="cyan")
        data = initialize_array(500000)
        nvtx.pop_range()
         
        with nvtx.annotate("data_processing_phase", color="magenta"):
            processed = process_data(data)
            print(f"Processed {len(processed)} chunks of data")
         
        with nvtx.annotate("gpu_phase", color="orange"):
            gpu_result = gpu_computation()
            print(f"GPU computed {len(gpu_result)} values")
            print(f"Sample GPU result: {gpu_result[:5]}")
         
        with nvtx.annotate("final_processing", color="blue"):
            time.sleep(0.05)
             
            with nvtx.annotate("aggregation", color="green"):
                total = sum(processed) + sum(gpu_result[:1000])
                print(f"Final aggregated result: {total}")
    
    finally:
 
        nvtx.pop_range()
        nvtx.mark("Application End", color="yellow")

def custom_range_example():
 
    nvtx.push_range("custom_calculation", color="red")
    
    try:
 
        nvtx.mark("Starting heavy computation", color="white")
         
        result = 0
        for i in range(100):
            nvtx.push_range(f"iteration_{i}", color="gray")
            result += math.sin(i * 0.1) * math.cos(i * 0.05)
            time.sleep(0.001)
            nvtx.pop_range()
        
        nvtx.mark("Computation complete", color="white")
        return result
    finally:
        nvtx.pop_range()

if __name__ == "__main__":
  
    try:
        import nvtx
        from numba import cuda
        print("NVTX and Numba CUDA are available.")
         
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install required packages:")
        print("pip install nvtx numba")
        print("Note: CUDA toolkit must be installed for GPU operations")
        exit(1)
     
    main()
     
    print("\nRunning custom range example...")
    custom_range_example()
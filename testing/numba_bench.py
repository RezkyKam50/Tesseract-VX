import time
import numba
 
def calculate_coords_original(x, y, w, h):
    return (x + w, y + h)
 
@numba.jit(nopython=True)
def calculate_coords_numba(x, y, w, h):
    return (x + w, y + h)
 
def benchmark_coord_calculation():
    iterations = 1000000
     
    start = time.time()
    for i in range(iterations):
        calculate_coords_original(i, i+1, i+2, i+3)
    original_time = time.time() - start
     
    calculate_coords_numba(1, 2, 3, 4)   
    start = time.time()
    for i in range(iterations):
        calculate_coords_numba(i, i+1, i+2, i+3)
    numba_time = time.time() - start
    
    print(f"Original: {original_time:.6f}s")
    print(f"Numba:    {numba_time:.6f}s")
    print(f"Speedup:  {original_time/numba_time:.2f}x")


benchmark_coord_calculation()
 
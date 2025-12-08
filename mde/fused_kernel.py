import cupy as cp
import nvtx

fused_resize_bgr2rgb_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_resize_bgr2rgb_3d(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_w || y >= out_h) return;
     
    if (out_w <= 0 || out_h <= 0 || in_w <= 0 || in_h <= 0) return;
    
    const float scale_x = (float)in_w / (float)out_w;
    const float scale_y = (float)in_h / (float)out_h;
     
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
     
    src_x = fmaxf(0.0f, fminf(src_x, (float)(in_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(in_h - 1)));
     
    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
     
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    float dx1 = 1.0f - dx;
    float dy1 = 1.0f - dy;
     
    int idx00 = (y0 * in_w + x0) * 3;
    int idx01 = (y0 * in_w + x1) * 3;
    int idx10 = (y1 * in_w + x0) * 3;
    int idx11 = (y1 * in_w + x1) * 3;
    int dst_idx = (y * out_w + x) * 3;
      
    float b00 = src[idx00 + 0];  // B channel
    float b01 = src[idx01 + 0];
    float b10 = src[idx10 + 0];
    float b11 = src[idx11 + 0];
    
    float g00 = src[idx00 + 1];  // G channel
    float g01 = src[idx01 + 1];
    float g10 = src[idx10 + 1];
    float g11 = src[idx11 + 1];
    
    float r00 = src[idx00 + 2];  // R channel
    float r01 = src[idx01 + 2];
    float r10 = src[idx10 + 2];
    float r11 = src[idx11 + 2];
     
    float top_b = b00 * dx1 + b01 * dx;
    float bottom_b = b10 * dx1 + b11 * dx;
    dst[dst_idx + 2] = top_b * dy1 + bottom_b * dy;  // B -> R position
    
    float top_g = g00 * dx1 + g01 * dx;
    float bottom_g = g10 * dx1 + g11 * dx;
    dst[dst_idx + 1] = top_g * dy1 + bottom_g * dy;  // G -> G position
    
    float top_r = r00 * dx1 + r01 * dx;
    float bottom_r = r10 * dx1 + r11 * dx;
    dst[dst_idx + 0] = top_r * dy1 + bottom_r * dy;  // R -> B position
}
''', 'fused_resize_bgr2rgb_3d')

def fused_resize_bgr2rgb_3d(img_cp, out_h, out_w, block_size):
 
    h, w, c = img_cp.shape
 
    img_f = img_cp.astype(cp.float32, copy=False)
    out = cp.empty((out_h, out_w, 3), dtype=cp.float32)
    
    grid_x = (out_w + block_size[0] - 1) // block_size[0]
    grid_y = (out_h + block_size[1] - 1) // block_size[1]
    grid = (grid_x, grid_y)
    
    fused_resize_bgr2rgb_kernel(
        grid,
        block_size,
        (img_f, out, h, w, out_h, out_w)
    )
    
    return out
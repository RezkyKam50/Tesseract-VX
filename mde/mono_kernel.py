import cupy as cp
import nvtx

bilinear_kernel_2d = cp.RawKernel(r'''
extern "C" __global__
void resize_bilinear(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= out_w || idy >= out_h) return;
     
    const float scale_x = __fdividef((float)(in_w - 1), (float)(out_w - 1));
    const float scale_y = __fdividef((float)(in_h - 1), (float)(out_h - 1));
     
    const float src_x = idx * scale_x;
    const float src_y = idy * scale_y;
    
    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);
     
    const float dx = src_x - x0;
    const float dy = src_y - y0;
    const float one_minus_dx = 1.0f - dx;
    const float one_minus_dy = 1.0f - dy;
    
    const float v00 = src[y0 * in_w + x0];
    const float v01 = src[y0 * in_w + x1];
    const float v10 = src[y1 * in_w + x0];
    const float v11 = src[y1 * in_w + x1];
     
    const float top = __fmaf_rn(v01, dx, __fmaf_rn(v00, one_minus_dx, 0.0f));
    const float bottom = __fmaf_rn(v11, dx, __fmaf_rn(v10, one_minus_dx, 0.0f));
    
    dst[idy * out_w + idx] = __fmaf_rn(bottom, dy, __fmaf_rn(top, one_minus_dy, 0.0f));
     
}
''', 'resize_bilinear')

bilinear_kernel_3d = cp.RawKernel(r'''
extern "C" __global__
void resize_bilinear_3d(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    int channels
){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_w || y >= out_h) return;
     
    const float scale_x = __fdividef((float)in_w, (float)out_w);
    const float scale_y = __fdividef((float)in_h, (float)out_h);
     
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
     
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1));
    
    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);
    
    const float dx = src_x - x0;
    const float dy = src_y - y0;
    const float one_minus_dx = 1.0f - dx;
    const float one_minus_dy = 1.0f - dy;
     
    const int src_idx_y0 = y0 * in_w * channels;
    const int src_idx_y1 = y1 * in_w * channels;
    const int src_idx_x0 = x0 * channels;
    const int src_idx_x1 = x1 * channels;
    const int dst_idx_base = (y * out_w + x) * channels;
     
    if (channels == 3) {
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            const float v00 = src[src_idx_y0 + src_idx_x0 + c];
            const float v01 = src[src_idx_y0 + src_idx_x1 + c];
            const float v10 = src[src_idx_y1 + src_idx_x0 + c];
            const float v11 = src[src_idx_y1 + src_idx_x1 + c];
             
            const float top = __fmaf_rn(v01, dx, __fmaf_rn(v00, one_minus_dx, 0.0f));
            const float bottom = __fmaf_rn(v11, dx, __fmaf_rn(v10, one_minus_dx, 0.0f));
            dst[dst_idx_base + c] = __fmaf_rn(bottom, dy, __fmaf_rn(top, one_minus_dy, 0.0f));
        }
    } else {
        #pragma unroll 4
        for (int c = 0; c < channels; c++) {
            const float v00 = src[src_idx_y0 + src_idx_x0 + c];
            const float v01 = src[src_idx_y0 + src_idx_x1 + c];
            const float v10 = src[src_idx_y1 + src_idx_x0 + c];
            const float v11 = src[src_idx_y1 + src_idx_x1 + c];
            
            const float top = __fmaf_rn(v01, dx, __fmaf_rn(v00, one_minus_dx, 0.0f));
            const float bottom = __fmaf_rn(v11, dx, __fmaf_rn(v10, one_minus_dx, 0.0f));
            dst[dst_idx_base + c] = __fmaf_rn(bottom, dy, __fmaf_rn(top, one_minus_dy, 0.0f));
        }
    }
     
}
''', 'resize_bilinear_3d')

bgr2rgb_float_kernel = cp.RawKernel(r'''
extern "C" __global__
void bgr2rgb_float(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int h, int w
){

    int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y >= h) return;
    
    const int row_start = y * w * 3;
     
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int x = x_start + i;
        if (x >= w) break;
        
        int src_idx = row_start + x * 3;
        int dst_idx = src_idx;
         
        float b = src[src_idx];
        float g = src[src_idx + 1];
        float r = src[src_idx + 2];
        
        dst[dst_idx] = r;
        dst[dst_idx + 1] = g;
        dst[dst_idx + 2] = b;
    }
     
}
''', 'bgr2rgb_float')

def cupy_resize_2d(depth_cp, out_h, out_w, block_size):
    
    in_h, in_w = depth_cp.shape
    depth_cp_f = depth_cp.astype(cp.float32, copy=False)
    out_cp = cp.empty((out_h, out_w), dtype=cp.float32)
     
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1]
    )
     
    bilinear_kernel_2d(
        grid,
        block_size,
        (depth_cp_f, out_cp, in_h, in_w, out_h, out_w)
    )
     
    return out_cp

def cupy_resize_3d(img_cp, out_w, out_h, block_size):
    
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"
        
    img_f = img_cp.astype(cp.float32, copy=False)
    out = cp.empty((out_h, out_w, 3), dtype=cp.float32)
     
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1]
    )
      
    bilinear_kernel_3d(
        grid,
        block_size,
        (
            img_f,
            out,
            h, w,
            out_h, out_w,
            3  # channels
        )
    )
     
    return out

def cupy_cvt_bgr2rgb_float(img_cp, block_size):
    
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"
    
    out = cp.empty_like(img_cp)
     
    grid = (
        (w + block_size[0] - 1) // block_size[0],
        (h + block_size[1] - 1) // block_size[1]
    )
      
    bgr2rgb_float_kernel(
        grid,
        block_size,
        (img_cp, out, h, w)
    )
     
    return out

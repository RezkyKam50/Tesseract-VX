import cupy as cp
import nvtx
from cuda_kernels.compiler_opt import options, backend

# -> adding "cutlass" to function name triggers several optimization 
# Refs:
# https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/
# https://news.ycombinator.com/item?id=45458948
# the trick is mainly for fp8 computation but we'll try it here

bilinear_kernel_2c = cp.RawKernel(r'''
extern "C" __global__
void cutlass_resize_bilinear(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_w || y >= out_h) return;
     
    const float scale_x = __fdividef((float)(in_w - 1), (float)(out_w - 1));
    const float scale_y = __fdividef((float)(in_h - 1), (float)(out_h - 1));
     
    const float src_x = x * scale_x;
    const float src_y = y * scale_y;
    
    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);
     
    const float dx = src_x - x0;
    const float dy = src_y - y0;
    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;
    
    const float v00 = src[y0 * in_w + x0];
    const float v01 = src[y0 * in_w + x1];
    const float v10 = src[y1 * in_w + x0];
    const float v11 = src[y1 * in_w + x1];
     
    const float top = __fmaf_rn(v01, dx, __fmaf_rn(v00, dx1, 0.0f));
    const float bottom = __fmaf_rn(v11, dx, __fmaf_rn(v10, dx1, 0.0f));
    
    dst[y * out_w + x] = __fmaf_rn(bottom, dy, __fmaf_rn(top, dy1, 0.0f));
     
}
''', 'cutlass_resize_bilinear', options=options, backend=backend)

def cupy_resize_2c(depth_cp, out_h, out_w, block_size):
    
    in_h, in_w = depth_cp.shape
    depth_cp_f = depth_cp.astype(cp.float32, copy=False)
    out_cp = cp.empty((out_h, out_w), dtype=cp.float32)
     
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1]
    )
     
    bilinear_kernel_2c(
        grid,
        block_size,
        (depth_cp_f, out_cp, in_h, in_w, out_h, out_w)
    )
     
    return out_cp


bilinear_kernel_3c = cp.RawKernel(r'''
extern "C" __global__
void cutlass_resize_bilinear_3c(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    int c
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
     
    if (x >= out_w || y >= out_h) return;

    const float scale_x = __fdividef((float)in_w, (float)out_w);
    const float scale_y = __fdividef((float)in_h, (float)out_h);
    
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
    
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1.0001f));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1.0001f));

    
    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);
    
    const float dx = src_x - x0;
    const float dy = src_y - y0;
    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;
    
    const int src_idx_y0 = y0 * in_w * c;
    const int src_idx_y1 = y1 * in_w * c;
    const int src_idx_x0 = x0 * c;
    const int src_idx_x1 = x1 * c;
    
    const float v00 = src[src_idx_y0 + src_idx_x0 + ch];
    const float v01 = src[src_idx_y0 + src_idx_x1 + ch];
    const float v10 = src[src_idx_y1 + src_idx_x0 + ch];
    const float v11 = src[src_idx_y1 + src_idx_x1 + ch];
    
    const float top = __fmaf_rn(v01, dx, __fmaf_rn(v00, dx1, 0.0f));
    const float bottom = __fmaf_rn(v11, dx, __fmaf_rn(v10, dx1, 0.0f));
    
    const int dst_idx = (y * out_w + x) * c + ch;
    dst[dst_idx] = __fmaf_rn(bottom, dy, __fmaf_rn(top, dy1, 0.0f));
}
''', 'cutlass_resize_bilinear_3c', options=options, backend=backend)

def cupy_resize_3c(img_cp, out_w, out_h, block_size):
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"

    img_cp = cp.ascontiguousarray(img_cp)
    img_f = img_cp.astype(cp.float32, copy=True)
    out = cp.empty((out_h, out_w, 3), dtype=cp.float32)
    
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1],
        c
    )
    
    bilinear_kernel_3c(
        grid,
        block_size,
        (
            img_f,
            out,
            h, w,
            out_h, out_w,
            c
        )
    )
    
    return out


bgr2rgb_float_kernel = cp.RawKernel(r'''
extern "C" __global__
void cutlass_bgr2rgb_float(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int h, int w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x >= w || y >= h) return;

    
    int idx = (y * w + x) * 3 + c;
    
    // Map BGR to RGB: channel 0<->2, channel 1 stays
    int src_c = (c == 0) ? 2 : (c == 2) ? 0 : 1;
    int src_idx = (y * w + x) * 3 + src_c;
    
    dst[idx] = src[src_idx];
}
''', 'cutlass_bgr2rgb_float', options=options, backend=backend)

def cupy_cvt_bgr2rgb_float(img_cp, block_size):
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"
    img_cp = cp.ascontiguousarray(img_cp)
    out = cp.empty_like(img_cp)
    
    grid = (
        (w + block_size[0] - 1) // block_size[0],
        (h + block_size[1] - 1) // block_size[1],
        c
    )
    
    bgr2rgb_float_kernel(
        grid,
        block_size,
        (img_cp, out, h, w)
    )
    
    return out

_cust_preprocess_kernel = cp.RawKernel(r'''
extern "C" __global__
void cutlass_preprocess_kernel(
    const float* __restrict__ resized_img,
    float* __restrict__ padded_img,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    const int target_h,
    const int target_w,
    const int padded_h,
    const int padded_w,
    const int c
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= padded_w || y >= padded_h || ch >= c) return;
    
    const float inv_255 = 0.003921568627f;
    const float fill_value = 114.0f;
    
    int out_idx = (y * padded_w + x) * 3 + ch;
     
    float inv_std_c = __frcp_rn(std[ch]);
    
    if (y < target_h && x < target_w) {
        int in_idx = (y * target_w + x) * 3;
         
        int src_c;
        if (ch == 0) src_c = 2;      // dst R <- src B
        else if (ch == 1) src_c = 1; // dst G <- src G
        else src_c = 0;            // dst B <- src R
        
        float pixel_val = __ldg(&resized_img[in_idx + src_c]);
        pixel_val = __fmaf_rn(pixel_val, inv_255, -mean[ch]) * inv_std_c;
        padded_img[out_idx] = pixel_val;
    } else {
        float padding_val = __fmaf_rn(fill_value, inv_255, -mean[ch]) * inv_std_c;
        padded_img[out_idx] = padding_val;
    }
}
''', 'cutlass_preprocess_kernel', options=options, backend=backend)

def cust_mot_preprocess(
    resized_img,          
    target_h,         
    target_w,         
    padded_h,       
    padded_w,         
    mean,                  
    std,                   
    block_size,         
):
    _, _, c = resized_img.shape
    
    assert c == 3, "Only 3-channel images supported"

    if not isinstance(resized_img, cp.ndarray):
        resized_img = cp.array(resized_img, dtype=cp.float32)
    
    padded_img_hwc = cp.empty((padded_h, padded_w, c), dtype=cp.float32)
     
    grid = (
        (padded_w + block_size[0] - 1) // block_size[0],
        (padded_h + block_size[1] - 1) // block_size[1],
        (c + block_size[2] - 1) // block_size[2]
    )
    
    
    mean_cp = cp.array(mean, dtype=cp.float32)
    std_cp = cp.array(std, dtype=cp.float32)
     
    _cust_preprocess_kernel(
        grid,
        block_size,
        (
            resized_img,                 
            padded_img_hwc,               
            mean_cp,                 
            std_cp,                      
            target_h,             
            target_w,               
            padded_h,                
            padded_w,      
            c                         
        )
    )
    
    return padded_img_hwc

_cust_transpose_kernel = cp.RawKernel(r'''
extern "C" __global__
void cutlass_transpose_kernel(
    const float* padded_img_hwc,  
    float* padded_img_chw,         
    const int h,
    const int w,
    const int c
) {
    // 3D thread indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;    // width
    int y = blockIdx.y * blockDim.y + threadIdx.y;    // height
    int ch = blockIdx.z * blockDim.z + threadIdx.z;    // channel
    
    if (x >= w || y >= h || ch >= c) {
        return;
    }
    
    int hw = h * w;
    int hw_idx = y * w + x;
     
    int hwc_idx = hw_idx * c + ch;
    int chw_idx = ch * hw + hw_idx;
    padded_img_chw[chw_idx] = padded_img_hwc[hwc_idx];
}
''', 'cutlass_transpose_kernel', options=options, backend=backend)


def cust_mot_transpose_hwc_to_chw(
    padded_img_hwc,   
    block_size       
):
    h, w, c = padded_img_hwc.shape
    
    assert c == 3, "Only 3-channel images supported"

    padded_img_chw = cp.empty((c, h, w), dtype=cp.float32)
    
    grid = (
        (w + block_size[0] - 1) // block_size[0],
        (h + block_size[1] - 1) // block_size[1],
        (c + block_size[2] - 1) // block_size[2]
    )
     
    _cust_transpose_kernel(
        grid,
        block_size,
        (
            padded_img_hwc,   
            padded_img_chw,   
            h,          
            w,            
            c          
        )
    )
     
    padded_img_chw = cp.ascontiguousarray(padded_img_chw)
    
    return padded_img_chw
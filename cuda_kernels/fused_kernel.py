import cupy as cp
from cuda_kernels.compiler_opt import options, backend

# -> adding "cutlass" to function name triggers several optimization 
# Refs:
# https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/
# https://news.ycombinator.com/item?id=45458948
# the trick is mainly for fp8 computation but we'll try it here

fused_resize_bgr2rgb_kernel = cp.RawKernel(r'''
extern "C" __global__
void cutlass_fused_resize_bgr2rgb_3c(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= out_w || y >= out_h || c >= 3 || 
        out_w <= 0 || out_h <= 0 || in_w <= 0 || in_h <= 0) return;
    
    // Map BGR to RGB: channel 0 (B in src) -> channel 2 (R in dst)
    //                channel 1 (G in src) -> channel 1 (G in dst)
    //                channel 2 (R in src) -> channel 0 (B in dst)
    int src_c, dst_c;
    if (c == 0) {
        src_c = 2;  // R in source
        dst_c = 0;  // B in destination
    } else if (c == 1) {
        src_c = 1;  // G in source
        dst_c = 1;  // G in destination
    } else { // c == 2
        src_c = 0;  // B in source
        dst_c = 2;  // R in destination
    }
    
    const float scale_x = __fdividef((float)in_w, (float)out_w);  
    const float scale_y = __fdividef((float)in_h, (float)out_h);
     
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
                                   
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1));                                       

    int x0 = __float2int_rd(src_x);
    int y0 = __float2int_rd(src_y);                                      
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
                                           
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    float dx1 = 1.0f - dx;
    float dy1 = 1.0f - dy;
     
    int idx00 = (y0 * in_w + x0) * 3 + src_c;
    int idx01 = (y0 * in_w + x1) * 3 + src_c;
    int idx10 = (y1 * in_w + x0) * 3 + src_c;
    int idx11 = (y1 * in_w + x1) * 3 + src_c;
    int dst_idx = (y * out_w + x) * 3 + dst_c;
       
    float v00 = src[idx00];
    float v01 = src[idx01];
    float v10 = src[idx10];
    float v11 = src[idx11];
                                                                 
    float top = __fmaf_rn(v00, dx1, v01 * dx);
    float bottom = __fmaf_rn(v10, dx1, v11 * dx);
    dst[dst_idx] = __fmaf_rn(top, dy1, bottom * dy);
}
''', 'cutlass_fused_resize_bgr2rgb_3c', options=options, backend=backend)

def fused_resize_bgr2rgb_3c(img_cp, out_h, out_w, block_size):

    h, w, c = img_cp.shape

    assert c == 3, "Only 3-channel images supported"
    
    img_f = img_cp.astype(cp.float32, copy=False)
    out = cp.empty((out_h, out_w, 3), dtype=cp.float32) 
     

    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1],
        (c + block_size[2] - 1) // block_size[2]
    )
    
    fused_resize_bgr2rgb_kernel(
        grid,
        block_size,
        (img_f, out, h, w, out_h, out_w)
    )
    return out

_cust_fused_preprocess_chwT = cp.RawKernel(r'''
extern "C" __global__
void cutlass_fused_preprocess_kernel(
    const float* __restrict__ resized_img,   
    float* __restrict__ padded_img_chw,      
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
     
    float inv_std[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        inv_std[i] = __frcp_rn(std[i]);
    }
     
    int hw = padded_h * padded_w;
    int chw_idx = ch * hw + y * padded_w + x;
     
    if (y < target_h && x < target_w) {
                                           
        int in_base = (y * target_w + x) * c;
         
        float pixel_val;
        if (ch == 0) {  // R channel from B input
            pixel_val = __ldg(&resized_img[in_base + 2]);
        } else if (ch == 1) {  // G channel from G input
            pixel_val = __ldg(&resized_img[in_base + 1]);
        } else {  // ch == 2, B channel from R input
            pixel_val = __ldg(&resized_img[in_base + 0]);
        }
         
        int mean_idx;
        if (ch == 0) {  // R channel uses mean[2]
            mean_idx = 2;
        } else if (ch == 1) {  // G channel uses mean[1]
            mean_idx = 1;
        } else {  // B channel uses mean[0]
            mean_idx = 0;
        }
        
        padded_img_chw[chw_idx] = __fmaf_rn(pixel_val, inv_255, -mean[mean_idx]) * inv_std[mean_idx];
    } else {
 
        float padding_val = __fmaf_rn(114.0f, inv_255, -mean[ch]) * inv_std[ch];
        padded_img_chw[chw_idx] = padding_val;
    }
}
''', 'cutlass_fused_preprocess_kernel', options=options, backend=backend)

def cust_mot_fused_preproc_hwcT(
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
     
    padded_img_chw = cp.empty((c, padded_h, padded_w), dtype=cp.float32)
     
    grid = (
        (padded_w + block_size[0] - 1) // block_size[0],
        (padded_h + block_size[1] - 1) // block_size[1],
        (c + block_size[2] - 1) // block_size[2] 
    )
    
    mean_cp = cp.array(mean, dtype=cp.float32)
    std_cp = cp.array(std, dtype=cp.float32)
     
    _cust_fused_preprocess_chwT(
        grid,
        block_size,
        (
            resized_img,           
            padded_img_chw,        
            mean_cp,                
            std_cp,               
            target_h,         
            target_w,           
            padded_h,         
            padded_w,          
            c                
        )
    )
    
    return padded_img_chw


# padded_img[:target_h, :target_w, :] = resized_img
# padded_img = padded_img[:, :, ::-1] / 255.0  
# mean_array = cp.array(self.mean).reshape(1, 1, 3)
# padded_img -= mean_array
# std_array = cp.array(self.std).reshape(1, 1, 3)
# padded_img /= std_array
# padded_img = padded_img.transpose((2, 0, 1))
# padded_img = cp.ascontiguousarray(padded_img, dtype=cp.float32)

_cust_fused_resize_preprocess_chwT = cp.RawKernel(r'''
extern "C" __global__
void cutlass_fused_resize_preprocess_transpose(
    const float* __restrict__ src,
    float* __restrict__ dst_chw,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int in_h, int in_w,
    int target_h, int target_w,
    int padded_h, int padded_w,
    int c
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= padded_w || y >= padded_h || ch >= 3) return;
    
    const float inv_255 = 0.003921568627f;
    const float inv_std_c = __frcp_rn(std[ch]);
    
    const int hw = padded_h * padded_w;
    const int hw_idx = y * padded_w + x;
    
    float normalized;
     
    if (y < target_h && x < target_w) {
 
        const float scale_x = __fdividef((float)in_w, (float)target_w);
        const float scale_y = __fdividef((float)in_h, (float)target_h);
        
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
        float resized_val = __fmaf_rn(bottom, dy, __fmaf_rn(top, dy1, 0.0f));
         
        normalized = __fmaf_rn(resized_val, inv_255, -mean[ch]) * inv_std_c;
    } else {
        normalized = __fmaf_rn(114.0f, inv_255, -mean[ch]) * inv_std_c;
    }
     
    int dst_c = 2 - ch;  // 0->2, 1->1, 2->0
     
    int chw_idx = dst_c * hw + hw_idx;
    dst_chw[chw_idx] = normalized;
}
''', 'cutlass_fused_resize_preprocess_transpose', options=options, backend=backend)


def cust_mot_resize_preprocess_chwT(
    img_cp,          
    target_h,          
    target_w,      
    padded_h,         
    padded_w,       
    mean,            
    std,             
    block_size
):
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"
     
    img_f = img_cp.astype(cp.float32, copy=False)
     
    out_chw = cp.empty((3, padded_h, padded_w), dtype=cp.float32)
     
    grid = (
        (padded_w + block_size[0] - 1) // block_size[0],
        (padded_h + block_size[1] - 1) // block_size[1],
        (c + block_size[2] - 1) // block_size[2] 
    )
     
    mean_cp = cp.array(mean, dtype=cp.float32)
    std_cp = cp.array(std, dtype=cp.float32)
     
    _cust_fused_resize_preprocess_chwT(
        grid,
        block_size,
        (
            img_f,
            out_chw,
            mean_cp,
            std_cp,
            h, w,
            target_h, target_w,
            padded_h, padded_w,
            c
        )
    )
    
    return cp.ascontiguousarray(out_chw)

# img_cp = img_cp / 255.0
# img_cp = (img_cp - self.mean) / self.std
# img_cp = cp.transpose(img_cp, (2, 0, 1))
# img_cp = cp.expand_dims(img_cp, axis=0)


_cust_fused_nhwc_nchw = cp.RawKernel(r'''
extern "C" __global__
void cutlass_preprocess_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    const int h,
    const int w,
    const int c
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= w || y >= h || ch >= c)
        return;

    int input_idx = (y * w + x) * c + ch;
    int output_idx = (ch * h + y) * w + x;

    float px = input[input_idx];

    // Fast divide by 255
    float normalized = px * (1.0f / 255.0f);

    // Use fdividef + fmaf so the compiler stops pretending it's too busy
    float invstd = fdividef(1.0f, std[ch]);

    // (normalized - mean[ch]) / std[ch]
    // result = (normalized - mean)*invstd
    float result = fmaf(normalized - mean[ch], invstd, 0.0f);

    output[output_idx] = result;
}
''', 'cutlass_preprocess_fused_kernel', options=options, backend=backend)


def cust_mde_nhwc_nchw(img_cp, mean, std, block_size):
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"

    img_cp = img_cp.astype(cp.float32)
    out = cp.empty((1, c, h, w), dtype=cp.float32)

    grid_x = (w + block_size[0] - 1) // block_size[0]
    grid_y = (h + block_size[1] - 1) // block_size[1]
    grid_z = (c + block_size[2] - 1) // block_size[2]

    _cust_fused_nhwc_nchw(
        (grid_x, grid_y, grid_z),
        block_size,
        (img_cp, out[0], mean, std, h, w, c)
    )

    return out
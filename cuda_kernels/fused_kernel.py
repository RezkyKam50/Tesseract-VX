import cupy as cp
import nvtx

fused_resize_bgr2rgb_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_resize_bgr2rgb_3c(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= out_w || y >= out_h || channel >= 3 || 
        out_w <= 0 || out_h <= 0 || in_w <= 0 || in_h <= 0) return;
    
    // Map BGR to RGB: channel 0 (B in src) -> channel 2 (R in dst)
    //                channel 1 (G in src) -> channel 1 (G in dst)
    //                channel 2 (R in src) -> channel 0 (B in dst)
    int src_channel, dst_channel;
    if (channel == 0) {
        src_channel = 2;  // R in source
        dst_channel = 0;  // B in destination
    } else if (channel == 1) {
        src_channel = 1;  // G in source
        dst_channel = 1;  // G in destination
    } else { // channel == 2
        src_channel = 0;  // B in source
        dst_channel = 2;  // R in destination
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
     
    int idx00 = (y0 * in_w + x0) * 3 + src_channel;
    int idx01 = (y0 * in_w + x1) * 3 + src_channel;
    int idx10 = (y1 * in_w + x0) * 3 + src_channel;
    int idx11 = (y1 * in_w + x1) * 3 + src_channel;
    int dst_idx = (y * out_w + x) * 3 + dst_channel;
       
    float v00 = src[idx00];
    float v01 = src[idx01];
    float v10 = src[idx10];
    float v11 = src[idx11];
                                                                 
    float top = __fmaf_rn(v00, dx1, v01 * dx);
    float bottom = __fmaf_rn(v10, dx1, v11 * dx);
    dst[dst_idx] = __fmaf_rn(top, dy1, bottom * dy);
}
''', 'fused_resize_bgr2rgb_3c')

def fused_resize_bgr2rgb_3c(img_cp, out_h, out_w, block_size):

    h, w, c = img_cp.shape

    if c != 3:
        raise ValueError(f"Input image must have 3 channels (BGR), got {c} channels")
    
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
void fused_preprocess_kernel(
    const float* __restrict__ resized_img,   
    float* __restrict__ padded_img_chw,      
    const float* __restrict__ mean,
    const float* __restrict__ std,
    const int target_height,
    const int target_width,
    const int padded_height,
    const int padded_width,
    const int channels
) {
 
    int x = blockIdx.x * blockDim.x + threadIdx.x;     
    int y = blockIdx.y * blockDim.y + threadIdx.y;    
    int c = blockIdx.z * blockDim.z + threadIdx.z;     
     
    if (x >= padded_width || y >= padded_height || c >= channels) return;
    
    const float inv_255 = 0.003921568627f;
     
    float inv_std[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        inv_std[i] = __frcp_rn(std[i]);
    }
     
    int hw = padded_height * padded_width;
    int chw_idx = c * hw + y * padded_width + x;
     
    if (y < target_height && x < target_width) {
                                           
        int in_base = (y * target_width + x) * channels;
         
        float pixel_val;
        if (c == 0) {  // R channel from B input
            pixel_val = __ldg(&resized_img[in_base + 2]);
        } else if (c == 1) {  // G channel from G input
            pixel_val = __ldg(&resized_img[in_base + 1]);
        } else {  // c == 2, B channel from R input
            pixel_val = __ldg(&resized_img[in_base + 0]);
        }
         
        int mean_idx;
        if (c == 0) {  // R channel uses mean[2]
            mean_idx = 2;
        } else if (c == 1) {  // G channel uses mean[1]
            mean_idx = 1;
        } else {  // B channel uses mean[0]
            mean_idx = 0;
        }
        
        padded_img_chw[chw_idx] = __fmaf_rn(pixel_val, inv_255, -mean[mean_idx]) * inv_std[mean_idx];
    } else {
 
        float padding_val = __fmaf_rn(114.0f, inv_255, -mean[c]) * inv_std[c];
        padded_img_chw[chw_idx] = padding_val;
    }
}
''', 'fused_preprocess_kernel')

def cust_mot_fused_preproc_hwcT(
    resized_img,            
    target_height,          
    target_width,         
    padded_height,        
    padded_width,          
    mean,                  
    std,                   
    block_size,            
):
    _, _, channels = resized_img.shape
    
    assert channels == 3, "Only 3-channel images supported"
    
    if not isinstance(resized_img, cp.ndarray):
        resized_img = cp.array(resized_img, dtype=cp.float32)
     
    padded_img_chw = cp.empty((channels, padded_height, padded_width), dtype=cp.float32)
     
    grid = (
        (padded_width + block_size[0] - 1) // block_size[0],
        (padded_height + block_size[1] - 1) // block_size[1],
        (channels + block_size[2] - 1) // block_size[2] 
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
            target_height,         
            target_width,           
            padded_height,         
            padded_width,          
            channels                
        )
    )
    
    return padded_img_chw

_cust_fused_resize_preprocess_chwT = cp.RawKernel(r'''
extern "C" __global__
void fused_resize_preprocess_transpose(
    const float* __restrict__ src,
    float* __restrict__ dst_chw,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int in_h, int in_w,
    int target_h, int target_w,
    int padded_h, int padded_w,
    int channels
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= padded_w || y >= padded_h || c >= 3) return;
    
    const float inv_255 = 0.003921568627f;
    const float inv_std_c = __frcp_rn(std[c]);
    
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
        const float one_minus_dx = 1.0f - dx;
        const float one_minus_dy = 1.0f - dy;
        
        const int src_idx_y0 = y0 * in_w * channels;
        const int src_idx_y1 = y1 * in_w * channels;
        const int src_idx_x0 = x0 * channels;
        const int src_idx_x1 = x1 * channels;
         
        const float v00 = src[src_idx_y0 + src_idx_x0 + c];
        const float v01 = src[src_idx_y0 + src_idx_x1 + c];
        const float v10 = src[src_idx_y1 + src_idx_x0 + c];
        const float v11 = src[src_idx_y1 + src_idx_x1 + c];
        
        const float top = __fmaf_rn(v01, dx, __fmaf_rn(v00, one_minus_dx, 0.0f));
        const float bottom = __fmaf_rn(v11, dx, __fmaf_rn(v10, one_minus_dx, 0.0f));
        float resized_val = __fmaf_rn(bottom, dy, __fmaf_rn(top, one_minus_dy, 0.0f));
         
        normalized = __fmaf_rn(resized_val, inv_255, -mean[c]) * inv_std_c;
    } else {
        normalized = __fmaf_rn(114.0f, inv_255, -mean[c]) * inv_std_c;
    }
     
    int dst_c = 2 - c;  // 0->2, 1->1, 2->0
     
    int chw_idx = dst_c * hw + hw_idx;
    dst_chw[chw_idx] = normalized;
}
''', 'fused_resize_preprocess_transpose')


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
void preprocess_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    const int H,
    const int W,
    const int C
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (w >= W || h >= H || c >= C)
        return;

    int input_idx = (h * W + w) * C + c;
    int output_idx = (c * H + h) * W + w;

    float px = input[input_idx];

    // Fast divide by 255
    float normalized = px * (1.0f / 255.0f);

    // Use fdividef + fmaf so the compiler stops pretending it's too busy
    float invstd = fdividef(1.0f, std[c]);

    // (normalized - mean[c]) / std[c]
    // result = (normalized - mean)*invstd
    float result = fmaf(normalized - mean[c], invstd, 0.0f);

    output[output_idx] = result;
}
''', 'preprocess_fused_kernel')



def cust_mde_nhwc_nchw(img_cp, mean, std, block_size):
    H, W, C = img_cp.shape

    img_cp = img_cp.astype(cp.float32)
    out = cp.empty((1, C, H, W), dtype=cp.float32)

    grid_x = (W + block_size[0] - 1) // block_size[0]
    grid_y = (H + block_size[1] - 1) // block_size[1]
    grid_z = (C + block_size[2] - 1) // block_size[2]

    _cust_fused_nhwc_nchw(
        (grid_x, grid_y, grid_z),
        block_size,
        (img_cp, out[0], mean, std, H, W, C)
    )

    return out



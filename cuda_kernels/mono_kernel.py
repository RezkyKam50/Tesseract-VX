import cupy as cp
import nvtx

bilinear_kernel_2c = cp.RawKernel(r'''
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
void resize_bilinear_3c(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    int channels
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
     
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
    
    const int dst_idx = (y * out_w + x) * channels + c;
    dst[dst_idx] = __fmaf_rn(bottom, dy, __fmaf_rn(top, one_minus_dy, 0.0f));
}
''', 'resize_bilinear_3c')

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
void bgr2rgb_float(
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
    int src_channel = (c == 0) ? 2 : (c == 2) ? 0 : 1;
    int src_idx = (y * w + x) * 3 + src_channel;
    
    dst[idx] = src[src_idx];
}
''', 'bgr2rgb_float')

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
void preprocess_kernel(
    const float* __restrict__ resized_img,
    float* __restrict__ padded_img,
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
    const float fill_value = 114.0f;
    
    int out_idx = (y * padded_width + x) * 3 + c;
     
    float inv_std_c = __frcp_rn(std[c]);
    
    if (y < target_height && x < target_width) {
        int in_idx = (y * target_width + x) * 3;
         
        int src_c;
        if (c == 0) src_c = 2;      // dst R <- src B
        else if (c == 1) src_c = 1; // dst G <- src G
        else src_c = 0;            // dst B <- src R
        
        float pixel_val = __ldg(&resized_img[in_idx + src_c]);
        pixel_val = __fmaf_rn(pixel_val, inv_255, -mean[c]) * inv_std_c;
        padded_img[out_idx] = pixel_val;
    } else {
        float padding_val = __fmaf_rn(fill_value, inv_255, -mean[c]) * inv_std_c;
        padded_img[out_idx] = padding_val;
    }
}
''', 'preprocess_kernel')

def cust_mot_preprocess(
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
    
    padded_img_hwc = cp.empty((padded_height, padded_width, channels), dtype=cp.float32)
     
    grid = (
        (padded_width + block_size[0] - 1) // block_size[0],
        (padded_height + block_size[1] - 1) // block_size[1],
        (channels + block_size[2] - 1) // block_size[2]
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
            target_height,             
            target_width,               
            padded_height,                
            padded_width,      
            channels                         
        )
    )
    
    return padded_img_hwc

_cust_transpose_kernel = cp.RawKernel(r'''
extern "C" __global__
void transpose_kernel(
    const float* padded_img_hwc,  
    float* padded_img_chw,         
    const int height,
    const int width,
    const int channels
) {
    // 3D thread indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;    // width
    int y = blockIdx.y * blockDim.y + threadIdx.y;    // height
    int c = blockIdx.z * blockDim.z + threadIdx.z;    // channel
    
    if (x >= width || y >= height || c >= channels) {
        return;
    }
    
    int hw = height * width;
    int hw_idx = y * width + x;
    
    // Direct assignment without loop
    int hwc_idx = hw_idx * channels + c;
    int chw_idx = c * hw + hw_idx;
    padded_img_chw[chw_idx] = padded_img_hwc[hwc_idx];
}
''', 'transpose_kernel')


def cust_mot_transpose_hwc_to_chw(
    padded_img_hwc,   
    block_size       
):
    height, width, channels = padded_img_hwc.shape
    
    assert channels == 3, "Only 3-channel images supported"

    padded_img_chw = cp.empty((channels, height, width), dtype=cp.float32)
    
    grid = (
        (width + block_size[0] - 1) // block_size[0],
        (height + block_size[1] - 1) // block_size[1],
        (channels + block_size[2] - 1) // block_size[2]
    )
     
    _cust_transpose_kernel(
        grid,
        block_size,
        (
            padded_img_hwc,   
            padded_img_chw,   
            height,          
            width,            
            channels          
        )
    )
     
    padded_img_chw = cp.ascontiguousarray(padded_img_chw)
    
    return padded_img_chw








import cv2, numpy as np

def _depth_alpha_beta(min_val, max_val):
    alpha = 255.0 / (max_val - min_val) if max_val > min_val else 1.0
    beta = -min_val * alpha
    return alpha, beta

def _count_dim(tlwh):
    return np.array([int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])], dtype=np.int32)

def _get_depth_at_box(depth_map, x, y, w, h):
 
    is_gpu = hasattr(depth_map, 'download')
    if is_gpu:
 
        if depth_map.empty():
            return np.nan
        rows, cols = depth_map.size()
    else:
 
        if depth_map.size == 0 or w <= 0 or h <= 0:
            return np.nan
        rows, cols = depth_map.shape[0], depth_map.shape[1]
     
    x = max(0, min(x, cols - 1))
    y = max(0, min(y, rows - 1))
    w = min(w, cols - x)
    h = min(h, rows - y)
    
    if w <= 0 or h <= 0:
        return np.nan
    
    if is_gpu:
        stream = cv2.cuda_Stream()
         
        depth_cpu = depth_map.download(stream)
        stream.waitForCompletion()
         
        region = depth_cpu[y:y+h, x:x+w]
    else:
        region = depth_map[y:y+h, x:x+w]
    
    total = np.sum(region)
    count = region.size
    
    return total / count if count > 0 else np.nan

def _count_vertical(w, h, threshold):
    return w / h > threshold

def _calculate_fps(current_time, start_time):
    return 1.0 / ((current_time - start_time) + 1e-6)




import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mde.mde_engine import TRTInference
 
trt_model_path = 'trt_models/depth_anything_vitl-fp16.engine'  
trt_inference = TRTInference(trt_model_path)
 
raw_img = cv2.imread('./assets/examples/demo13.jpg')
depth = trt_inference.infer(raw_img)
 
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_uint8 = depth_normalized.astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

cv2.imshow("Input Image", raw_img)
cv2.imshow("Depth Map", depth_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
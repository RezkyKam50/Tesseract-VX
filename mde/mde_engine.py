# TRT Inference Engine for Depth-Anything-V2.
# This code is borrowed from Depth-Anything-V2 repository with minor tweaks for gpu accel.

import cv2
import numpy as np
import tensorrt as trt
import numba 
import pycuda.driver as cuda
import pycuda.autoinit

class TRT_MDE:
    
    def __init__(self, trt_path):
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        with open(trt_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': tensor_name,
                    'shape': shape
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': tensor_name,
                    'shape': shape
                })
    
    def infer(self, input_image):        
        input_shape = self.inputs[0]['shape']
        input_h = input_shape[2] if input_shape[2] > 0 else 518
        input_w = input_shape[3] if input_shape[3] > 0 else 518
  
        gpu_mat = cv2.cuda_GpuMat()

        # CPU -> GPU
        gpu_mat.upload(input_image)
         
        resized_gpu = cv2.cuda.resize(gpu_mat, (input_w, input_h))
        rgb_gpu = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGR2RGB)
         
        # GPU -> CPU 
        img = rgb_gpu.download()
        img = img.astype(np.float32) / 255.0
         
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
         
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
         
        self.context.set_input_shape(self.inputs[0]['name'], img.shape)
         
        np.copyto(self.inputs[0]['host'], img.ravel())

        # CPU -> GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
         
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
         
        self.context.execute_async_v3(stream_handle=self.stream.handle)
         
        # GPU -> CPU 
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
         
        output_shape = self.context.get_tensor_shape(self.outputs[0]['name'])
        depth = self.outputs[0]['host'].reshape(output_shape)
        depth = depth[0, 0] if depth.ndim == 4 else depth[0]
 
        depth_gpu = cv2.cuda_GpuMat()

        # CPU -> GPU
        depth_gpu.upload(depth)
        resized_depth = cv2.cuda.resize(depth_gpu, (input_image.shape[1], input_image.shape[0]))

        # GPU -> CPU
        depth = resized_depth.download()
 
        
        return depth
import cv2
import cupy as cp
import tensorrt as trt
import nvtx
import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit

class TRT_MDE:
    def __init__(self, trt_path):
        self.logger = trt.Logger(trt.Logger.WARNING) 
        with open(trt_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.cv_stream = cv2.cuda.Stream()

        self.event = cuda.Event()
        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32)
         
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype, mem_flags=cuda.host_alloc_flags.WRITECOMBINED)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': tensor_name,
                    'shape': shape,
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': tensor_name,
                    'shape': shape,
                    'dtype': dtype
                })
         
        self.input_shape = self.inputs[0]['shape']
        self.input_h = self.input_shape[2] if self.input_shape[2] > 0 else 518
        self.input_w = self.input_shape[3] if self.input_shape[3] > 0 else 518
          
        
        
    def _resize(self, image):
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(image, stream=self.cv_stream)
        resized = cv2.cuda.resize(gpu_mat, (self.input_w, self.input_h), stream=self.cv_stream)
        rgb = cv2.cuda.cvtColor(resized, cv2.COLOR_BGR2RGB, stream=self.cv_stream)
        self.cv_stream.waitForCompletion()
        return rgb
    
    def _preprocess(self, gpu_mat):
        img_np = gpu_mat.download(stream=self.cv_stream)
        self.cv_stream.waitForCompletion()
         
        img_cp = cp.asarray(img_np)
        img_cp = img_cp.astype(cp.float16) / 255.0
        img_cp = (img_cp - self.mean) / self.std
        img_cp = cp.transpose(img_cp, (2, 0, 1))
        img_cp = cp.expand_dims(img_cp, axis=0)
        
        return img_cp
    
    def infer(self, input_image):
        nvtx.push_range("MDE Inference", color="red")
        
        try:
 
            rgb_gpu = self._resize(input_image)
             
            self.cv_stream.waitForCompletion()
             
            img_cp = self._preprocess(rgb_gpu)
            img_flat = img_cp.ravel()
             
            img_gpuarray = gpuarray.to_gpu_async(
                img_flat.get(), 
                stream=self.stream
            )
            
            cuda.memcpy_dtod_async(
                self.inputs[0]['device'],
                img_gpuarray.ptr,
                img_flat.nbytes,
                self.stream
            )
             
            for inp in self.inputs:
                self.context.set_tensor_address(inp['name'], int(inp['device']))
            for out in self.outputs:
                self.context.set_tensor_address(out['name'], int(out['device']))
             
            self.context.execute_async_v3(stream_handle=self.stream.handle)
             
            cuda.memcpy_dtoh_async(
                self.outputs[0]['host'], 
                self.outputs[0]['device'], 
                self.stream
            )
             
            self.stream.synchronize()
             
            output_shape = self.context.get_tensor_shape(self.outputs[0]['name'])
            depth = self.outputs[0]['host'].reshape(output_shape)
             
            if depth.ndim == 4:
                depth = depth[0, 0]
            elif depth.ndim == 3:
                depth = depth[0]
             
            depth_gpu = cv2.cuda_GpuMat()
            depth_gpu.upload(depth, stream=self.cv_stream)
             
            original_h, original_w = input_image.shape[:2]
            resized_depth_gpu = cv2.cuda.resize(
                depth_gpu, 
                (original_w, original_h),
                stream=self.cv_stream
            )
             
            depth_result = resized_depth_gpu.download(stream=self.cv_stream)
            self.cv_stream.waitForCompletion()
            
        finally:
            nvtx.pop_range()
        
        return depth_result
 
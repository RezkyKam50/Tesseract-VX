import cupy as cp
import cv2
from cuda_kernels.mono_kernel import *
from cuda_kernels.fused_kernel import *
import tensorrt as trt
from nvtx import push_range, pop_range

import pycuda.driver as cuda
import pycuda.autoinit


class TRT_MDE:
    def __init__(self, trt_path):
        self.logger = trt.Logger(trt.Logger.WARNING) 
        with open(trt_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.gpu_block2c = (32, 16) # for older architecture series < 20 use 16, 16
        self.gpu_block3c = (32, 16, 1)  

        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32)
         
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_bindings = []

        self.event = cuda.Event()
        self.stream = cuda.Stream()
        self.stream_ptr = self.stream.handle

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
                self.output_bindings.append(int(device_mem))
         
        self.input_shape = self.inputs[0]['shape']
        self.input_h = self.input_shape[2] if self.input_shape[2] > 0 else 518
        self.input_w = self.input_shape[3] if self.input_shape[3] > 0 else 518

    def _resize(self, img_cp, fused):
        push_range("Resize Kernel Func.")
        if fused:
            rgb_cp = fused_resize_bgr2rgb_3c(
                img_cp, 
                self.input_h, 
                self.input_w, 
                self.gpu_block3c
            )
        else:
            resized_cp = cupy_resize_3c(
                img_cp, 
                self.input_w, 
                self.input_h, 
                self.gpu_block3c
            )
            self.cp_wait() # this kernel is superfast lul, gotta sync so it didnt output garbage below

            rgb_cp = cupy_cvt_bgr2rgb_float(
                resized_cp, 
                self.gpu_block3c
            )
 
        pop_range()
        return rgb_cp
    
    def _preprocess(self, img_cp, fused):
        push_range("Preprocess Kernel Func.")
        if fused:
            img_cp = cust_mde_nhwc_nchw(
                img_cp,
                self.mean,
                self.std,
                self.gpu_block3c
            )
            self.cp_wait()
        else:
            img_cp = img_cp / 255.0
            img_cp = (img_cp - self.mean) / self.std
            img_cp = cp.transpose(img_cp, (2, 0, 1))
            img_cp = cp.expand_dims(img_cp, axis=0)
            pop_range()
        return img_cp
    
    def _trt2cp2trt(self, output_shape):
        push_range("Cupy Wrapper for TRT CTX Func.")
        output_mem = cp.cuda.UnownedMemory(
            int(self.outputs[0]['device']),
            self.outputs[0]['host'].nbytes,
            owner=None
        )
        output_ptr = cp.cuda.MemoryPointer(output_mem, 0)
        depth_cp = cp.ndarray(
            output_shape,
            dtype=self.outputs[0]['dtype'],
            memptr=output_ptr
        )
        pop_range()
        return depth_cp

    def _postprocess(self, input_image, depth_cp):
        push_range("Postprocess Func.")
        depth_cp = cp.ascontiguousarray(depth_cp.astype(cp.float32)) # returns h, w
        original_h, original_w = input_image.shape[:2]
        depth_resized_cp = cupy_resize_2c(depth_cp, original_h, original_w, self.gpu_block2c)
        pop_range()
        return cp.asnumpy(depth_resized_cp)
    
    def cp_wait(self):
        cp.cuda.get_current_stream().synchronize()  

    def infer(self, input_image: cp.asnumpy):

        push_range("MDE Inference", color="red")

        # check if 'input_image' is cupy or numpy array
        # directly feeding cupy array can minimize latency
        if isinstance(input_image, cp.ndarray):
            img_cp = input_image
        else:
            img_cp = cp.asarray(input_image)
 

        rgb_cp = self._resize(img_cp, fused=False)
        img_cp = self._preprocess(rgb_cp, fused=False)

        img_flat = img_cp.ravel()
            
        self.context.set_tensor_address(self.inputs[0]['name'], img_flat.data.ptr)

        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], self.output_bindings[i])
            
        self.context.execute_async_v3(stream_handle=self.stream_ptr)
        output_shape = self.context.get_tensor_shape(self.outputs[0]['name'])
 

        depth_cp = self._trt2cp2trt(output_shape)
        depth_result = self._postprocess(input_image, depth_cp)
 

        pop_range()

         
        return depth_result
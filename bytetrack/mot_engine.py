# Predictor class for ByteTrack MOT
import cupy as cp

from cuda_kernels.fused_kernel import *
from cuda_kernels.mono_kernel import *

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import torch
from postprocess import postproc

from nvtx import push_range, pop_range

class TRT_MOT:
    def __init__(self, model, exp, device, trt_file):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_file, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.decoder = model.head.decode_outputs   
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size

        print(f"self.num_classes {self.num_classes}")
        print(f"self.confthre {self.confthre}")
        print(f"self.nmsthre {self.nmsthre}")
        print(f"self.test_size {self.test_size}")
        self.device = device

        self.gpu_block2c = (32, 16) # for older architecture series < 20 use 16, 16
        self.gpu_block3c = (32, 16, 1)  

        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32)
        
        with torch.no_grad():
            dummy_input = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            _ = model(dummy_input)  # this initializes model.head.hw and model.head.strides
        
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

    def preproc(self, image, input_size, fused):

        padded_img = cp.ones((input_size[0], input_size[1], 3)) * 114.0

        padded_height, padded_width = input_size

        img = cp.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])

        target_height = int(img.shape[0] * r)
        target_width = int(img.shape[1] * r)
        
        if fused:
            padded_img = cust_mot_resize_preprocess_chwT(
                img,
                target_height,       
                target_width,       
                padded_height,   
                padded_width,     
                self.mean,
                self.std,
                self.gpu_block3c
            )

        else:
            resized_img = cupy_resize_3c(
                img, 
                target_width, 
                target_height, 
                self.gpu_block3c
            ) 

            padded_img = cust_mot_preprocess(
                resized_img,
                target_height,
                target_width,
                padded_height,
                padded_width,
                self.mean,
                self.std,
                self.gpu_block3c
            )

            padded_img = cust_mot_transpose_hwc_to_chw(
                padded_img,
                self.gpu_block3c
            )

        
        return padded_img, r

    def _trt2cp(self, output_idx=0):
        push_range("TRT to CuPy Wrapper")
        output_mem = cp.cuda.UnownedMemory(
            int(self.outputs[output_idx]['device']),
            self.outputs[output_idx]['host'].nbytes,
            owner=None
        )
        output_ptr = cp.cuda.MemoryPointer(output_mem, 0)
        output_shape = self.context.get_tensor_shape(self.outputs[output_idx]['name'])
        output_cp = cp.ndarray(
            output_shape,
            dtype=self.outputs[output_idx]['dtype'],
            memptr=output_ptr
        )
        pop_range()
        return output_cp
    
    def cp_wait(self):
        cp.cuda.get_current_stream().synchronize()
    
    def infer(self, img, timer):

        img_info = {"id": 0, "file_name": None}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
          
        # preproc  
        img, ratio = self.preproc(img, self.test_size,  fused=False)

        img_info["ratio"] = ratio
             
        img_cp = cp.asarray(img, dtype=cp.float32)
        if img_cp.ndim == 3:
            img_cp = cp.expand_dims(img_cp, axis=0)
         
        img_flat = cp.ascontiguousarray(img_cp).ravel()
            
        self.context.set_tensor_address(self.inputs[0]['name'], img_flat.data.ptr)
            
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], self.output_bindings[i])
            
        self.context.execute_async_v3(stream_handle=self.stream_ptr)
             
        outputs_cp = self._trt2cp(output_idx=0)
 
         
        outputs = torch.as_tensor(outputs_cp, device=self.device)
        outputs = self.decoder(outputs, dtype=outputs.type())
 
        # postproc
        outputs = postproc(
            outputs, 
            self.num_classes, 
            self.confthre, 
            self.nmsthre
        )
          
        return outputs, img_info

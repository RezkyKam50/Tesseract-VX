#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 

import torch
import torchvision
import cupy as cp

def findbox(prediction):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    return output

corner_kernel = cp.RawKernel(r'''
extern "C" __global__
void get_pred_corner_kernel(const float* pred_in, float* pred_out, 
                           int batch_size, int num_boxes, int num_features) {
    // x thread: batch index
    // y thread: box index  
    // z thread: coordinate index (0-3 for x1,y1,x2,y2)
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (b >= batch_size || n >= num_boxes || c >= 4) return;
    
    int base_idx = b * num_boxes * num_features + n * num_features;
    
    float cx = pred_in[base_idx + 0];
    float cy = pred_in[base_idx + 1];
    float w = pred_in[base_idx + 2];
    float h = pred_in[base_idx + 3];
    
    float result;
    if (c == 0) {
        result = __fmaf_rn(w, -0.5f, cx);  // cx - w/2 as (-0.5*w + cx)
    } else if (c == 1) {
        result = __fmaf_rn(h, -0.5f, cy);  // cy - h/2
    } else if (c == 2) {
        result = __fmaf_rn(w, 0.5f, cx);   // cx + w/2
    } else {
        result = __fmaf_rn(h, 0.5f, cy);   // cy + h/2
    }                     

    pred_out[base_idx + c] = result;
}
''', 'get_pred_corner_kernel')


def get_pred_corner(prediction: torch.tensor, block: tuple):

    prediction = cp.asarray(prediction, copy=False)

    batch_size, num_boxes, num_features = prediction.shape
     
    pred_out = cp.empty_like(prediction)
     
    if num_features > 4:
        pred_out[:, :, 4:] = prediction[:, :, 4:]
     
     
    blocks = (
        (batch_size + block[0] - 1) // block[0],
        (num_boxes + block[1] - 1) // block[1],
        (4 + block[2] - 1) // block[2]
    )
     
    corner_kernel(
        blocks,
        block,
        (prediction, pred_out, batch_size, num_boxes, num_features)
    )
    cp.cuda.stream.get_current_stream().synchronize()
     
    prediction[:, :, :4] = pred_out[:, :, :4]
    output = [None for _ in range(len(prediction))]
    return output


 
# class_conf, class_pred = torch.max(
#     image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
# )
# conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
# detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
# detections = detections[conf_mask]

detection_kernel = cp.RawKernel(r'''
__global__ void build_detections_kernel(
    const float* image_pred, 
    float* class_confs,
    int* class_idxs,
    int num_boxes, 
    int num_classes
) {
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= num_boxes) return;
    
    int base_idx = box_idx * (5 + num_classes);
     
    float max_conf = -INFINITY;
    int max_idx = 0;
    
    #pragma unroll 4                            
    for (int c = 0; c < num_classes; c++) {
        float conf = image_pred[base_idx + 5 + c];
        if (conf > max_conf) {
            max_conf = conf;
            max_idx = c;
        }
    }
     
    class_confs[box_idx] = max_conf;
    class_idxs[box_idx] = max_idx;
}
''', 'build_detections_kernel')


def build_detections(image_pred, num_classes, conf_thre):     
    class_conf, class_pred = torch.max(image_pred[:, 5:5+num_classes], 1)
     
    scores = image_pred[:, 4] * class_conf
     
    mask = scores >= conf_thre
    
    if mask.any():
        detections = torch.stack([
            image_pred[mask, 0],  # x1
            image_pred[mask, 1],  # y1
            image_pred[mask, 2],  # x2
            image_pred[mask, 3],  # y2
            image_pred[mask, 4],  # obj_conf
            class_conf[mask],     # class_conf
            class_pred[mask].float()  # class_idx
        ], dim=1)
        return detections
    
    return torch.empty((0, 7), device=image_pred.device)


def postproc(prediction: torch.tensor, num_classes: int, conf_thre: float, nms_thre:float, block: tuple):

    torch.cuda.synchronize() 
    output = get_pred_corner(prediction, block)
    
    for i, image_pred in enumerate(prediction):
 
        if not image_pred.size(0):
            continue
    
        detections = build_detections(image_pred, num_classes, conf_thre)

        if not detections.size(0):
            continue
 
        torch.cuda.synchronize() 

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


 

 
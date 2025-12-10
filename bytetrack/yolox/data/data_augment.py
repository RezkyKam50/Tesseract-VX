#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import cv2
import numpy as np, cupy as cp

from yolox.utils import xyxy2cxcywh

import math
import random

import numba


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    # Convert to GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    
    # Convert BGR to HSV on GPU
    hsv_gpu = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HSV)
    
    # Split channels on GPU
    h_gpu, s_gpu, v_gpu = cv2.cuda.split(hsv_gpu)
    
    # Download to CPU for LUT operations
    hue = h_gpu.download()
    sat = s_gpu.download()
    val = v_gpu.download()
    
    dtype = img.dtype  # uint8

    lut_hue, lut_sat, lut_val = find_hsvr(hgain, sgain, vgain, dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    
    # Upload to GPU for conversion
    hsv_gpu = cv2.cuda_GpuMat()
    hsv_gpu.upload(img_hsv)
    result_gpu = cv2.cuda.cvtColor(hsv_gpu, cv2.COLOR_HSV2BGR)
    result_gpu.download(img)



@numba.njit(parallel=True, cache=True)
def find_hsvr(hgain, sgain, vgain, dtype):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    return lut_hue, lut_sat, lut_val


@numba.njit(parallel=True, cache=True)
def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


@numba.njit(parallel=True, cache=True)
def rotation_mat(img, border, degrees, scale, shear, translate):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(scale[0], scale[1])
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    return height, width, s, M

def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):

    height, width, s, M = rotation_mat(img, border, degrees, scale, shear, translate)

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
 
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        
        if perspective:
            gpu_warped = cv2.cuda.warpPerspective(
                gpu_img, M, (width, height), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114)
            )
        else:  # affine
            gpu_warped = cv2.cuda.warpAffine(
                gpu_img, M[:2], (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114)
            )
         
        img = gpu_warped.download()
 
    n = len(targets)
    if n:
        targets = find_cords(width, height, perspective, M, n, s)
    return img, targets


@numba.njit(parallel=True, cache=True)
def find_cords(width, height, perspective, M, n, s):
    # warp points
    xy = np.ones((n * 4, 3))
    xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        n * 4, 2
    )  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    if perspective:
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
    else:  # affine
        xy = xy[:, :2].reshape(n, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

    # clip boxes
    #xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
    #xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

    # filter candidates
    i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
    targets = targets[i]
    targets[:, :4] = xy[i]
    
    targets = targets[targets[:, 0] < width]
    targets = targets[targets[:, 2] > 0]
    targets = targets[targets[:, 1] < height]
    targets = targets[targets[:, 3] > 0]

    return targets

@numba.njit(cache=True)
def _convert(image, alpha=1, beta=0):
    tmp = image.astype(float) * alpha + beta
    tmp[tmp < 0] = 0
    tmp[tmp > 255] = 255
    image[:] = tmp

def _distort(image):
    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    # Use GPU for color conversion
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    hsv_gpu = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HSV)
    image = hsv_gpu.download()

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    # Convert back to BGR
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    bgr_gpu = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_HSV2BGR)
    image = bgr_gpu.download()

    return image


@numba.njit(parallel=True, cache=True)
def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


# def preproc(image, input_size, mean, std, swap=(2, 0, 1)):

#     img, r, padded_img = find_r(input_size, image)

#     gpu_img = cv2.cuda_GpuMat()
#     gpu_img.upload(img)
    
#     new_width = int(img.shape[1] * r)
#     new_height = int(img.shape[0] * r)
    
#     # Resize on GPU
#     gpu_resized = cv2.cuda.resize(gpu_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#     resized_img = gpu_resized.download()
    
#     padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
#     padded_img = padded_img[:, :, ::-1]
#     padded_img /= 255.0
#     if mean is not None:
#         padded_img -= mean
#     if std is not None:
#         padded_img /= std
#     padded_img = padded_img.transpose(swap)
#     padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
 
#     return padded_img, r


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):

    if len(image.shape) == 3:
        padded_img = cp.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = cp.ones(input_size) * 114.0

    img = cp.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])

    target_height = int(img.shape[0] * r)
    target_width = int(img.shape[1] * r)

    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Invalid target size: ({target_width}, {target_height})")

    resized_img = cp.array(cv2.resize(
        cp.asnumpy(img),
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32))

    if len(image.shape) == 3:
        padded_img[:target_height, :target_width, :] = resized_img
    else:
        padded_img[:target_height, :target_width] = resized_img

    padded_img = padded_img[:, :, ::-1] / 255.0  # BGR to RGB and normalize

    if mean is not None:
        mean_array = cp.array(mean).reshape(1, 1, 3)
        padded_img -= mean_array

    if std is not None:
        std_array = cp.array(std).reshape(1, 1, 3)
        padded_img /= std_array

    padded_img = padded_img.transpose(swap)
    padded_img = cp.ascontiguousarray(padded_img, dtype=cp.float32)
    
    return padded_img, r



def find_r(input_size, image):
    if len(image.shape) == 3:
        return find_r_3d(input_size, image)
    else:
        return find_r_2d(input_size, image)

@numba.njit(cache=True, parallel=True)
def find_r_2d(input_size, image):
    h, w = image.shape
    padded_img = np.ones(input_size, dtype=np.float64) * 114.0
    r = min(input_size[0] / h, input_size[1] / w)
    return image.copy(), r, padded_img

@numba.njit(cache=True, parallel=True)
def find_r_3d(input_size, image):
    h, w, c = image.shape
    padded_img = np.ones((input_size[0], input_size[1], c), dtype=np.float64) * 114.0
    r = min(input_size[0] / h, input_size[1] / w)
    return image.copy(), r, padded_img




class TrainTransform:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=100):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        ids = targets[:, 5].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        ids_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        image_t = _distort(image)
        image_t, boxes = _mirror(image_t, boxes)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        ids_t = ids[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            ids_t = ids_o

        labels_t = np.expand_dims(labels_t, 1)
        ids_t = np.expand_dims(ids_t, 1)

        targets_t = np.hstack((labels_t, boxes_t, ids_t))
        padded_labels = np.zeros((self.max_labels, 6))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 5))
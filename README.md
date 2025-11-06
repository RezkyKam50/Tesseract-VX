**Pre-recorded TRT Inference w/ Proximity Alert**

![Demo GIF](demo/demo.gif)

> Left: Object Track, Right: Depth Estimation

**Fedora 42 Setup**

> notice: run every script (.py, .sh) from parents directory.

Start by compiling OpenCV from source first, this ensures you have CUDA enabled CV libraries .**You need the latest CUDA Toolkit (13.0) for this**

> sudo chmod +x ./scripts/gcc_switcher.sh

You need gcc-14 and g++-14 on your system wide environment

> ./scripts/gcc_switcher.sh

Source the exact version 13.0 CUDA Toolkit

> sudo chmod +x ./scripts/cuda_toolkit.sh

Build OpenCV from source with CUDA enabled

> sudo chmod +x ./scripts/build_cv_cuda.sh
> ./scripts/build_cv_cuda.sh

Depending on your Distro, install "uv" package manager systemwide, this is crucial for dependencies resolving.

> sudo chmod +x ./scripts/setup_dav-2.sh
> ./scripts/setup_dav-2.sh

> sudo chmod +x ./scripts/setup_tensorrt.sh
> ./scripts/setup_tensorrt.sh

Under 'checkpoints' directory, there should be the weight of the models pulled from HF **fresh ovenbaked**.
The default script is hardcoded for 'vitl' for Large size.

Convert HF to ONNX

> python3 ./utils/onnx_conv.py

Convert ONNX to TensorRT (.engine)

> python3 ./utils/onnx_tensorrt.py

Now, the MDE model is fresh ovenbaked, lets continue to baking the ByteTracker model.

> sudo chmod +x ./src/bytetrack/pretrained/install_mot.sh

Download from ByteTrack repository

> ./src/bytetrack/pretrained/install_mot.sh

convert from .pth to TRT .pth (this will produce .pth and .engine) but we'll use the .pth under './src/bytetrack/trt_models'

> python3 ./src/bytetrack/utils/conv_trt.sh

Check first that theres model in each of those folders and every step runs smoothly without error.
Now run:

> sudo chmod +x spectralgraph.sh && ./spectralgraph.sh

# Citation

```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}

@misc{yang2024depthv2,
      title={Depth Anything V2}, 
      author={Lihe Yang and Bingyi Kang and Zilong Huang and Zhen Zhao and Xiaogang Xu and Jiashi Feng and Hengshuang Zhao},
      year={2024},
      eprint={2406.09414},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.09414}, 
}
```

# Acknowledgement

Special thanks to the team behind [ByteTrack](https://github.com/FoundationVision/ByteTrack) and [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2).

**TRT Inference w/ Proximity Alert (Red) (P = avg_depth > 200 )**

** *Working on bounding box collision.

![Demo GIF](demo/demo.gif)

> Left: Object Track, Right: Depth Estimation

Relative distance were simply calculated by summing the average depth value inside the bounding box.
Getting the absolute (Real) distance can be done by calibrating average depth with measures, though its quiet tricky due to difference in focal length.

**Build from source**

git clone --recurse git@github.com:RezkyKam50/Tesseract-VX.git

cd Tesseract-VX

git submodule update --init --recursive

./configure.sh

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

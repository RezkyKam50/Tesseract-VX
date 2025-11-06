import torch
import time
import sys
import os
import argparse
import glob

try:
    from .utilities import Engine
except ImportError:
    from utilities import Engine


DEFAULT_ONNX_INPUT=" "  
DEFAULT_TRT_OUTPUT=" "

def export_trt(trt_path: str, onnx_path: str, use_fp16: bool):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Onnx file doesn't exist: {onnx_path}")

    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TensorRT engine from ONNX model."
    )
    parser.add_argument(
        "--trt-path",
        type=str,
        default=f"./trt_models/{DEFAULT_TRT_OUTPUT}",
        help="Path to save the TensorRT engine file.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=f"./onnx_models/{DEFAULT_ONNX_INPUT}",
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--use-fp32",
        action="store_true",
        help="Use FP32 precision (default is FP16).",
    )
    args = parser.parse_args()

    export_trt(
        trt_path=args.trt_path, onnx_path=args.onnx_path, use_fp16=not args.use_fp32
    )
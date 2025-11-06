from loguru import logger
import tensorrt as trt
import torch
from torch2trt import torch2trt
from yolox.exp import get_exp
import os
import argparse

# this script produces .engine and .pth (torch2trt)

@logger.catch
def main(args):
    MODEL_NAME = args.model_name
    EXP_FILE = args.exp_file
    EXP_NAME = args.exp_name
    CKPT_PATH = args.ckpt_path
    OUTPUT_DIR = args.output_dir
    FP16_MODE = args.fp16

    exp = get_exp(EXP_FILE, EXP_NAME)
    experiment_name = EXP_NAME

    model = exp.get_model()
    file_name = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(file_name, exist_ok=True)

    ckpt_file = CKPT_PATH
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint successfully.")

    model.eval()
    model.cuda()
    model.head.decode_in_inference = False

    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
    logger.info("Converting model to TensorRT...")

    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=FP16_MODE, 
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << 32),
    )

    trt_pth = os.path.join(file_name, f"{MODEL_NAME}.pth")
    torch.save(model_trt.state_dict(), trt_pth)
    logger.info(f"Saved TensorRT .pth at {trt_pth}")

    engine_file = os.path.join(file_name, f"{MODEL_NAME}.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())
    logger.info(f"Saved TensorRT .engine at {engine_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLOX model to TensorRT")
    
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="bytetrack_tiny_mot17",
        help="Name of the model (used for output files)"
    )
    
    parser.add_argument(
        "--exp-file", 
        type=str, 
        default="./src/bytetrack/exp/yolox_tiny_mix_det.py",
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--exp-name", 
        type=str, 
        default="yolox_tiny_mix_det",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--ckpt-path", 
        type=str, 
        default="./src/bytetrack/pretrained/bytetrack_tiny_mot17.pth.tar",
        help="Path to checkpoint file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./trt_models/",
        help="Output directory for TensorRT models"
    )
    
    parser.add_argument(
        "--fp16", 
        action="store_true",
        default=True,
        help="Enable FP16 mode for TensorRT (default: True)"
    )
    
    parser.add_argument(
        "--no-fp16", 
        dest="fp16",
        action="store_false",
        help="Disable FP16 mode for TensorRT"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
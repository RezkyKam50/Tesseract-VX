import argparse
import torch
import torch.onnx
import glob
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mde.depth_anything_v2.dpt import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    # default to Large

    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    args = parser.parse_args()
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    path = glob.glob(f'./checkpoints/Depth-Anything-V2-*/depth_anything_v2_{args.encoder}.pth')[0]
    depth_anything.load_state_dict(torch.load(f'{path}', map_location='cpu'))
    depth_anything = depth_anything.to('cpu').eval()
 
    dummy_input = torch.ones((3, args.input_size, args.input_size)).unsqueeze(0)
 
    example_output = depth_anything.forward(dummy_input)

    onnx_path = f'./onnx_models/depth_anything_v2_{args.encoder}.onnx'
 
    torch.onnx.export(depth_anything, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"], verbose=True)

    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    main()
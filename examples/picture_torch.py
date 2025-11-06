import cv2
import torch
import cupy as cp
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mde.depth_anything_v2.dpt import DepthAnythingV2

os.environ["QT_QPA_PLATFORM"] = "xcb"

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
 
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/Depth-Anything-V2-Large/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
 
raw_img = cv2.imread('./assets/examples/demo13.jpg')
depth = model.infer_image(raw_img)  

depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_uint8 = depth_normalized.astype(cp.uint8)
 
depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
 
cv2.imshow("Input Image", raw_img)
cv2.imshow("Depth Map", depth_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()


 
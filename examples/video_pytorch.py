import cv2
import torch
import numpy as np
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

encoder = 'vitb'   
checkpoint_path = f'checkpoints/Depth-Anything-V2-Base/depth_anything_v2_{encoder}.pth'
 
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model = model.to(DEVICE).bfloat16().eval()
 
video_path = './assets/examples_video/ferris_wheel.mp4'
cap = cv2.VideoCapture(video_path)
 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('depth_output.mp4', fourcc, fps, (width, height))
 
target_w, target_h = 518*2, 384*2

print("Running depth inference... press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    frame_resized = cv2.resize(frame, (target_w, target_h))
 
    with torch.amp.autocast(enabled=True, device_type=DEVICE, dtype=torch.bfloat16):
        depth = model.infer_image(frame_resized)
 
    depth = cv2.resize(depth, (width, height))
 
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
 
    combined = cv2.hconcat([frame, depth_colored])
    cv2.imshow("Depth Video", combined)
 
    out.write(depth_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

 

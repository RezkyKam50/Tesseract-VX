class AppArgs:
    WINDOW_NAME = "Tesseract-VX"
    CUDA_DEVICE = 0
    QT_PLATFORM = "xcb"     # options: "xcb", "offscreen"
    NV_PRIME    = False     # for hybrid boards (NVIDIA Optimus)
    GLX_VENDOR  = "nvidia"  # options: "nvidia"
    CP_TF32     = 1         # Enable Cupy TF32 mode

class ModelArgs:
    # ByteTrack Model
    EXP_FILE = "./bytetrack/exp/yolox_l_mix_det.py"
    MOT_PATH = "./trt_models/bytetrack/bytetrack_l.engine"
    # MDE Model
    MDE_PATH = './trt_models/DAV2/depth_anything_v2_vitl.engine'

class TrackArgs:

    # Functional
    TRACK_THRESH = 0.5
    TRACK_BUFFER = 10
    PROXIMITY_THRESH = 320
    MATCH_THRESH = 0.9
    ASPECT_RATIO_THRESH = 1.6
    MIN_BOX_AREA = 10
    FRAME_RATE = 30

    # Misc.
    HIGHLIGHT_ALPHA = 0.1   
    BORDER_THICKNESS = 1

class FootageArgs:
    OUTPUT_DIR          = "./output_videos"
    DEFAULT_OUTPUT_NAME = "tracked_output.mp4"
    VIDEO_CODEC         = "mp4v"  # or "avc1" for H.264

class FontConfig:
    import cv2
    FONT_FACE               = cv2.FONT_HERSHEY_DUPLEX
    FPS_SCALE               = 1.0
    TRACKER_INFO_SCALE      = 0.7
    DEPTH_TEXT_SCALE        = 0.6
    FPS_THICKNESS           = 1
    TRACKER_INFO_THICKNESS  = 1
    DEPTH_TEXT_THICKNESS    = 1
    FPS_COLOR               = (0, 255, 0)  # Green
    TRACKER_INFO_COLOR      = (255, 255, 255)
    THEMECOLORS             = [
        (255, 255, 255),    # White
        # (255, 0, 0),      # Blue
        # (0, 0, 255),      # Red
        # (255, 255, 0),    # Cyan
        # (255, 0, 255),    # Magenta
        # (0, 255, 255),    # Yellow
        # (128, 255, 0),    # Light Green
        # (255, 128, 0),    # Orange
    ]
    PROXIMITY_COLOR = [
        (0, 0, 255)      # Red 
    ]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="ByteTrack with Depth Estimation")
    parser.add_argument(
        "--source", 
        type=str, 
        default="0",
        help="Video source: camera ID (e.g., 0) or path to MP4 file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: auto-generated in output_videos/)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying video (faster processing)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output video"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable CPU Parallelism"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable LLVM"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU Offloading"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging"
    )


    return parser.parse_args()


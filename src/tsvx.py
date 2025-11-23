from __future__ import annotations

import os, os.path as osp, time, sys
import argparse
from pathlib import Path
 
from loguru import logger
 
import torch, cv2
import cupy as cp, numpy as np, numba

from bytetrack.yolox.exp                  import get_exp
from bytetrack.yolox.utils                import get_model_info
from bytetrack.yolox.tracker.byte_tracker import BYTETracker
from bytetrack.yolox.tracking_utils.timer import Timer
 
from bytetrack.mot_engine import TORCH_MOT
from mde.mde_engine       import TRT_MDE

import pycuda.driver as cuda
import pycuda.autoinit

from tsvx_args import (
    AppArgs, ModelArgs, TrackArgs, FootageArgs, FontConfig, parse_args)


class Initialize:
    '''
    Init for Flight-check
    '''
    def __init__(
            self, 
            source, 
            output_path, 
            cuda_device, 
            qt_platform,
            nv_prime,
            glx_vendor
        ):

        self.source      = source
        self.output_path = output_path

        os.environ["QT_QPA_PLATFORM"]           = f"{qt_platform}"
        os.environ["CUDA_VISIBLE_DEVICES"]      = f"{cuda_device}"
        os.environ["NV_PRIME_RENDER_OFFLOAD"]   = "1" if nv_prime else "0"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = f"{glx_vendor}"

    def check_cuda_support(self):
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"OpenCV CUDA support: {cuda_available}")

        if cuda_available and torch.cuda.is_available():
            print(f"Found CUDA device/s: {torch.cuda.device_count()}")
        else:
            print("One of OpenCV or PyTorch CUDA support is unavailable. Exiting...")
            sys.exit(1)

        return cuda_available

    def initialize_video_source(self, debugging):
        try:
            device_id = int(self.source)
            cap = cv2.VideoCapture(device_id)
            source_type = "camera"
            logger.info(f"Using camera device: {device_id}")
        except ValueError:
            if not osp.exists(self.source):
                raise FileNotFoundError(f"Video file not found: {self.source}")
            cap = cv2.VideoCapture(self.source)
            source_type = "video"
            logger.info(f"Using video file: {self.source}")
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        fps             = cap.get(cv2.CAP_PROP_FPS)
        width           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if debugging is True and isinstance(self.source, str) and (os.path.sep in self.source or '/' in self.source):
            logger.info(f"Video properties: {width}x{height} @ {fps:.2f} fps")
            if source_type == "video":
                logger.info(f"Total frames: {total_frames}")
            
        return cap, source_type, fps, width, height, total_frames

    def initialize_video_writer(self):

        output                       = self.generate_output_path()
        _, _, fps, width, height, _  = self.initialize_video_source(self.source)

        output_dir = osp.dirname(output)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        combined_width = width * 2
        
        fourcc = cv2.VideoWriter_fourcc(*FootageArgs.VIDEO_CODEC)
        writer = cv2.VideoWriter(output, fourcc, fps, (combined_width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output}")
        
        logger.info(f"Saving output to: {output}")
        logger.info(f"Output resolution: {combined_width}x{height} @ {fps:.2f} fps")
        
        return writer

    def generate_output_path(self):
        if self.output_path:
            return self.output_path
        os.makedirs(FootageArgs.OUTPUT_DIR, exist_ok=True)
        try:
            int(self.source)
            base_name = f"camera_{self.source}"
        except ValueError:
            base_name = Path(self.source).stem
        
        timestamp   = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"{base_name}_{timestamp}.mp4"
        
        return osp.join(FootageArgs.OUTPUT_DIR, output_name)


class LoadModel:
    '''
    TSVX MDE & MOT Loader, Consist of Vanilla TRT (MDE) and PyTorch TRT (MOT)
    '''
    def __init__(self):

        Initialize.check_cuda_support()

    def mde_model(self):
        return TRT_MDE(ModelArgs.MDE_PATH)

    def bytetrack_model(self):
        device  = torch.device("cuda")

        logger.info(f"Using device: {device}")
        logger.info("Loading ByteTrack model configuration...")

        exp     = get_exp(ModelArgs.EXP_FILE, None)
        model   = exp.get_model().to(device)

        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        model.eval()
        assert osp.exists(ModelArgs.MOT_PATH), f"TensorRT model not found at {ModelArgs.MOT_PATH}"
        logger.info("Using TensorRT for ByteTrack inference")
        model.head.decode_in_inference = False

        decoder     = model.head.decode_outputs
        predictor   = TORCH_MOT(model, exp, device, ModelArgs.MOT_PATH, decoder, fp16=False)
        tracker     = BYTETracker(frame_rate=TrackArgs.FRAME_RATE)
        timer       = Timer()
        
        return predictor, tracker, timer


class DepthProcess:
    '''
    MDE Instance for RT Depth Estimation
    '''
    def __init__(self, optimize, offload):
        
        self.offload = offload
        self.depth_alpha_beta_jit = numba.jit(nopython=optimize)(self.depth_alpha_beta)

    def process_depth_map(self, depth, frame_shape, stream):

        gpu_depth               = cv2.cuda_GpuMat()
        # -- offload -> gpu     : depth obj., lane
        gpu_depth.upload(depth, stream)
        #
        depth_map, gpu_depth_resized = self.resize_depth(
            gpu_depth, frame_shape, stream)
        #
        stream.synchronize()
        min_val, max_val, _, _  = cv2.cuda.minMaxLoc(gpu_depth_resized)
        alpha, beta             = self.depth_alpha_beta_jit(min_val, max_val)

        depth_colored = self.normalize_depth(
            gpu_depth_resized, alpha, beta, stream)

        return depth_map, depth_colored

    def depth_alpha_beta(self, min_val, max_val):

        alpha   = 255.0 / (max_val - min_val) if max_val > min_val else 1.0
        beta    = -min_val * alpha

        return alpha, beta

    def resize_depth(self, gpu_depth, frame_shape, stream):
        target_size             = (frame_shape[1], frame_shape[0])
        gpu_depth_resized       = cv2.cuda.resize(gpu_depth, target_size, stream=stream)

        if self.offload is True:
            # -- offload -> cpu     : resized
            depth_map               = gpu_depth_resized.download(stream)
        else:
            depth_map               = gpu_depth_resized

        return depth_map, gpu_depth_resized

    def normalize_depth(self, gpu_depth_resized, alpha, beta, stream):
        gpu_depth_normalized    = gpu_depth_resized.convertTo(cv2.CV_8UC3, alpha, beta, stream=stream)

        if self.offload is True:
            # -- offload -> cpu : normalized
            depth_colored           = cv2.applyColorMap(
                gpu_depth_normalized.download(stream), cv2.COLORMAP_BONE)
        else:
            depth_colored           = cv2.applyColorMap(
                gpu_depth_normalized, cv2.COLORMAP_BONE)

        return depth_colored



class TrackerProcess:
    '''
    MOT Instance for RT Multi-Object Tracking
    '''
    def __init__(self, optimize, parallelism):

        self.box_depth_jit      = numba.jit(nopython=optimize, parallel=parallelism)(self.get_depth_at_box)
        self.count_dim_jit      = numba.jit(nopython=optimize)                      (self.count_dim)
        self.count_vertical_jit = numba.jit(nopython=optimize)                      (self.count_vertical)
        
    def count_dim(self, tlwh):
        return [int(v) for v in tlwh]

    def get_depth_at_box(self, depth_map, tlwh):

        x, y, w, h = self.count_dim_jit(tlwh)

        if depth_map.size == 0 or w <= 0 or h <= 0:
            return np.nan
        
        x = max(0, min(x, depth_map.shape[1] - 1))
        y = max(0, min(y, depth_map.shape[0] - 1))
        w = min(w, depth_map.shape[1] - x)
        h = min(h, depth_map.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return np.nan
        
        region = depth_map[y:y+h, x:x+w]    # vectorized ops. can be parallelized
        total = np.sum(region)
        count = region.size
        
        return total / count if count > 0 else np.nan

    def draw_transparent_highlight(self, img, x, y, w, h, color, alpha=TrackArgs.HIGHLIGHT_ALPHA, border_thickness=TrackArgs.BORDER_THICKNESS):
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, border_thickness)

    def count_vertical(tlwh):
        return tlwh[2] / tlwh[3] > TrackArgs.ASPECT_RATIO_THRESH 

    def draw_tracked_objects(self, frame, depth_colored, depth_map, online_targets):
        tracked_count = 0
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            vertical = self.count_vertical_jit(tlwh)

            if tlwh[2] * tlwh[3] > TrackArgs.MIN_BOX_AREA and not vertical:
                tracked_count += 1
                x, y, w, h = self.count_dim(tlwh)
                color = FontConfig.THEMECOLORS[tid % len(FontConfig.THEMECOLORS)]
                self.draw_transparent_highlight(frame, x, y, w, h, color)
                self.draw_transparent_highlight(depth_colored, x, y, w, h, color)

                avg_depth = self.box_depth_jit(depth_map, tlwh)

                self.render_text_overlay(avg_depth, tid, x, y, frame, depth_colored, color)
                
        return tracked_count
    
    def render_text_overlay(self, avg_depth, tid, x, y, frame, depth_colored, color):
        if avg_depth is not None:
            depth_text = f"ID: {tid} | distance: {avg_depth:.2f}"
            cv2.putText(frame, depth_text, (x, y - 10),
                    FontConfig.FONT_FACE, FontConfig.DEPTH_TEXT_SCALE,
                    color, FontConfig.DEPTH_TEXT_THICKNESS)
            cv2.putText(depth_colored, depth_text, (x, y - 10),
                    FontConfig.FONT_FACE, FontConfig.DEPTH_TEXT_SCALE,
                    color, FontConfig.DEPTH_TEXT_THICKNESS)
            
            if avg_depth > TrackArgs.PROXIMITY_THRESH:
                cv2.putText(frame, "Proximity Alert", (x, y - 25),
                        FontConfig.FONT_FACE, FontConfig.DEPTH_TEXT_SCALE,
                        (0,0,255), FontConfig.DEPTH_TEXT_THICKNESS)
            del depth_text
                
        elif avg_depth is None or (isinstance(avg_depth, float) and np.isnan(avg_depth)):
            depth_text = f"ID: {tid} | distance: N/A"
            cv2.putText(frame, depth_text, (x, y - 10),
                    FontConfig.FONT_FACE, FontConfig.DEPTH_TEXT_SCALE,
                    color, FontConfig.DEPTH_TEXT_THICKNESS)
            cv2.putText(depth_colored, depth_text, (x, y - 10),
                    FontConfig.FONT_FACE, FontConfig.DEPTH_TEXT_SCALE,
                    color, FontConfig.DEPTH_TEXT_THICKNESS)
            del depth_text
        
class TSVX:
    '''
    MOT & MDE Wrapper Instance for TesseractVX
    '''
    def __init__(self, optimize, parallelism, offload):

        self.TrackerInstance = TrackerProcess   (optimize=optimize, parallelism=parallelism)
        self.DepthInstance   = DepthProcess     (optimize=optimize, offload=offload)

        self.calculate_fps_jit = numba.jit(nopython=optimize)(self.calculate_fps)

    def create_combined_view(self, frame, depth_colored):
        return cv2.hconcat([frame, depth_colored])

    def calculate_fps(self, current_time, start_time):
        return 1.0 / ((current_time - start_time) + 1e-6)   

    def print_controls(self, source_type, save_video):
        print("\nRunning ByteTrack with Depth Estimation...")
        print(f"Source type: {source_type}")
        print(f"Saving video: {'Yes' if save_video else 'No'}")
        print("\nControls:")
        print("  'q' - quit")
        if source_type == "video":
            print("  'space' - pause/resume")

    def depth_tracking_loop(self, cap, depth_trt_inference, bytetrack_predictor, bytetrack_tracker, 
                                bytetrack_timer, stream_0, stream_1, video_writer=None, source_type="camera", 
                                total_frames=0, display=True):
        frame_id = 0
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if source_type == "video":
                        logger.info("End of video reached")
                    else:
                        logger.warning("Frame capture failed")
                    break
                
                start_time = time.time()
                depth = depth_trt_inference.infer(frame, stream_0)  
                depth_map, depth_colored = self.DepthInstance.process_depth_map(depth, frame.shape, stream_1)
                stream_1.synchronize() 

                outputs, img_info        = bytetrack_predictor.inference(frame, bytetrack_timer)
                
                tracked_count = 0

                if outputs[0] is not None:
                    online_targets = bytetrack_tracker.update(
                        outputs[0],
                        [
                        img_info['height'], 
                        img_info['width']
                        ],
                        bytetrack_predictor.test_size
                    )
                    tracked_count = self.TrackerInstance.draw_tracked_objects(frame, depth_colored, depth_map, online_targets)
                    bytetrack_timer.toc()
                else:
                    bytetrack_timer.toc()

                fps        = self.calculate_fps_jit(time.time(), start_time)

                self.draw_overlay_info(frame, fps, tracked_count, frame_id, total_frames)
                combined = self.create_combined_view(frame, depth_colored)

                if video_writer is not None:
                    video_writer.write(combined)
                
                if display:
                    cv2.imshow(AppArgs.WINDOW_NAME, combined)
                
                frame_id += 1
            
            if display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and source_type == "video":
                    paused = not paused
                    logger.info(f"Video {'paused' if paused else 'resumed'}")
            else:
                continue
        
        return tracked_count, frame_id

    def draw_overlay_info(self, frame, fps, num_trackers, frame_id=None, total_frames=None):
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                    FontConfig.FONT_FACE, FontConfig.FPS_SCALE,
                    FontConfig.FPS_COLOR, FontConfig.FPS_THICKNESS)
        cv2.putText(frame, f"Tracked: {num_trackers}", (20, 70),
                    FontConfig.FONT_FACE, FontConfig.TRACKER_INFO_SCALE,
                    FontConfig.TRACKER_INFO_COLOR, FontConfig.TRACKER_INFO_THICKNESS)
        if frame_id is not None and total_frames is not None and total_frames > 0:
            progress_text = f"Frame: {frame_id}/{total_frames}"
            cv2.putText(frame, progress_text, (20, 100),
                        FontConfig.FONT_FACE, FontConfig.TRACKER_INFO_SCALE,
                        FontConfig.TRACKER_INFO_COLOR, FontConfig.TRACKER_INFO_THICKNESS)

    def cleanup(self, cap, video_writer=None):
        cap.release()
        if video_writer is not None:
            video_writer.release()
            logger.info("Video saved successfully")
        cv2.destroyAllWindows()


if __name__ == "__main__":

    args = parse_args()

    InitInstance = Initialize(
        source      = args.source,
        output_path = args.output,
        cuda_device = AppArgs.CUDA_DEVICE,
        qt_platform = AppArgs.QT_PLATFORM,
        nv_prime    = AppArgs.NV_PRIME,
        glx_vendor  = AppArgs.GLX_VENDOR
    )
    
    MainInstance = TSVX(
        optimize    = args.optimize if args.optimize    else False,
        parallelism = args.parallel if args.parallel    else False,
        offload     = args.offload  if args.offload     else False,
        # debug       = args.debug    if args.debug else False, // TODO: add debug interface for every instances
    )
     
    depth_trt_inference = LoadModel.mde_model()
    bytetrack_predictor, bytetrack_tracker, bytetrack_timer = LoadModel.bytetrack_model()

    cap, source_type, fps, width, height, total_frames = InitInstance.initialize_video_source(args.debug)
     
    video_writer = None

    if args.save_video:
        video_writer    = InitInstance.initialize_video_writer()
    
    # lane for MDE infer
    stream_0 = cuda.Stream()
    # lane for separate depth processing after MDE infer 
    # Infer -> Depth Process -> BBProcess, haven't benchmarked the benefit of 
    # this additional stream yet.
    stream_1 = cuda.Stream()

    MainInstance.print_controls(source_type, args.save_video)
     
    final_count, total_processed = MainInstance.depth_tracking_loop(
        cap, depth_trt_inference, 
        bytetrack_predictor, bytetrack_tracker, bytetrack_timer,
        stream_0, stream_1, video_writer, source_type, total_frames,
        display=not args.no_display
    )
    
    MainInstance.cleanup(cap, video_writer)
    print(f"\nFinal stats:")
    print(f"  Total frames processed: {total_processed}")
    print(f"  Tracked objects in last frame: {final_count}")
 
        
 
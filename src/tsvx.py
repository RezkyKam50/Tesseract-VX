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

from tsvx_kernel import (
    _depth_alpha_beta, _count_dim, _get_depth_at_box, _count_vertical, _calculate_fps)

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
         
        self.cap = None
        self.source_type = None
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None

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
         
        self.cap = cap
        self.source_type = source_type
        self.fps = fps
        self.width = width
        self.height = height
        self.total_frames = total_frames
        
        if debugging is True and isinstance(self.source, str) and (os.path.sep in self.source or '/' in self.source):
            logger.info(f"Video properties: {width}x{height} @ {fps:.2f} fps")
            if source_type == "video":
                logger.info(f"Total frames: {total_frames}")
            
        return cap, source_type, fps, width, height, total_frames

    def initialize_video_writer(self):
        if self.cap is None:
            raise RuntimeError("Video source must be initialized before creating writer")
        
        output = self.generate_output_path()
        output_dir = osp.dirname(output)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        combined_width = self.width * 2
        
        fourcc = cv2.VideoWriter_fourcc(*FootageArgs.VIDEO_CODEC)
        writer = cv2.VideoWriter(output, fourcc, self.fps, (combined_width, self.height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output}")
        
        logger.info(f"Saving output to: {output}")
        logger.info(f"Output resolution: {combined_width}x{self.height} @ {self.fps:.2f} fps")
        
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
    def __init__(self, Init):

        Init.check_cuda_support()

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
        
        tracker_args = argparse.Namespace(
            track_thresh=TrackArgs.TRACK_THRESH,
            track_buffer=TrackArgs.TRACK_BUFFER,
            match_thresh=TrackArgs.MATCH_THRESH,
            aspect_ratio_thresh=TrackArgs.ASPECT_RATIO_THRESH,
            min_box_area=TrackArgs.MIN_BOX_AREA,
            mot20=False
        )
        
        tracker     = BYTETracker(tracker_args, frame_rate=TrackArgs.FRAME_RATE)
        timer       = Timer()
        
        return predictor, tracker, timer


class DepthProcess:
    '''
    MDE Instance for RT Depth Estimation
    '''
    def __init__(self, optimize, offload):
        
        self.offload = offload

        # arrange mem.
        self.gpu_depth  = cv2.cuda_GpuMat()
         
        if optimize:
            self._depth_alpha_beta_compiled = numba.jit(nopython=True, cache=True)(_depth_alpha_beta)
        else:
            self._depth_alpha_beta_compiled = _depth_alpha_beta

    def process_depth_map(self, depth, frame_shape, stream):

        # -- offload -> gpu     : depth obj., lane
        self.gpu_depth.upload(depth)
        #
        depth_map, gpu_depth_resized = self.resize_depth(
            self.gpu_depth, frame_shape, stream)
        #
        min_val, max_val, _, _  = cv2.cuda.minMaxLoc(gpu_depth_resized)
        alpha, beta             = self._depth_alpha_beta_compiled(min_val, max_val)

        depth_colored = self.normalize_depth(
            gpu_depth_resized, alpha, beta, stream)

        return depth_map, depth_colored

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
        gpu_depth_normalized = gpu_depth_resized.convertTo(cv2.CV_8UC3, alpha, beta, stream=stream)
        
        if self.offload is True:
            depth_normalized_cpu = gpu_depth_normalized.download(stream)
            depth_colored = cv2.applyColorMap(depth_normalized_cpu, cv2.COLORMAP_BONE)
        else:
            stream.waitForCompletion()
             
            depth_normalized_cpu = gpu_depth_normalized.download()
             
            depth_gpu = cp.asarray(depth_normalized_cpu)
 
            depth_colored_gpu = cp.zeros((depth_gpu.shape[0], depth_gpu.shape[1], 3), dtype=cp.uint8)
             
            x = depth_gpu.astype(cp.float32) / 255.0
            r = cp.minimum(255, cp.maximum(0, (1.0 - x) * 255))
            g = cp.minimum(255, cp.maximum(0, (1.0 - x * 0.5) * 255))
            b = cp.minimum(255, cp.maximum(0, x * 255))
            
            depth_colored_gpu[:, :, 0] = b.astype(cp.uint8)
            depth_colored_gpu[:, :, 1] = g.astype(cp.uint8)
            depth_colored_gpu[:, :, 2] = r.astype(cp.uint8)
            
            depth_colored = cp.asnumpy(depth_colored_gpu)
        
        return depth_colored


class TrackerProcess:
    '''
    MOT Instance for RT Multi-Object Tracking
    '''
    def __init__(self, optimize, parallelism):

        if optimize:
            self._count_dim_compiled = numba.jit(nopython=True, cache=True)(_count_dim)
            self._get_depth_compiled = numba.jit(nopython=True, parallel=parallelism, cache=True)(_get_depth_at_box)
            self._count_vertical_compiled = numba.jit(nopython=True, cache=True)(_count_vertical)
        else:
            self._count_dim_compiled = _count_dim
            self._get_depth_compiled = _get_depth_at_box
            self._count_vertical_compiled = _count_vertical
        
    def count_dim(self, tlwh):
        dims = self._count_dim_compiled(tlwh)
        return [int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])]

    def get_depth_at_box(self, depth_map, tlwh):
        x, y, w, h = self.count_dim(tlwh)
        return self._get_depth_compiled(depth_map, x, y, w, h)

    def count_vertical(self, tlwh):
        return self._count_vertical_compiled(tlwh[2], tlwh[3], TrackArgs.ASPECT_RATIO_THRESH)

    def draw_transparent_highlight(self, img, x, y, w, h, color, alpha=TrackArgs.HIGHLIGHT_ALPHA, border_thickness=TrackArgs.BORDER_THICKNESS):
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, border_thickness)

    def draw_tracked_objects(self, frame, depth_colored, depth_map, online_targets):
        tracked_count = 0
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            vertical = self.count_vertical(tlwh)

            if tlwh[2] * tlwh[3] > TrackArgs.MIN_BOX_AREA and not vertical:
                tracked_count += 1
                x, y, w, h = self.count_dim(tlwh)
                color = FontConfig.THEMECOLORS[tid % len(FontConfig.THEMECOLORS)]
                self.draw_transparent_highlight(frame, x, y, w, h, color)
                self.draw_transparent_highlight(depth_colored, x, y, w, h, color)

                avg_depth = self.get_depth_at_box(depth_map, tlwh)

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
    def __init__(self, Init, LoadIns, args):
        self.args = args

        self.TrackerInstance    = TrackerProcess(optimize=args.optimize, parallelism=args.parallel)
        self.DepthInstance      = DepthProcess(optimize=args.optimize, offload=args.offload)
 
        if args.optimize:
            self._calculate_fps_compiled = numba.jit(nopython=args.optimize)(_calculate_fps)
        else:
            self._calculate_fps_compiled = _calculate_fps

        self.cap, self.source_type, self.fps, self.width, self.height, self.total_frames = Init.initialize_video_source(debugging=args.debug)

        self.depth_trt_inference = LoadIns.mde_model()
        self.bytetrack_predictor, self.bytetrack_tracker, self.bytetrack_timer = LoadIns.bytetrack_model()

        self.video_writer = None

        if args.save_video:
            self.video_writer = Init.initialize_video_writer()
            
        # lane for separate depth processing after MDE infer 
        # Infer -> Depth Process -> BBProcess, haven't benchmarked the benefit of 
        # this additional stream yet.
        self.stream_1 = cv2.cuda.Stream()

    def create_combined_view(self, frame, depth_colored):
        return cv2.hconcat([frame, depth_colored])

    def calculate_fps(self, current_time, start_time):
        return self._calculate_fps_compiled(current_time, start_time)

    def print_controls(self):
        print("\nRunning ByteTrack with Depth Estimation...")
        print(f"Source type: {self.source_type}")
        print(f"Saving video: {'Yes' if self.args.save_video else 'No'}")
        print("\nControls:")
        print("  'q' - quit")
        if self.source_type == "video":
            print("  'space' - pause/resume")

    def depth_tracking_loop(self):
        self.frame_id = 0
        self.paused = False

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    if self.source_type == "video":
                        logger.info("End of video reached")
                    else:
                        logger.warning("Frame capture failed")
                    break
                
                self.start_time = time.time()
                depth = self.depth_trt_inference.infer(frame)  
                depth_map, depth_colored = self.DepthInstance.process_depth_map(depth, frame.shape, self.stream_1)

                outputs, img_info = self.bytetrack_predictor.inference(frame, self.bytetrack_timer)
                
                self.tracked_count = 0

                if outputs[0] is not None:
                    online_targets = self.bytetrack_tracker.update(
                        outputs[0],
                        [
                        img_info['height'], 
                        img_info['width']
                        ],
                        self.bytetrack_predictor.test_size
                    )
                    self.tracked_count = self.TrackerInstance.draw_tracked_objects(frame, depth_colored, depth_map, online_targets)
                    self.bytetrack_timer.toc()
                else:
                    self.bytetrack_timer.toc()

                self.fps = self.calculate_fps(time.time(), self.start_time)

                self.draw_overlay_info(frame, self.fps, self.tracked_count, self.frame_id, self.total_frames)
                combined = self.create_combined_view(frame, depth_colored)

                if self.video_writer is not None:
                    self.video_writer.write(combined)
                
                if not self.args.no_display:
                    cv2.imshow(AppArgs.WINDOW_NAME, combined)
                
                self.frame_id += 1
            
            if not self.args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and self.source_type == "video":
                    self.paused = not self.paused
                    logger.info(f"Video {'paused' if self.paused else 'resumed'}")
            else:
                continue
        
        return self.tracked_count, self.frame_id

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

    def cleanup(self):
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
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
    
    LoadInstance = LoadModel(InitInstance)

    MainInstance = TSVX(
        Init        = InitInstance,
        LoadIns     = LoadInstance,
        args        = args
    )
     
    MainInstance.print_controls()
     
    final_count, total_processed = MainInstance.depth_tracking_loop()
    
    MainInstance.cleanup()
    print(f"\nFinal stats:")
    print(f"  Total frames processed: {total_processed}")
    print(f"  Tracked objects in last frame: {final_count}")
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import cv2
import numpy as np
import sys

Gst.init(None)

def create_pipeline(source_type, source_path=None):
    if source_type == "camera":
        pipeline_desc = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
    elif source_type == "file":
        if not source_path:
            raise ValueError("Need a source_path for file mode.")

        pipeline_desc = (
            f"filesrc location={source_path} ! "
            "decodebin ! videoconvert ! "
            "video/x-raw,format=RGB ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
    else:
        raise ValueError("Invalid source_type: choose 'camera' or 'file'.")

    pipeline = Gst.parse_launch(pipeline_desc)
    return pipeline


def on_new_sample(sink, user_data):
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR

    buf = sample.get_buffer()
    caps = sample.get_caps()
    img_width = caps.get_structure(0).get_value('width')
    img_height = caps.get_structure(0).get_value('height')

    # Extract raw bytes
    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR

    frame = np.frombuffer(map_info.data, dtype=np.uint8)
    frame = frame.reshape((img_height, img_width, 3))
    buf.unmap(map_info)

    # Display using OpenCV
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        user_data["running"] = False

    return Gst.FlowReturn.OK


def run(source_type, source_path=None):
    pipeline = create_pipeline(source_type, source_path)
    sink = pipeline.get_by_name("sink")

    loop_flag = {"running": True}
    sink.connect("new-sample", on_new_sample, loop_flag)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        while loop_flag["running"]:
            # GStreamer needs its iteration
            msg = pipeline.get_bus().timed_pop_filtered(
                10000,
                Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if msg:
                break
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python3 script.py camera")
        print("  python3 script.py file video.mp4")
        sys.exit(0)

    mode = sys.argv[1]

    if mode == "camera":
        run("camera")
    elif mode == "file":
        if len(sys.argv) < 3:
            print("Missing video file path.")
            sys.exit(1)
        run("file", sys.argv[2])
    else:
        print("Unknown mode.")

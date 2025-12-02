python -m bytetrack.utils.conv_onnx \
-f ./bytetrack/exp/yolox_x_mot17_half.py \
-c ./checkpoints/bytetrack_x_mot17.pth.tar \
--output-name ./onnx_models/bytetrack_x.onnx \
-o 18 \
--no-onnxsim
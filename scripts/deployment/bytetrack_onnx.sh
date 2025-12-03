python -m bytetrack.utils.conv_onnx \
-f ./bytetrack/exp/yolox_l_mix_det.py \
-c ./checkpoints/bytetrack_l_mot17.pth.tar \
--output-name ./onnx_models/bytetrack_l.onnx \
-o 18 \
--no-onnxsim
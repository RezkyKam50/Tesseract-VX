python ./src/bytetrack/utils/conv_onnx.py \
    -f ./src/bytetrack/exp/yolox_tiny_mix_det.py \
    -c ./src/bytetrack/pretrained/bytetrack_tiny_mot17.pth.tar \
    --output-name ./onnx_models/bytetrack_tiny.onnx \
    --no-onnxsim
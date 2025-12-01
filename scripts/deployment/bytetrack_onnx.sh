python -m src.bytetrack.utils.conv_onnx \
    -f ./src/bytetrack/exp/yolox_tiny_mix_det.py \
    -c ./checkpoints/bytetrack_tiny_mot17.pth.tar \
    --output-name ./onnx_models/bytetrack_tiny.onnx \
    --no-onnxsim

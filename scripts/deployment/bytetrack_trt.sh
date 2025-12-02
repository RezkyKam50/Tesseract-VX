python -m bytetrack.utils.conv_trt \
    --model-name bytetrack_s \
    --ckpt-path ./checkpoints/bytetrack_s_mot17.pth.tar \
    --exp-file ./bytetrack/exp/yolox_s_mix_det.py \
    --exp-name bytetrack \
    --fp16
    
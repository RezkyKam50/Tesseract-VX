python -m bytetrack.utils.conv_trt \
    --model-name bytetrack_l \
    --ckpt-path ./checkpoints/bytetrack_l_mot17.pth.tar \
    --exp-file ./bytetrack/exp/yolox_l_mix_det.py \
    --exp-name bytetrack \
    --fp16
    
 
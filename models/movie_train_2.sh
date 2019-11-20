export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/visualbert
export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py \
  -folder ../self_trained_models/mpii/mpii-pre-train-from-bert \
  -config ../configs/video/mpii-pre-train-from-bert.json

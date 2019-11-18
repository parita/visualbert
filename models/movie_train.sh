export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/visualbert
export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/

CUDA_VISIBLE_DEVICES=2,3,4 python train.py \
  -folder ../self_trained_models/mpii/mpii-pre-train \
  -config ../configs/video/mpii-pre-train.json

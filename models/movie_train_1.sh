export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/visualbert
export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python train.py \
  -folder ../self_trained_models/mpii/fine-tune-qa \
  -config ../configs/video/fine-tune-qa.json

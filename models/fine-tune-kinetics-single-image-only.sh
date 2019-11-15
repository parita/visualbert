export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/visualbert
export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/

# Fine tuning with single image pretraining with 1000 images of different videos
CUDA_VISIBLE_DEVICES=4,5 python train.py \
  -folder ../self_trained_models/vcr/fine-tune-qa-1/ \
  -config ../configs/vcr-kinetics/fine-tune-qa.json

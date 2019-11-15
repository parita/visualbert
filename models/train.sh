export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/visualbert
export PYTHONPATH=$PYTHONPATH:/proj/vondrick/parita/projects/

#CUDA_VISIBLE_DEVICES=7 python train.py \
#  -folder ../self_trained_models/vcr-kinetics/coco-pre-train-1 \
#  -config ../configs/vcr-kinetics/coco-pre-train.json

# Fine tuning with single image pretraining with 1000 images of different videos
#CUDA_VISIBLE_DEVICES=4,5 python train.py \
#  -folder ../self_trained_models/vcr/fine-tune-qa-1/ \
#  -config ../configs/vcr-kinetics/fine-tune-qa.json

#CUDA_VISIBLE_DEVICES=4,5 python train.py \
#  -folder ../self_trained_models/vcr-kinetics/kinetics-only-image-fine-tune-qa \
#  -config ../configs/vcr-kinetics/fine-tune-qa.json

#CUDA_VISIBLE_DEVICE=0 python train.py \
#  -folder ../self_trained_models/vcr-kinetics/test \
#  -config ../configs/vcr-kinetics/kinetics-pre-train.json

# Kinetics pre trained model with background correction
#CUDA_VISIBLE_DEVICES=6,7 python train.py \
#  -folder ../self_trained_models/vcr-kinetics/kinetics-pre-train-3 \
#  -config ../configs/vcr-kinetics/kinetics-pre-train.json

# Kinetics pre-training with only images and 1000 images with negative pairs from different videos
#CUDA_VISIBLE_DEVICES=4,5 python train.py \
#  -folder ../self_trained_models/vcr-kinetics/kinetics-pre-train-4 \
#  -config ../configs/vcr-kinetics/kinetics-pre-train-only-image.json


# Test accuracy for model fine tuned with kinetics pre training on only images with 1000 image (different videos)
#CUDA_VISIBLE_DEVICES=0 python train.py \
#  -folder ../self_trained_models/vcr-kinetics/fine-tune-qa-kinetics-image-only \
#  -config ../configs/vcr-kinetics/fine-tune-qa-test.json

# Kinetics fine-tuning with background correction
CUDA_VISIBLE_DEVICES=6,7 python train.py \
  -folder ../self_trained_models/vcr-kinetics/fine-tune-qa-kinetics \
  -config ../configs/vcr-kinetics/fine-tune-qa-kinetics.json

# Baseline Accuracy
#CUDA_VISIBLE_DEVICES=6,7 python train.py \
#  -folder ../self_trained_models/vcr/fine-tune-qa/ \
#  -config ../configs/vcr/fine-tune-qa.json

CUDA_VISIBLE_DEVICES=7 python extract_image_features_kinetics.py \
  --cfg /proj/vondrick/parita/projects/visualbert/model_checkpoints/detectron/35861858/e2e_faster_rcnn_R-101-FPN_2x.yaml \
  --wts /proj/vondrick/parita/projects/visualbert/model_checkpoints/detectron/35861858/model_final.pkl \
  --min_bboxes 150 \
  --max_bboxes 150 \
  --feat_name gpu_0/fc6 \
  --output_dir ../../X_VLOG/features/test/npz_files/ \
  --image-ext jpg \
  --one_giant_file ../../X_VLOG/features/test/features_test_150.th \
  /proj/vondrick/parita/projects/visualbert/X_VLOG/pairs/data/test



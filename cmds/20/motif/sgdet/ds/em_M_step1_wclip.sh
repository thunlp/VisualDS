OUTPATH=$EXP/20/sgdet/ds/m_step1_ex
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --master_port 10097 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/wsup-20.yaml" \
  DATASETS.TRAIN \(\"20DS_VG_CCKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 21 \
  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  WSUPERVISE.LOSS_TYPE  softwt WSUPERVISE.DATASET ClipDataset  WSUPERVISE.CLIP_FILE  datasets/vg/20/cc_clip_logits.pk \
  TEST.INFERENCE "LOGITS" \
  MODEL.ROI_HEADS.DETECTIONS_PER_IMG 64




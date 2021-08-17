OUTPATH=$EXP/20/sgdet/ds/m_step2
mkdir -p $OUTPATH
cp $EXP/20/sgdet/ds/m_step1/last_checkpoint $OUTPATH/last_checkpoint # finetune based on m_step1 model
cp $EXP/20/predcls/ds/e_step2/em_E.pk $OUTPATH # use e_step2 relabeled data

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
  --master_port 10095 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/wsup-20.yaml" \
  DATASETS.TRAIN \(\"20DS_VG_CCKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 80000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 21 \
  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  WSUPERVISE.LOSS_TYPE  softwt WSUPERVISE.DATASET ClipDataset  WSUPERVISE.CLIP_FILE  $OUTPATH/em_E.pk \
  TEST.INFERENCE "LOGITS" \
  SOLVER.SCHEDULE.MAX_DECAY_STEP 1 SOLVER.BASE_LR 0.0001  MODEL.ROI_HEADS.DETECTIONS_PER_IMG 64



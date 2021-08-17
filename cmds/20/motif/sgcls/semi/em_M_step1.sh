# pretrain
OUTPATH=$EXP/20/sgcls/semi/m_step1_pretrain
mkdir -p $OUTPATH
cp $EXP/20/predcls/semi/e_step1/em_E.pk $OUTPATH/em_E.pk

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --master_port 10091 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/wsup-20.yaml" \
  DATASETS.TRAIN \(\"20DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
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
  WSUPERVISE.LOSS_TYPE  softce WSUPERVISE.DATASET ClipDataset  WSUPERVISE.CLIP_FILE  $OUTPATH/em_E.pk \
  SOLVER.SCHEDULE.MAX_DECAY_STEP 2   TEST.INFERENCE "SOFTMAX"

# finetune
OUTPATH=$EXP/20/sgcls/semi/m_step1
mkdir -p $OUTPATH
cp $EXP/20/sgcls/semi/m_step1_pretrain/last_checkpoint  $OUTPATH/last_checkpoint   # finetune based  on pretrained model

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10081 --nproc_per_node=2 \
  tools/relation_train_net.py --config-file "configs/sup-20.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.PRE_VAL False \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" \
  SOLVER.MAX_ITER 80000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove MODEL.PRETRAINED_DETECTOR_CKPT ./maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.MAX_DECAY_STEP 2

OUTPATH=$EXP/20/predcls/ds/e_step2_ex
mkdir -p $OUTPATH
cp $EXP/20/predcls/ds/m_step1_ex/last_checkpoint    $OUTPATH/last_checkpoint #use m_step1 model to relable DS data

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
  --master_port 10085 --nproc_per_node=2 \
  tools/em_E_step.py --config-file "configs/wsup-20.yaml" \
  DATASETS.TRAIN \(\"20DS_VG_CCKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" SOLVER.MAX_ITER 1000000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR $EXP/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 21 \
  SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  WSUPERVISE.DATASET ClipDataset  EM.MODE E  WSUPERVISE.CLIP_FILE  datasets/vg/20/cc_clip_logits.pk


# cut top k% and merge with external information
cp tools/score.py $OUTPATH
cp tools/merge_cut_off.py $OUTPATH
cp datasets/vg/20/cc_clip_logits.pk $OUTPATH
cd $OUTPATH
python score.py
python merge_cut_off.py 0.25 0.9 # keep 25% positive samples and use 0.9*model_score+0.1*ext_score to relabel DS data
cd $SG

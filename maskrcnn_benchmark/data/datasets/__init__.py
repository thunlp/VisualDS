# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .clip_wsup_dataset import ClipDataset
from .clip_wsup_dataset_wosftmax import ClipDatasetWosftmax
from .coco import COCODataset
from .concat_dataset import ConcatDataset
from .visual_genome import VGDataset
from .voc import PascalVOCDataset
from .wsup_visual_genome import WVGDataset
__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "VGDataset", "WVGDataset", "ClipDataset", "ClipDatasetWosftmax"]

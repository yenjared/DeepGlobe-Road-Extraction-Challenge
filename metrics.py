import numpy as np

def seg_iou(mask,gt):
  return np.sum(np.logical_and(mask>128,gt>128)) / np.sum(np.logical_or(mask>128,gt>128))
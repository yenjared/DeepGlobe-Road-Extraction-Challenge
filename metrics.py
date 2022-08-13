import numpy as np

def seg_iou(mask,gt):
  return np.sum(np.logical_and(mask,gt))/np.sum(np.logical_or(mask,gt))
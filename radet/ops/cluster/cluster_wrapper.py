import torch
from . import cluster_ext
import numpy as np


def cluster_nms(bboxes, scores, categories, iou_threshold=0.65):
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)
    else:
        assert isinstance(bboxes, torch.Tensor)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    else:
        assert isinstance(scores, torch.Tensor)

    if isinstance(categories, np.ndarray):
        categories = torch.from_numpy(categories)
    else:
        assert isinstance(categories, torch.Tensor)

    instance_ids, clusters_num = cluster_ext.cluster_nms(bboxes, scores, categories, iou_threshold)
    return instance_ids, clusters_num

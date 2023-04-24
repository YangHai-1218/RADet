import torch
from . import vote_ext




def vote_nms(bboxes, cls_scores, labels, nms_cfg, score_factor=None, max_num=0):
    nms_cfg_ = nms_cfg.copy()
    nms_threshold = nms_cfg_.pop('iou_threshold', 0.6)
    cluster_score_type = nms_cfg_.pop('cluster_score', 'cls')
    vote_score_type = nms_cfg_.pop('vote_score', 'iou')
    iou_enable = nms_cfg_.pop('iou_enable', False)
    sigma = nms_cfg_.pop('sigma', 0.025)
    if isinstance(cluster_score_type, (list, tuple)):
        cluster_score = cls_scores * score_factor
    elif cluster_score_type == 'cls':
        cluster_score = cls_scores
    elif cluster_score_type == 'iou':
        cluster_score = score_factor
    else:
        raise RuntimeError(f"Unexpected cluster score type:{cluster_score_type}")

    if isinstance(vote_score_type, (list, tuple)):
        vote_score = (cls_scores * score_factor).clone()
    elif vote_score_type == 'cls':
        vote_score = cls_scores
    elif vote_score_type == 'iou':
        vote_score = score_factor
    else:
        raise RuntimeError(f"Unexpected vote score type:{vote_score_type}")

    voted_bboxes, voted_labels, voted_scores = vote_ext.vote_nms(bboxes,
                                                                 cluster_score,
                                                                 vote_score,
                                                                 labels,
                                                                 nms_threshold,
                                                                 iou_enable,
                                                                 sigma)
    voted_bboxes = torch.cat([voted_bboxes, voted_scores.view(-1, 1)], dim=-1)
    if max_num > 0:
        voted_bboxes = voted_bboxes[:max_num]
        voted_labels = voted_labels[:max_num]
    return voted_bboxes, voted_labels



def global_vote_nms(bboxes, cls_scores, labels, nms_cfg, score_factor=None, max_num=0):
    nms_cfg_ = nms_cfg.copy()
    nms_threshold = nms_cfg_.pop('iou_threshold', 0.6)
    cluster_score_type = nms_cfg_.pop('cluster_score', 'cls')
    vote_score_type = nms_cfg_.pop('vote_score', 'iou')
    iou_enable = nms_cfg_.pop('iou_enable', False)
    sigma = nms_cfg_.pop('sigma', 0.025)
    if isinstance(cluster_score_type, (list, tuple)):
        cluster_score = cls_scores * score_factor
    elif cluster_score_type == 'cls':
        cluster_score = cls_scores
    elif cluster_score_type == 'iou':
        cluster_score = score_factor
    else:
        raise RuntimeError(f"Unexpected cluster score type:{cluster_score_type}")

    if isinstance(vote_score_type, (list, tuple)):
        vote_score = (cls_scores * score_factor).clone()
    elif vote_score_type == 'cls':
        vote_score = cls_scores
    elif vote_score_type == 'iou':
        vote_score = score_factor
    else:
        raise RuntimeError(f"Unexpected vote score type:{vote_score_type}")

    voted_bboxes, voted_labels, voted_scores = vote_ext.global_vote_nms(bboxes,
                                                                 cluster_score,
                                                                 vote_score,
                                                                 labels,
                                                                 nms_threshold,
                                                                 iou_enable,
                                                                 sigma)
    voted_bboxes = torch.cat([voted_bboxes, voted_scores.view(-1, 1)], dim=-1)
    if max_num > 0:
        voted_bboxes = voted_bboxes[:max_num]
        voted_labels = voted_labels[:max_num]
    return voted_bboxes, voted_labels
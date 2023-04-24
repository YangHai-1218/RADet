import torch
import torch.nn as nn
import torch.nn.functional as F
from radet.models import HEADS
from radet.models.dense_heads import ATSSHead
from mmcv.runner import force_fp32
from radet.core import multi_apply, bbox_overlaps
from radet.ops import vote_nms, global_vote_nms
from mmcv.ops import batched_nms
INF = 1e8
EPS = 1e-12



@HEADS.register_module()
class RADetHead(ATSSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 strides=(8, 16, 32, 64, 128),
                 **kwargs):

        self.strides = strides
        super(RADetHead, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_iou = self.loss_centerness

    def forward_single(self, x, scale):
        cls_score, bbox_pred, iou_pred = super(RADetHead, self).forward_single(x, scale)
        bbox_pred = F.relu(bbox_pred)
        return cls_score, bbox_pred, iou_pred

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      points_to_gt_index=None,
                      points_weight=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, points_to_gt_index, points_weight, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list


    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_categories = []
        mlvl_anchors_filtered = []
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            score_thr = cfg.score_thr
            candidate_inds = scores > score_thr
            nms_pre = cfg.get('nms_pre', -1)
            candidate_num = candidate_inds.sum()
            if nms_pre > candidate_num:
                nms_pre = candidate_num

            if nms_pre == 0:
                continue

            scores = scores[candidate_inds]

            scores, topk_inds = scores.topk(nms_pre, sorted=False)
            candidate_nonzeros = candidate_inds.nonzero(as_tuple=False)[topk_inds, :]
            topk_inds = candidate_nonzeros[:, 0]
            categories = candidate_nonzeros[:, 1]
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            centerness = centerness[topk_inds]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_categories.append(categories)
            mlvl_anchors_filtered.append(anchors)
        if len(mlvl_bboxes) == 0:
            return torch.empty((0, 5)), torch.empty((0, 1),dtype=torch.int)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_anchors_filtered = torch.cat(mlvl_anchors_filtered)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_anchors_filtered /= mlvl_anchors_filtered.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_categories = torch.cat(mlvl_categories)

        if with_nms:
            if cfg.nms.type == 'vote':
                det_bboxes, det_labels = vote_nms(
                    mlvl_bboxes.cpu(), mlvl_scores.cpu(), mlvl_categories.cpu(),
                    nms_cfg=cfg.nms, score_factor=mlvl_centerness.cpu(),
                    max_num=cfg.max_per_img)
            elif cfg.nms.type == 'global_vote':
                det_bboxes, det_labels = global_vote_nms(
                    mlvl_bboxes.cpu(), mlvl_scores.cpu(), mlvl_categories.cpu(),
                    nms_cfg=cfg.nms, score_factor=mlvl_centerness.cpu(),
                    max_num=cfg.max_per_img)
            else:
                det_bboxes, keep = batched_nms(mlvl_bboxes, mlvl_scores*mlvl_centerness, mlvl_categories, cfg.nms)
                if cfg.max_per_img > 0:
                    det_bboxes, keep = det_bboxes[:cfg.max_per_img], keep[:cfg.max_per_img]
                det_labels = mlvl_categories[keep]
            return det_bboxes, det_labels
        else:
            mlvl_scores = mlvl_scores * mlvl_centerness
            mlvl_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], dim=-1)
            mlvl_bboxes = torch.cat([mlvl_bboxes, mlvl_anchors_filtered], dim=-1)
            return mlvl_bboxes, mlvl_categories



    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             gt_labels,
             points_to_gt_index,
             points_weight,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            iou_preds (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            points_to_gt_index (list[Tensor]): assigned target for each anchor box(point), shape(n, N), 
            points_weight (lis)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(iou_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device
        )
        
        labels, bbox_targets, weights, all_level_anchor_boxes = \
            self.get_targets(anchors_list=anchor_list, bbox_preds=bbox_preds, cls_scores=cls_scores,
                             gt_bboxes_list=gt_bboxes, gt_labels_list=gt_labels,
                             points_to_gt_index_list=points_to_gt_index, points_weight_list=points_weight, 
                             image_metas=img_metas)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and iou_preds
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_iou_preds = [
            iou_pred.permute(0, 2, 3, 1).reshape(-1)
            for iou_pred in iou_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_iou_preds = torch.cat(flatten_iou_preds)

        # prepare ground truth
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_weights = torch.cat(weights)

        flatten_anchor_boxes = torch.cat(all_level_anchor_boxes)

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero(as_tuple=False).reshape(-1)
        

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_iou_preds = flatten_iou_preds[pos_inds]
        pos_weights = flatten_weights[pos_inds]

        num_pos = pos_weights.sum()
        
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            weight=flatten_weights,
            avg_factor=num_pos + num_imgs)

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_anchor_boxes = flatten_anchor_boxes[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_anchor_boxes, pos_bbox_preds)
            pos_decoded_bbox_targets = self.bbox_coder.decode(pos_anchor_boxes, pos_bbox_targets)

            pos_iou_targets = bbox_overlaps(pos_decoded_bbox_preds, pos_decoded_bbox_targets, is_aligned=True).detach()

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_bbox_targets,
                weight=(pos_iou_targets.clamp(min=EPS) * pos_weights),
                avg_factor=(pos_iou_targets.clamp(min=EPS) * pos_weights).sum(),
            )
            loss_iou = self.loss_iou(pos_iou_preds,
                                     pos_iou_targets,
                                     weight=pos_weights,
                                     avg_factor=pos_weights.sum())
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_iou = pos_iou_preds.sum()


        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou=loss_iou,
        )

    def get_targets(self,
                    anchors_list,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    points_to_gt_index_list,
                    points_weight_list,
                    image_metas,):
        '''
        Compute regression, classification and  centers targets for points in multiple images.
        Args:
            anchors_list (list[list[Tensor]]): Anchors of each fpn level, each has shape
                (num_anchors, 2).
            cls_scores (list[Tensor]): Box scores for each scale level with shape
                (N, num_anchors* * num_classes, H, W)
            bbox_preds (list[Tensor]): Box preds for each scale level with shape
                (N, num_anchors * 4, H, W)
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            distance_map_list (list[list[Tensor]]): Pseudo distance maps of each box

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
                concat_lvl_weights (list[Tensor]): Weights of each cell at each level
        '''
        num_imgs = len(anchors_list)
        assert len(anchors_list) == len(image_metas) == len(points_to_gt_index_list) \
               == len(points_weight_list) == len(gt_labels_list) == len(gt_bboxes_list)

        num_levels = len(anchors_list[0])
        assert num_levels == len(bbox_preds) == len(cls_scores)
        num_level_anchors = [anchors.size(0) for anchors in anchors_list[0]]
        self.num_level_anchors = num_level_anchors

        anchors_list = [torch.cat(anchor) for anchor in anchors_list]
        labels_list, bbox_targets_list, weights_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points_to_gt_index_list,
            points_weight_list,
            image_metas,
            anchors_list,
        )

        # split to per img, per level
        labels_list = [labels.split(num_level_anchors, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_level_anchors, 0)
            for bbox_targets in bbox_targets_list
        ]
        weights_list = [weights.split(num_level_anchors, 0) for weights in weights_list]
        anchors_list = [anchors.split(num_level_anchors, 0) for anchors in anchors_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_weights = []
        concat_lvl_anchors = []

        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list])
            )
            concat_lvl_bbox_targets.append(
                torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            )
            concat_lvl_weights.append(
                torch.cat([weights[i] for weights in weights_list])
            )
            concat_lvl_anchors.append(
                torch.cat([anchors[i] for anchors in anchors_list])
            )
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_weights, concat_lvl_anchors



    def _get_target_single(self, 
                        gt_bboxes, 
                        gt_labels, 
                        points_to_gt_index,
                        points_weight,
                        image_meta, 
                        anchor_boxes):
        '''Compute regression, classification targets and weights for a single image'''
        num_points = anchor_boxes.size(0)
        num_gts = gt_labels.size(0)
        labels = gt_labels.new_full((num_points,), self.num_classes)
        bbox_targets = gt_bboxes.new_zeros((num_points, 4))
        if num_gts == 0:
            return labels, bbox_targets, points_weight

        # points_to_gt_index is 1-based, -1 means negative sample, 0 means ignore,
        pos_inds, non_neg_inds = points_to_gt_index > 0, points_to_gt_index > -1
        labels[non_neg_inds] = gt_labels[points_to_gt_index[non_neg_inds] - 1]
        bbox_targets[pos_inds] = self.bbox_coder.encode(anchor_boxes[pos_inds], gt_bboxes[points_to_gt_index[pos_inds] - 1])
        return labels, bbox_targets, points_weight
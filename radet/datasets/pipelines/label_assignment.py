import numpy as np
import torch
import math
from ..builder import PIPELINES
from radet.core.anchor import build_anchor_generator

from PIL import ImageDraw, ImageColor, Image
from matplotlib import pyplot as plt
import time


INF = 1e8
EPS = 1e-8
@PIPELINES.register_module()
class LabelAssignment:
    '''
    Can not perform dynamic sampling. 
    Effective for BOP challange, where the image resolution is fixed, and an image will presents multiple objects.
    Args:
        strides (list): stride for each feature level
        regress_ranges (list[tuple]): regress_ranges for each feature level
        positive_num (int): randomly sample fixed number of cells to be positive sample
        neg_threshold (float): cells with distance above this thrshold are regared as candidates
        adapt_positve_num (bool): if change the positve num for each object according to the object size
        balance_sample (bool): when the number of candidate sample is less than positive number, 
            repeatly sample until the number of positive samples reaches positive number

    
    '''
    def __init__(self, 
                strides=(8, 16, 32, 64, 128), 
                regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF)),
                anchor_generator_cfg: dict()=None,
                positive_num=10,
                neg_threshold=0.2,
                adapt_positive_num=False,
                balance_sample=False,
                multiply_samplepro_for_weight=False,
                ambiguous_sample='min_area',
                random_sample_by_distance=True) -> None:
        assert len(strides) == len(regress_ranges)
        self.num_levels = len(strides)
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.positive_num = positive_num
        self.ambiguous_sample = ambiguous_sample
        self.neg_threshold = neg_threshold
        self.adapt_positive_num = adapt_positive_num
        self.balance_sample = balance_sample
        self.random_sample_by_distance = random_sample_by_distance
        self.multiply_sample_pro_for_weight = multiply_samplepro_for_weight
        self.anchor_generator = build_anchor_generator(anchor_generator_cfg)
    



    def generate_candidate_cell(self, gt_bboxes, gt_labels, anchor_boxes, regress_ranges):
        num_gts =  gt_labels.shape[0]
        regress_ranges = np.repeat(regress_ranges[:, None, :], num_gts, axis=1)
        xs = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2
        ys = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2

        xs = np.repeat(xs[:, None], num_gts, axis=1)
        ys = np.repeat(ys[:, None], num_gts, axis=1)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = np.stack((left, top, right, bottom), axis=-1)
        min_side = np.min(bbox_targets, axis=-1)
        is_in_bbox = min_side > 0.01
        max_side = np.max(bbox_targets, axis=-1)
        is_cared_in_level = ((max_side >= regress_ranges[..., 0]) &
                             (max_side <= regress_ranges[..., 1]))
        is_candidate = is_cared_in_level & is_in_bbox
        return is_candidate
    
    def cal_sample_pro(self, distance_map:np.ndarray, anchor_boxes:np.ndarray):
        num_points, num_gt = anchor_boxes.shape[0], distance_map.shape[0]
        xs, ys = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2, (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2
        xs, ys = xs.astype(np.int64), ys.astype(np.int64)
        distance_cell = np.zeros((num_points, num_gt), dtype=np.float32)
        distance_map_float = distance_map.astype(np.float32)
        for i in range(num_gt):
            distance_cell[:, i] = distance_map_float[i, ys, xs]
        return distance_cell

    def adapt_cal_k(self, candidate_anchor_sizes, object_size):
        anchor_size_lvl, anchor_num_lvl = np.unique(candidate_anchor_sizes, return_counts=True)
        anchor_num_lvl_ratio = anchor_num_lvl / candidate_anchor_sizes.shape[0]
        dk = np.exp((object_size - anchor_size_lvl) / (2 * anchor_size_lvl))
        dk = (anchor_num_lvl_ratio * dk).sum()
        dk = self.positive_num * dk
        dk = int(dk + 0.5)
        return dk

    def random_sample(self, candidate_points_pro, candidate_points_index, candidate_anchor_sizes, object_size):
        non_neg_candidate = candidate_points_pro > (self.neg_threshold * np.max(candidate_points_pro))
        non_neg_index = candidate_points_index[non_neg_candidate]
        neg_index = candidate_points_index[~ non_neg_candidate]
        non_neg_num = non_neg_candidate.sum()
        non_neg_points_pro = candidate_points_pro[non_neg_candidate]
        sample_pro = non_neg_points_pro / np.sum(non_neg_points_pro)
        if self.adapt_positive_num:
            positive_num = self.adapt_cal_k(candidate_anchor_sizes, object_size)
        else:
            positive_num = self.positive_num
        
        if non_neg_num < positive_num:
            if self.balance_sample:
                if self.random_sample_by_distance:
                    chosen_index = np.random.choice(a=non_neg_num, size=positive_num, p=sample_pro, replace=True)
                else:
                    chosen_index = np.random.choice(a=non_neg_num, size=positive_num, replace=True)
            else:
                chosen_index = np.arange(0, non_neg_num)
        else:
            if self.random_sample_by_distance:
                chosen_index = np.random.choice(a=non_neg_num, size=positive_num, p=sample_pro, replace=False)
            else:
                chosen_index = np.random.choice(a=non_neg_num, size=positive_num, replace=False)
        
        sampled_flag = np.zeros_like(non_neg_index, dtype=np.bool_)
        sampled_flag[chosen_index] = True
        pos_index, pos_idx_count = np.unique(chosen_index, return_counts=True)
        weight = pos_idx_count.astype(np.float32)
        if self.multiply_sample_pro_for_weight:
            weight *= non_neg_points_pro[pos_index]
        pos_index = non_neg_index[pos_index]
        untouched_index = non_neg_index[~sampled_flag]
        return pos_index, untouched_index, neg_index, weight



    
    def __call__(self, results):
        image_h, image_w, _ = results['img_shape']
        featmap_sizes = [(math.ceil(image_h/stride), math.ceil(image_w/stride)) for stride in self.strides]
        multi_level_anchor_list = self.anchor_generator.grid_anchors(featmap_sizes, 'cpu')

        num_level_anchors = [anchors.size(0) for anchors in multi_level_anchor_list]
        expanded_regress_ranges = [
            multi_level_anchor_list[0][i].new_tensor(self.regress_ranges[i])[None].expand((num_level_anchors[i], 2))
            for i in range(self.num_levels)
        ]

        concat_anchor_boxes = torch.cat(multi_level_anchor_list, dim=0).numpy()
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0).numpy()
        concat_anchor_box_sizes = concat_anchor_boxes[:, 2] - concat_anchor_boxes[:, 0]

        gt_bboxes, gt_labels, distance_maps = results['gt_bboxes'], results['gt_labels'], results['distance_maps']
        distance_maps = distance_maps.to_ndarray()

        candidate_flag = self.generate_candidate_cell(gt_bboxes, gt_labels, concat_anchor_boxes, concat_regress_ranges)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        distance_map_cells = self.cal_sample_pro(distance_maps, concat_anchor_boxes)
        if self.ambiguous_sample == 'max_dis':
            is_max_distance = torch.zeros_like(is_candidate)
            is_max_distance.scatter_(1, distance_map_cells.argmax(dim=1, keepdims=True), 1.0)
            is_candidate = is_candidate & is_max_distance

        num_points = len(concat_anchor_boxes)
        num_gt = len(gt_bboxes)
        
        points_to_gt_index = np.full((num_points, ), -1, dtype=np.int64)
        weights = np.ones((num_points, ), dtype=np.float32)
        
        # sort the ground truth from small to large
        sorted_area_index = sorted(range(num_gt), key=lambda k: areas[k])

        for gt_index in sorted_area_index:
            gt_bbox = gt_bboxes[gt_index]
            is_candidate_per_gt = candidate_flag[:, gt_index]
            candidate_points_index,  = np.nonzero(is_candidate_per_gt)

            if self.ambiguous_sample == 'min_area':
                not_assigned_index, = np.nonzero(points_to_gt_index[candidate_points_index] == -1)
                candidate_points_index = candidate_points_index[not_assigned_index]

            candidate_points_num = candidate_points_index.shape[0]
            if candidate_points_num == 0:
                continue

            w, h = (gt_bbox[2] - gt_bbox[0]), (gt_bbox[3] - gt_bbox[1])
            candidate_points_pro = distance_map_cells[candidate_points_index, gt_index]
            candidate_points_pro = candidate_points_pro.clip(min=EPS)

            pos_index, untouched_index, neg_index, pos_weights = self.random_sample(
                candidate_points_pro, candidate_points_index, concat_anchor_box_sizes[candidate_points_index], max(w, h)
            )
            # 1-base, -1 means negative, 0 means ignore
            points_to_gt_index[pos_index] = gt_index + 1
            points_to_gt_index[untouched_index] = 0
            weights[pos_index] = pos_weights
            weights[untouched_index] = 0.
        
        results['points_to_gt_index'] = points_to_gt_index
        results['points_weight'] = weights
        # debug(results['img'], gt_bboxes, gt_labels, points_to_gt_index, concat_anchor_boxes)
        return results  


@PIPELINES.register_module()
class LabelAssignmentParallel:
    '''
    Can not perform dynamic sampling. 
    Effective for BOP challange, where the image resolution is fixed, and an image will presents multiple objects.
    Args:
        strides (list): stride for each feature level
        regress_ranges (list[tuple]): regress_ranges for each feature level
        positive_num (int): randomly sample fixed number of cells to be positive sample
        neg_threshold (float): cells with distance above this thrshold are regared as candidates
        adapt_positve_num (bool): if change the positve num for each object according to the object size
        balance_sample (bool): when the number of candidate sample is less than positive number, 
            repeatly sample until the number of positive samples reaches positive number

    
    '''
    def __init__(self, 
                device='cuda',
                strides=(8, 16, 32, 64, 128), 
                regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF)),
                anchor_generator_cfg: dict()=None,
                positive_num=10,
                neg_threshold=0.2,
                adapt_positive_num=False,
                balance_sample=False,
                multiply_samplepro_for_weight=False,
                ambiguous_sample='min_area',
                random_sample_by_distance=True) -> None:
        assert len(strides) == len(regress_ranges)
        self.device = device
        self.num_levels = len(strides)
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.positive_num = positive_num
        self.ambiguous_sample = ambiguous_sample
        self.neg_threshold = neg_threshold
        self.adapt_positive_num = adapt_positive_num
        self.balance_sample = balance_sample
        self.random_sample_by_distance = random_sample_by_distance
        self.multiply_sample_pro_for_weight = multiply_samplepro_for_weight
        self.anchor_generator = build_anchor_generator(anchor_generator_cfg)
    



    def generate_candidate_cell(self, 
                                gt_bboxes:torch.Tensor, 
                                gt_labels:torch.Tensor, 
                                anchor_boxes:torch.Tensor, 
                                regress_ranges:torch.Tensor):
        num_gts =  gt_labels.shape[0]
        num_points = anchor_boxes.size(0)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        xs = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2
        ys = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2

        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), dim=-1)
        min_side = torch.min(bbox_targets, dim=-1)[0]
        is_in_bbox = min_side > 0.01
        max_side = torch.max(bbox_targets, dim=-1)[0]
        is_cared_in_level = ((max_side >= regress_ranges[..., 0]) &
                             (max_side <= regress_ranges[..., 1]))
        is_candidate = is_cared_in_level & is_in_bbox
        return is_candidate
    
    def cal_sample_pro(self, distance_map:torch.Tensor, anchor_boxes:torch.Tensor):
        num_points, num_gt = anchor_boxes.shape[0], distance_map.shape[0]
        xs, ys = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2, (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2
        xs, ys = xs.long(), ys.long()
        distance_cell = torch.zeros((num_points, num_gt), dtype=torch.float32, device=anchor_boxes.device)
        for i in range(num_gt):
            distance_cell[:, i] = distance_map[i, ys, xs]
        return distance_cell

    def adapt_cal_k(self, candidate_anchor_sizes:torch.Tensor, object_size):
        anchor_size_lvl, anchor_num_lvl = torch.unique(candidate_anchor_sizes, return_counts=True)
        anchor_num_lvl_ratio = anchor_num_lvl / candidate_anchor_sizes.shape[0]
        dk = torch.exp((object_size - anchor_size_lvl) / (2 * anchor_size_lvl))
        dk = (anchor_num_lvl_ratio * dk).sum()
        dk = self.positive_num * dk
        dk = (dk + 0.5).int()
        return dk

    
    def __call__(self, results):
        start = time.time()
        image_h, image_w, _ = results['img_shape']
        featmap_sizes = [(math.ceil(image_h/stride), math.ceil(image_w/stride)) for stride in self.strides]
        multi_level_anchor_list = self.anchor_generator.grid_anchors(featmap_sizes, self.device)

        num_level_anchors = [anchors.size(0) for anchors in multi_level_anchor_list]
        expanded_regress_ranges = [
            multi_level_anchor_list[0][i].new_tensor(self.regress_ranges[i])[None].expand((num_level_anchors[i], 2))
            for i in range(self.num_levels)
        ]

        concat_anchor_boxes = torch.cat(multi_level_anchor_list, dim=0)
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_anchor_box_sizes = concat_anchor_boxes[:, 2] - concat_anchor_boxes[:, 0]

        gt_bboxes, gt_labels, distance_maps = results['gt_bboxes'], results['gt_labels'], results['distance_maps']
        distance_maps = distance_maps.to_tensor(torch.float32, self.device)
        gt_bboxes = torch.from_numpy(gt_bboxes).to(self.device)
        gt_labels = torch.from_numpy(gt_labels).to(self.device)

        num_points = len(concat_anchor_boxes)
        num_gt = len(gt_bboxes)

        candidate_flag = self.generate_candidate_cell(gt_bboxes, gt_labels, concat_anchor_boxes, concat_regress_ranges)
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        object_size = torch.maximum(gt_bboxes[:, 2] - gt_bboxes[:, 0], gt_bboxes[:, 3] - gt_bboxes[:, 1])
        distance_map_cells = self.cal_sample_pro(distance_maps, concat_anchor_boxes)

        # handle ambiguous sample
        if self.ambiguous_sample == 'max_dis':
            is_max_distance = torch.zeros_like(candidate_flag)
            is_max_distance.scatter_(1, distance_map_cells.argmax(dim=1, keepdim=True), 1.0)
            candidate_flag = candidate_flag & is_max_distance
        elif self.ambiguous_sample == 'min_area':
            is_min_area = torch.zeros_like(candidate_flag)
            areas = areas[None].expand((num_points, num_gt))
            areas[~candidate_flag] = INF 
            is_min_area.scatter_(1, areas.argmin(dim=1, keepdim=True), 1.0)
            candidate_flag = candidate_flag & is_min_area

        
        points_to_gt_index = gt_bboxes.new_full((num_points, ), -1, dtype=torch.long)
        weights = gt_bboxes.new_ones((num_points, ), dtype=torch.float32)
    
        neg_thresh = torch.max(distance_map_cells, dim=0, keepdim=True)[0] * self.neg_threshold
        non_neg_flag = (distance_map_cells > neg_thresh) & candidate_flag
        distance_map_cells[~non_neg_flag] = 0.
        sample_pro_cells = distance_map_cells / torch.clamp(torch.sum(distance_map_cells, dim=0, keepdim=True), min=EPS)
        non_neg_num = torch.sum(non_neg_flag, axis=0, keepdim=False)
        non_neg_index, gt_index = torch.nonzero(non_neg_flag, as_tuple=True)
        non_neg_anchor_box_sizes = concat_anchor_box_sizes[non_neg_index]

        for i in range(num_gt):
            non_neg_index_per_gt = non_neg_index[gt_index == i]
            non_neg_anchor_box_sizes_per_gt = non_neg_anchor_box_sizes[gt_index == i]
            non_neg_num_per_gt = non_neg_num[i]
            if non_neg_num_per_gt == 0:
                continue
            if self.adapt_positive_num:
                positive_num = self.adapt_cal_k(non_neg_anchor_box_sizes_per_gt, object_size[i])
            else:
                positive_num = self.positive_num 
            

            if self.balance_sample:
                repeat_num = int(torch.ceil(positive_num / non_neg_num_per_gt))
                # chosen_index = np.random.choice(a=non_neg_num_per_gt*repeat_num, size=positive_num, replace=False)
                # chosen_index = np.concatenate([np.arange(non_neg_num_per_gt)] * repeat_num)[chosen_index]
                chosen_index = torch.randperm((non_neg_num_per_gt*repeat_num).item(), device=self.device)[:positive_num]
                chosen_index = torch.cat([torch.arange(non_neg_num_per_gt, device=self.device)] * repeat_num)[chosen_index]
            else:
                if non_neg_num_per_gt < positive_num:
                    # chosen_index = np.arange(non_neg_index_per_gt)
                    chosen_index = torch.arange(non_neg_num_per_gt, device=self.device)
                else:
                    # chosen_index = np.random.choice(a=non_neg_num_per_gt, size=positive_num, replace=False)
                    chosen_index = torch.randperm(non_neg_num_per_gt.item(), device=self.device)[:positive_num]
            
            sampled_flag = torch.zeros_like(non_neg_index_per_gt).to(torch.bool)
            sampled_flag[chosen_index] = True
            pos_index, pos_idx_count = torch.unique(chosen_index, return_counts=True)
            pos_index = non_neg_index_per_gt[pos_index]
            untouched_index = non_neg_index_per_gt[~sampled_flag]

            weight = pos_idx_count.to(torch.float32)
            if self.multiply_sample_pro_for_weight:
                weight *= sample_pro_cells[pos_index, i]
            points_to_gt_index[pos_index] = i+1
            points_to_gt_index[untouched_index] = 0
            weights[untouched_index] = 0.
            weights[pos_index] = weight
        
        points_to_gt_index = points_to_gt_index.to('cpu').numpy()
        weights = weights.to('cpu').numpy()
        results['points_to_gt_index'] = points_to_gt_index
        results['points_weight'] = weights
        end = time.time()
        print(f"label assignment consumes time:{end - start}, object num:{num_gt}, mean time:{(end-start)/num_gt}")
        # debug(results['img'], gt_bboxes, gt_labels, points_to_gt_index, concat_anchor_boxes)
        return results 



    
    # def __call__(self, results):
    #     start = time.time()
    #     image_h, image_w, _ = results['img_shape']
    #     featmap_sizes = [(math.ceil(image_h/stride), math.ceil(image_w/stride)) for stride in self.strides]
    #     multi_level_anchor_list = self.anchor_generator.grid_anchors(featmap_sizes, 'cpu')

    #     num_level_anchors = [anchors.size(0) for anchors in multi_level_anchor_list]
    #     expanded_regress_ranges = [
    #         multi_level_anchor_list[0][i].new_tensor(self.regress_ranges[i])[None].expand((num_level_anchors[i], 2))
    #         for i in range(self.num_levels)
    #     ]

    #     concat_anchor_boxes = torch.cat(multi_level_anchor_list, dim=0).numpy()
    #     concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0).numpy()
    #     concat_anchor_box_sizes = concat_anchor_boxes[:, 2] - concat_anchor_boxes[:, 0]

    #     gt_bboxes, gt_labels, distance_maps = results['gt_bboxes'], results['gt_labels'], results['distance_maps']
    #     distance_maps = distance_maps.to_ndarray()

    #     num_points = len(concat_anchor_boxes)
    #     num_gt = len(gt_bboxes)

    #     candidate_flag = self.generate_candidate_cell(gt_bboxes, gt_labels, concat_anchor_boxes, concat_regress_ranges)
    #     areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
    #     object_size = np.maximum(gt_bboxes[:, 2] - gt_bboxes[:, 0], gt_bboxes[:, 3] - gt_bboxes[:, 1])
    #     distance_map_cells = self.cal_sample_pro(distance_maps, concat_anchor_boxes)

    #     # handle ambiguous sample
    #     if self.ambiguous_sample == 'max_dis':
    #         is_max_distance = np.zeros_like(candidate_flag)
    #         np.put_along_axis(is_max_distance , np.argmax(distance_map_cells, axis=1, keepdims=True), 1.0)
    #         candidate_flag = candidate_flag & is_max_distance
    #     elif self.ambiguous_sample == 'min_area':
    #         is_min_area = np.zeros_like(candidate_flag)
    #         areas = np.repeat(areas[None], num_points, axis=0)
    #         areas[~candidate_flag] = INF 
    #         np.put_along_axis(is_min_area, np.argmin(areas, axis=1, keepdims=True), 1.0, axis=1)
    #         candidate_flag = candidate_flag & is_min_area

        
    #     points_to_gt_index = np.full((num_points, ), -1, dtype=np.int64)
    #     weights = np.ones((num_points, ), dtype=np.float32)
    
    #     neg_thresh = np.max(distance_map_cells, axis=0, keepdims=True) * self.neg_threshold
    #     non_neg_flag = (distance_map_cells > neg_thresh) & candidate_flag
    #     distance_map_cells[~non_neg_flag] = 0.
    #     sample_pro_cells = distance_map_cells / np.clip(np.sum(distance_map_cells, axis=0, keepdims=True), a_min=EPS, a_max=INF)
    #     non_neg_num = np.sum(non_neg_flag, axis=0, keepdims=False)
    #     non_neg_index, gt_index = np.nonzero(non_neg_flag)
    #     non_neg_anchor_box_sizes = concat_anchor_box_sizes[non_neg_index]
    #     for i in range(num_gt):
    #         non_neg_index_per_gt = non_neg_index[gt_index == i]
    #         non_neg_anchor_box_sizes_per_gt = non_neg_anchor_box_sizes[gt_index == i]
    #         if self.adapt_positive_num:
    #             positive_num = self.adapt_cal_k(non_neg_anchor_box_sizes_per_gt, object_size[i])
    #         else:
    #             positive_num = self.positive_num 
            
    #         non_neg_num_per_gt = non_neg_num[i]
    #         if self.balance_sample:
    #             repeat_num = int(np.ceil(positive_num / non_neg_num_per_gt))
    #             chosen_index = np.random.choice(a=non_neg_num_per_gt*repeat_num, size=positive_num, replace=False)
    #             chosen_index = np.concatenate([np.arange(non_neg_num_per_gt)] * repeat_num)[chosen_index]  
    #         else:
    #             if non_neg_num_per_gt < positive_num:
    #                 chosen_index = np.arange(non_neg_index_per_gt)
    #             else:
    #                 chosen_index = np.random.choice(a=non_neg_num_per_gt, size=positive_num, replace=False)
            
    #         sampled_flag = np.zeros_like(non_neg_index_per_gt, dtype=np.bool_)
    #         sampled_flag[chosen_index] = True
    #         pos_index, pos_idx_count = np.unique(chosen_index, return_counts=True)
    #         pos_index = non_neg_index_per_gt[pos_index]
    #         untouched_index = non_neg_index_per_gt[~sampled_flag]

    #         weight = pos_idx_count.astype(np.float32)
    #         if self.multiply_sample_pro_for_weight:
    #             weight *= sample_pro_cells[pos_index, i]
    #         points_to_gt_index[pos_index] = i+1
    #         points_to_gt_index[untouched_index] = 0
    #         weights[untouched_index] = 0.
    #         weights[pos_index] = weight

    #     # # sort the ground truth from small to large
    #     # sorted_area_index = sorted(range(num_gt), key=lambda k: areas[k])

    #     # for gt_index in sorted_area_index:
    #     #     gt_bbox = gt_bboxes[gt_index]
    #     #     is_candidate_per_gt = candidate_flag[:, gt_index]
    #     #     candidate_points_index,  = np.nonzero(is_candidate_per_gt)

    #     #     if self.ambiguous_sample == 'min_area':
    #     #         not_assigned_index, = np.nonzero(points_to_gt_index[candidate_points_index] == -1)
    #     #         candidate_points_index = candidate_points_index[not_assigned_index]

    #     #     candidate_points_num = candidate_points_index.shape[0]
    #     #     if candidate_points_num == 0:
    #     #         continue

    #     #     w, h = (gt_bbox[2] - gt_bbox[0]), (gt_bbox[3] - gt_bbox[1])
    #     #     candidate_points_pro = distance_map_cells[candidate_points_index, gt_index]
    #     #     candidate_points_pro = candidate_points_pro.clip(min=EPS)

    #     #     pos_index, untouched_index, neg_index, pos_weights = self.random_sample(
    #     #         candidate_points_pro, candidate_points_index, concat_anchor_box_sizes[candidate_points_index], max(w, h)
    #     #     )
    #     #     # 1-base, -1 means negative, 0 means ignore
    #     #     points_to_gt_index[pos_index] = gt_index + 1
    #     #     points_to_gt_index[untouched_index] = 0
    #     #     weights[pos_index] = pos_weights
    #     #     weights[untouched_index] = 0.
        
    #     results['points_to_gt_index'] = points_to_gt_index
    #     results['points_weight'] = weights
    #     end = time.time()
    #     print(f"label assignment consumes time:{end - start}")
    #     # debug(results['img'], gt_bboxes, gt_labels, points_to_gt_index, concat_anchor_boxes)
    #     return results 



def trans_paste(w, h, color, bg_img, alpha=1.0, box=(0, 0)):
    alpha = int(255*alpha)
    color = ImageColor.getrgb(color)
    color = tuple(list(color) + [alpha])
    fg_img = Image.new("RGBA", (h, w), color)
    bg_img.paste(fg_img, box, fg_img)
    return bg_img

def draw_bbox_text(drawobj, xmin, ymin, xmax, ymax, text, color, bd=2):
    drawobj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawobj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawobj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawobj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    strlen = len(text)
    if strlen > 0:
        drawobj.text((xmin+3, ymin), text, fill='Red')

def debug(image, gt_bboxes, gt_labels, points_to_gt_index, all_level_anchor_boxes, sample_pro=None, name_tab=[str(i) for i in range(100)]):
    color_list = list(ImageColor.colormap.keys())
    image = Image.fromarray(image[..., ::-1].astype(np.uint8))
    drawobj = ImageDraw.ImageDraw(image)
    for bbox, label in zip(gt_bboxes, gt_labels):
        draw_bbox_text(drawobj, bbox[0], bbox[1], bbox[2], bbox[3], color=color_list[label], text=name_tab[label])
    
    pos_anchor_box_index,  = np.nonzero(points_to_gt_index > 0)
    untouched_anchor_box_index, = np.nonzero(points_to_gt_index==0)

    pos_anchor_box, untouched_anchor_box = all_level_anchor_boxes[pos_anchor_box_index], all_level_anchor_boxes[untouched_anchor_box_index]
    # pos_sample_pro, untouched_sample_pro = sample_pro[pos_anchor_box_index], sample_pro[untouched_anchor_box_index]
    label_target = gt_labels[points_to_gt_index[pos_anchor_box_index] - 1]
    w, h = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) / 8, (pos_anchor_box[:, 3] - pos_anchor_box[:, 1]) / 8
    w, h = w.astype(np.int32), h.astype(np.int32)
    cx, cy = (pos_anchor_box[:, 2] + pos_anchor_box[:, 0]) / 2, (pos_anchor_box[:, 3] + pos_anchor_box[:, 1]) / 2
    x1, y1 = (cx - w / 2).astype(np.int32), (cy - h / 2).astype(np.int32)
    pos_anchor_num = pos_anchor_box.shape[0]
    for j in range(pos_anchor_num):
        # trans_paste(w[j], h[j], color=color_list[label_target[j]], bg_img=image, alpha=pos_sample_pro[j]*0.8, box=(x1[j], y1[j]))
        trans_paste(w[j]-2, h[j]-2, color=color_list[label_target[j]], bg_img=image, alpha=0.8, box=(x1[j]-1, y1[j]-1))
        # draw_bbox_text(drawobj, x1[j], y1[j], x1[j]+w[j], y1[j]+h[j], text='', color=color_list[label_target[j]], bd=1)

    w, h = (untouched_anchor_box[:, 2] - untouched_anchor_box[:, 0]) / 8, (untouched_anchor_box[:, 3] - untouched_anchor_box[:, 1]) / 8
    w, h = w.astype(np.int32), h.astype(np.int32)
    cx, cy = (untouched_anchor_box[:, 2] + untouched_anchor_box[:, 0]) / 2, (
                untouched_anchor_box[:, 3] + untouched_anchor_box[:, 1]) / 2
    x1, y1 = (cx - w / 2).astype(np.int32), (cy - h / 2).astype(np.int32)
    untouched_anchor_num = untouched_anchor_box.shape[0]
    for j in range(untouched_anchor_num):
        # trans_paste(w[j], h[j], color='white', bg_img=image, alpha=untouched_sample_pro[j]*0.5, box=(x1[j], y1[j]))
        trans_paste(w[j]-2, h[j]-2, color='white', bg_img=image, alpha=0.5, box=(x1[j]-1, y1[j]-1))
    # image.show()
    # print(f"image showed")
    plt.imshow(image)
    plt.savefig('debug/debug.png')
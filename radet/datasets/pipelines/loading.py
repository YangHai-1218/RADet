import os.path as osp

import mmcv
import numpy as np
import random
import pycocotools.mask as maskUtils

from radet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES
from radet.ops import GDT_box2distance, MBD_box2distance
import torch

from matplotlib import pyplot as plt
from matplotlib import cm
import cv2, math, os


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles(object):
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadMaskFromFile(object):
    '''
    Load instance ground truth mask from png file
    '''
    def __init__(self,
                 replace_path=('rgb', 'mask_visib'),
                 file_client_args=dict(backend='disk')):
        self.replace_path = replace_path
        self.file_client_args = file_client_args.copy()
        self.file_client = None


    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        h, w = results['img_info']['height'], results['img_info']['width']
        filename = filename.replace(self.replace_path[0], self.replace_path[1])
        num_gt = len(results['gt_bboxes'])
        filename_base = filename.rpartition('.')[0]
        masks = []
        for i in range(num_gt):
            filename = filename_base + '_' + str(i).zfill(6)+'.png'
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(img_bytes, flag='grayscale')
            img = (img / 255).astype(np.uint8)
            masks.append(img)
        masks = np.stack(masks, axis=0)
        masks = BitmapMasks(masks, h, w)
        results['gt_masks'] = masks
        results['mask_fields'].append('gt_masks')
        return results



@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 with_bop_mask=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_bop_mask = with_bop_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results
    
    def _load_bop_masks(self, results):
        '''Private function to load bop format mask
        Args:
            results (dict): Results dict from :obj:`dataset`
        Returns:
            
        '''
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_mask_paths = results['ann_info']['masks']
        gt_masks = []
        for mask_path in gt_mask_paths:
            filename = osp.join(results['seg_prefix'], mask_path)
            img_bytes = self.file_client.get(filename)
            gt_mask = mmcv.imfrombytes(img_bytes, flag='unchanged')
            dtype = gt_mask.dtype
            gt_mask = (gt_mask/gt_mask.max()).astype(dtype)
            gt_masks.append(gt_mask)
        
        gt_masks = BitmapMasks(gt_masks, h, w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results
        

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_bop_mask:
            results = self._load_bop_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_bop_mask={self.with_bop_mask}, '
        repr_str += f'with_seg={self.with_seg})'
        repr_str += f'poly2mask={self.poly2mask})'
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadProposals(object):
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations(object):
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
    """

    def __init__(self, min_gt_bbox_wh):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            return None
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results

@PIPELINES.register_module()
class GenerateDistanceMap(object):
    def __init__(self, 
                with_gt_mask=True, 
                small_object_size=32**2, 
                pad_ratio=0.05, 
                distance_transform='gdt',
                 **kwargs):
        self.with_gt_mask = with_gt_mask
        if self.with_gt_mask:
            self.forward = self.forward_with_gt_mask
            self.small_object_size = small_object_size
        else:
            self.forward = self.forward_wo_gt_mask
            self.small_object_size = small_object_size
            self.pad_ratio = pad_ratio

            if distance_transform == 'gdt':
                self.distance_transform = GDT_box2distance(**kwargs)
            elif distance_transform == 'mbd':
                self.distance_transform = MBD_box2distance(**kwargs)
            else:
                raise RuntimeError(f"Unexpected distance transform type, expect 'mbd' or 'gdt', got{distance_transform}")

    def show_distance_map(self, distance_map, image):

        if isinstance(distance_map, torch.Tensor):
            distance_map = distance_map.numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, bins = np.histogram(distance_map.reshape(-1), bins=100)
        colormap = cm.get_cmap("Reds")
        ax_image = plt.subplot(121)
        ax_dmap = plt.subplot(122)
        ax_image.imshow(image)
        ax_dmap.imshow(distance_map, cmap=colormap)

        plt.show()


    def forward_with_gt_mask(self, results):
        gt_masks = results['gt_masks']
        return gt_masks




    def forward_wo_gt_mask(self, results):
        img = results['img']
        img_h, img_w, _ = results['img_shape']
        assert isinstance(img, np.ndarray), f"image should be numpy.ndarray, got {type(img)}"
        assert img.dtype == np.uint8, f"image dtype should be np.uint8, got{img.dtype}"
        assert img.ndim == 3, f"image should have three channel and BGR format, got{img.ndim} channels"

        # bbox np.ndarray， xyxy format
        gt_bboxes = results['gt_bboxes']
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        maskenable = areas > self.small_object_size

        gt_bboxes_bak = gt_bboxes.copy().astype(np.int_)
        box_images = []
        bbox_images_region = np.zeros_like(gt_bboxes_bak)
        for i, xyxy in enumerate(gt_bboxes_bak):
            pad_x_l = math.ceil((xyxy[2] - xyxy[0])*self.pad_ratio)
            pad_x_r = pad_x_l
            pad_y_t = math.ceil((xyxy[3] - xyxy[1])*self.pad_ratio)
            pad_y_b = pad_y_t

            box_image = np.zeros((xyxy[3] - xyxy[1] + pad_y_t + pad_y_b, xyxy[2] - xyxy[0] + pad_x_l + pad_x_r, 3), dtype=np.uint8)
            random_color = [random.randint(0, 255) for _ in range(3)]
            box_image[:, :, :] = random_color

            box_image_h, box_image_w = box_image.shape[:2]
            xyxy_o = xyxy.copy()

            xyxy += np.array([-pad_x_l, -pad_y_t, pad_x_r, pad_y_b], dtype=xyxy.dtype)
            refined_x1 = np.clip(xyxy[0], a_min=0, a_max=img_w-1)
            refined_y1 = np.clip(xyxy[1], a_min=0, a_max=img_h-1)
            refined_x2 = np.clip(xyxy[2], a_min=0, a_max=img_w-1)
            refined_y2 = np.clip(xyxy[3], a_min=0, a_max=img_h-1)

            box_img_x1 = 0 + refined_x1 - xyxy[0]
            box_img_y1 = 0 + refined_y1 - xyxy[1]
            box_img_x2 = box_image_w - (xyxy[-2] - refined_x2)
            box_img_y2 = box_image_h - (xyxy[-1] - refined_y2)

            box_image[box_img_y1:box_img_y2, box_img_x1:box_img_x2] = img[refined_y1:refined_y2, refined_x1:refined_x2]

            bbox_images_region[i] = np.array([xyxy_o[0] - xyxy[0], xyxy_o[1] - xyxy[1],
                                              box_image_w - (xyxy[2] - xyxy_o[2]), box_image_h - (xyxy[-1] - xyxy_o[-1])])

            # origin_box_img = box_image[2:-2, 2:-2, :]
            # padded_box_img = np.pad(origin_box_img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=[128])
            # box_image[1:-1, 1:-1, :]  = padded_box_img
            # origin_box_img = np.pad(origin_box_img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=[128])
            # box_image[bbox_images_region[i, 1] - 1:bbox_images_region[i, 3] + 1,
            #            bbox_images_region[i, 0] - 1:bbox_images_region[i, 2] + 1, :] = origin_box_img
            box_images.append(box_image)

        distance_maps = self.distance_transform(box_images, maskenable, bbox_images_region)
        distance_maps_mapped = []
        for distance_map, bbox in zip(distance_maps, gt_bboxes):
            distance_map_mapped = torch.zeros((img_h, img_w), dtype=torch.float32)
            bbox_rounded = bbox.astype(np.int_)
            distance_map_mapped[bbox_rounded[1]:bbox_rounded[3], bbox_rounded[0]:bbox_rounded[2]] = distance_map
            distance_maps_mapped.append(distance_map_mapped)
        return distance_maps_mapped

    def __call__(self, results):
        distance_maps = self.forward(results)
        results['distance_maps'] = distance_maps
        return results
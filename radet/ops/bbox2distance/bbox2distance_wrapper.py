import torch
from . import bbox2distance_ext
import time
import numpy as np
import cv2


INF = 1e8
class MBD_box2distance(object):
    def __init__(self, mode='center', multi_scale=False, alpha=0.1, niter=4, base_size=300, interval=3):
        self.alpha = alpha
        self.niter = niter
        self.base_size = base_size
        self.interval = interval
        self.multi_scale = multi_scale
        self.size = 150
        assert mode in ['center', 'mean']
        self.mode = mode


    def cal_dmap_single_scale(self, image):
        image_h, image_w, _ = image.shape
        horinzontal_seeds_x = list(range(0, image_w, self.interval))
        if horinzontal_seeds_x[-1] != image_w - 1:
            horinzontal_seeds_x.append(image_w - 1)

        horinzontal_seeds_x = torch.tensor(horinzontal_seeds_x, dtype=torch.int64)
        top_seeds_y = torch.zeros_like(horinzontal_seeds_x)
        bottom_seeds_y = torch.full_like(horinzontal_seeds_x, fill_value=image_h - 1)

        vertical_seeds_y = torch.arange(1, image_h - 1, self.interval, dtype=torch.int64)
        left_seeds_x = torch.zeros_like(vertical_seeds_y)
        right_seeds_x = torch.full_like(vertical_seeds_y, fill_value=image_w - 1)

        seeds_x = torch.cat((horinzontal_seeds_x, horinzontal_seeds_x, left_seeds_x, right_seeds_x))
        seeds_y = torch.cat((top_seeds_y, bottom_seeds_y, vertical_seeds_y, vertical_seeds_y))
        box_image = torch.from_numpy(image)
        dmap = bbox2distance_ext.MBD(box_image, seeds_x, seeds_y, self.alpha, self.niter, self.base_size)
        return dmap

    def cal_dmap_single_im(self, box_image):
        image_h, image_w = box_image.shape[:2]
        target_scales = [1]
        if isinstance(self.multi_scale, dict):
            for size, scale in self.multi_scale.items():
                if image_h * image_w > size**2:
                    target_scales.append(scale)

        dmaps = []
        for scale in target_scales:
            image = cv2.resize(box_image, (int(image_w * scale), int(image_h * scale)))
            dmap = self.cal_dmap_single_scale(image).numpy()
            dmap = cv2.resize(dmap, (image_w, image_h))
            dmap = torch.from_numpy(dmap)
            dmap = dmap / dmap.max()
            dmaps.append(dmap)
        dmap = dmaps.pop(0)
        for dmap_ in dmaps:
            dmap += dmap_

        return dmap

    def __call__(self, box_images, mask_enable, bbox_images_xy):
        '''
       compute pixel distance to background by Geodesic distance transform
       :param box_images: list(p.ndarray), BGR format, uint8 dtype
       :param mask_enable: list(bool), if True, compute distance, else set the distance of all pixels to be equal
       :param bbox_images_xy: actual image region by removing padding area
       :return:
        '''
        dmaps = []
        for box_image, enable, bbox_image_xy in zip(box_images, mask_enable, bbox_images_xy):
            if not enable:
                box_image = torch.from_numpy(box_image)
                dmap = box_image.new_ones(box_image.shape[:2])
                dmap = dmap[bbox_image_xy[1]: bbox_image_xy[-1], bbox_image_xy[0]: bbox_image_xy[2]]
                dmaps.append(dmap)

            else:
                if self.mode == 'center':
                    image_h, image_w = box_image.shape[:2]
                    short_edge = min(image_w, image_h)
                    ratio = self.size / short_edge
                    new_w, new_h = int(image_w * ratio), int(image_h * ratio)
                    resized_image = cv2.resize(box_image, (new_w, new_h))
                    blured_image = cv2.GaussianBlur(resized_image, ksize=(9,9), sigmaX=0, borderType=cv2.BORDER_DEFAULT)
                    dmap = self.cal_dmap_single_scale(blured_image)
                    dmap = cv2.resize(dmap.numpy(), (image_w, image_h))
                    dmap = torch.from_numpy(dmap)
                else:
                    dmap = self.cal_dmap_single_scale(box_image)
                dmap = dmap[bbox_image_xy[1]: bbox_image_xy[-1], bbox_image_xy[0]: bbox_image_xy[2]]
                dmaps.append(dmap)
        return dmaps



class GDT_box2distance(object):
    def __init__(self, edge_mode='sed', interval=3, mode='center'):
        self.interval = interval
        if edge_mode == 'sed':
            self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
            self.extract_edge_func = self.sde_extract_edge
        else:
            self.extract_edge_func = self.sobel_extract_edge
        self.mode = mode
        self.size = 150

    def sde_extract_edge(self, image):
        '''
        :param image: np.ndarray, BGR format, uint8
        :return:
        '''
        edge = self.edge_detection.detectEdges(np.float32(image) / 255.0)
        return edge


    def sobel_extract_edge(self, image):
        '''
        :param image: np.ndarray, BGR format, uint8
        :return:
        '''
        src = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, borderType=cv2.BORDER_DEFAULT)
        gray_src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        edge_x = cv2.Sobel(gray_src, cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        edge_y = cv2.Sobel(gray_src, cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        edge = cv2.addWeighted(edge_x, 0.5, edge_y, 0.5, 0)
        edge = np.abs(edge)
        edge = edge / np.max(edge)
        return edge

    def cal_dmap_single_scale(self, box_image):
        image_h, image_w = box_image.shape[:2]
        horinzontal_seeds_x = list(range(0, image_w, self.interval))
        if horinzontal_seeds_x[-1] != image_w - 1:
            horinzontal_seeds_x.append(image_w - 1)

        horinzontal_seeds_x = torch.tensor(horinzontal_seeds_x, dtype=torch.int64)
        top_seeds_y = torch.zeros_like(horinzontal_seeds_x)
        bottom_seeds_y = torch.full_like(horinzontal_seeds_x, fill_value=image_h - 1)

        vertical_seeds_y = torch.arange(1, image_h - 1, self.interval, dtype=torch.int64)
        left_seeds_x = torch.zeros_like(vertical_seeds_y)
        right_seeds_x = torch.full_like(vertical_seeds_y, fill_value=image_w - 1)

        seeds_x = torch.cat((horinzontal_seeds_x, horinzontal_seeds_x, left_seeds_x, right_seeds_x))
        seeds_y = torch.cat((top_seeds_y, bottom_seeds_y, vertical_seeds_y, vertical_seeds_y))

        edge = self.extract_edge_func(box_image)
        dmap = bbox2distance_ext.GDT(torch.from_numpy(edge), seeds_x, seeds_y)
        return dmap


    def __call__(self, box_images, mask_enable, bbox_images_xy):
        '''
        compute pixel distance to background by Geodesic distance transform
        :param box_images: list(p.ndarray), BGR format, uint8 dtype
        :param mask_enable: list(bool), if True, compute distance, else set the distance of all pixels to be equal
        :param bbox_images_xy: actual image region by removing padding area
        :return:
        '''
        dmaps = []
        for box_image, enable, bbox_image_xy in zip(box_images, mask_enable, bbox_images_xy):
            image_h, image_w = box_image.shape[:2]
            if not enable:
                dmap = torch.ones((image_h, image_w), dtype=torch.float32)
                dmap = dmap[bbox_image_xy[1]: bbox_image_xy[-1], bbox_image_xy[0]: bbox_image_xy[2]]
                dmaps.append(dmap)

            else:
                if self.mode == 'center':
                    short_edge = min(image_w, image_h)
                    ratio = self.size / short_edge
                    new_w, new_h = int(image_w * ratio), int(image_h * ratio)
                    resized_image = cv2.resize(box_image, (new_w, new_h))
                    blured_image = cv2.GaussianBlur(resized_image, ksize=(9, 9), sigmaX=0,
                                                    borderType=cv2.BORDER_DEFAULT)
                    dmap = self.cal_dmap_single_scale(blured_image)
                    dmap = cv2.resize(dmap.numpy(), (image_w, image_h))
                    dmap = torch.from_numpy(dmap)
                else:
                    dmap = self.cal_dmap_single_scale(box_image)

                dmap = dmap[bbox_image_xy[1]: bbox_image_xy[-1], bbox_image_xy[0]: bbox_image_xy[2]]
                dmaps.append(dmap)
        return dmaps
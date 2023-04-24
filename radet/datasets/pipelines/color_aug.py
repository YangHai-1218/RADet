import random
import glob
from os import path as osp
import cv2, mmcv
from mmcv.utils import build_from_cfg
import numpy as np
from PIL import ImageEnhance, Image, ImageFilter

from ..builder import PIPELINES



@PIPELINES.register_module()
class RandomHSV:
    def __init__(self, h_ratio, s_ratio, v_ratio, prob=1.0) -> None:
        self.h_ratio = h_ratio
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio
        self.prob = prob


    def aug_hsv(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
        h = img_hsv[:, :, 0].astype(np.float32)  # hue
        s = img_hsv[:, :, 1].astype(np.float32)  # saturation
        v = img_hsv[:, :, 2].astype(np.float32)  # value
        a = random.uniform(-1, 1) * self.h_ratio + 1
        b = random.uniform(-1, 1) * self.s_ratio + 1
        c = random.uniform(-1, 1) * self.v_ratio + 1
        h *= a
        s *= b
        v *= c
        img_hsv[:, :, 0] = h if a < 1 else h.clip(None, 179)
        img_hsv[:, :, 1] = s if b < 1 else s.clip(None, 255)
        img_hsv[:, :, 2] = v if c < 1 else v.clip(None, 255)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    def __call__(self, results):
        if random.random() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = self.aug_hsv(img).astype(np.uint8)
        return results



@PIPELINES.register_module()
class RandomNoise:
    def __init__(self, noise_ratio, prob=1.0) -> None:
        self.noise_ratio = noise_ratio
        self.prob = prob 

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            noise_sigma = random.uniform(0, self.noise_ratio)
            gauss = np.random.normal(0, noise_sigma, img.shape) * 255
            img = img + gauss
            img[img > 255] = 255
            img[img < 0] = 0
            img = img.astype(np.uint8)
            results[key] = img
        return results


@PIPELINES.register_module()
class RandomSmooth:
    def __init__(self, max_kernel_size=7, prob=1.0) -> None:
        self.max_kernel_size = max_kernel_size
        self.kernel_sizes = [i*2+1 for i in range(self.max_kernel_size//2+1)]
        self.prob = prob 
        
    def __call__(self, results):
        if random.random() > self.prob:
            return results 
        for key in results.get('img_fields', ['img']):
            img = results[key]
            kernel_size = random.choice(self.kernel_sizes)
            img = cv2.blur(img, (kernel_size, kernel_size))
            results[key] = img 
        return results
            
        



@PIPELINES.register_module()
class RandomBackground:
    def __init__(self, 
                background_dir, 
                prob=0.8, 
                file_client_args=dict(backend='disk'), 
                flag='color') -> None:
        self.background_dir = background_dir
        self.background_images = glob.glob(osp.join(background_dir, '*.jpg')) + \
                                glob.glob(osp.join(background_dir, '*.png'))
        if len(self.background_images) == 0:
            raise RuntimeError(f'No background images found in {background_dir}')
        self.prob = prob
        self.augment_with_mask = True
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.flag = flag

    def merge_background_by_mask(self, foreImg, backImg, mask):
        forergb = foreImg[:, :, :3]
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.ones((foreImg.shape[0], foreImg.shape[1], 3), np.float32)
        background_mask = mask.get_background_mask()
        alpha[background_mask] = 0
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        bg_img_path = random.choice(self.background_images)
        img_bytes = self.file_client.get(filepath=bg_img_path)
        bg_img = mmcv.imfrombytes(img_bytes, flag=self.flag, channel_order='bgr')
        image = results.get('img')
        gt_mask = results.get('gt_masks')
        image = self.merge_background_by_mask(image, bg_img, gt_mask)
        results['img'] = image
        return results
    

class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, image):
        # pil image here, don't check
        if random.random() <= self.p:
            image = self._pillow_fn(image).enhance(factor=random.uniform(*self.factor_interval))
        return image

@PIPELINES.register_module()
class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0., 50.)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness,
                         p=p,
                         factor_interval=factor_interval)
@PIPELINES.register_module()
class PillowContrast(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.2, 50.)):
        super().__init__(pillow_fn=ImageEnhance.Contrast,
                         p=p,
                         factor_interval=factor_interval)

@PIPELINES.register_module()
class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, p=0.5, factor_interval=(0.1, 6.0)):
        super().__init__(pillow_fn=ImageEnhance.Brightness,
                         p=p,
                         factor_interval=factor_interval)

@PIPELINES.register_module()
class PillowColor(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.0, 20.0)):
        super().__init__(pillow_fn=ImageEnhance.Color,
                         p=p,
                         factor_interval=factor_interval)

@PIPELINES.register_module()
class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.factor_intervel = factor_interval
    
    def __call__(self, image):
        k = random.randint(*self.factor_intervel)
        image = image.filter(ImageFilter.GaussianBlur(k))
        return image

@PIPELINES.register_module()
class CosyPoseAug:
    def __init__(self, p=0.8, pipelines=[]):
        super().__init__()
        self.p = p
        self.pipelines = [
           build_from_cfg(p, PIPELINES) for p in pipelines
        ]
    
    def pil_to_cv2(self, image):
        cv2_image = np.array(image)
        # RGB to BGR
        cv2_image = cv2_image[:, :, ::-1].copy()
        return cv2_image

    def cv2_to_pil(self, image):
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        return pil_image

    def __call__(self, results):
        if random.random() > self.p:
            return results 
        image = results['img']
        pil_image = self.cv2_to_pil(image)
        for p in self.pipelines:
            pil_image = p(pil_image)
        results['img'] = self.pil_to_cv2(pil_image)
        return results
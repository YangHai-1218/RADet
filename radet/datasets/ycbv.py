from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class YcbvDataset(CocoDataset):
    CLASSES = ('master_chef_can', 'cracker_box', 'sugar_box', 'tomato_soup_can',
               'mustard_bottle', 'tuna_fish_can', 'pudding_box', 'gelatin_box',
               'potted_meat_can', 'banana', 'pitcher_base', 'bleach_cleanser',
               'bowl', 'mug', 'power_drill', 'wood_block', 'scissors', 'large_marker',
               'large_clamp', 'extra_large_clamp', 'foam_brick')
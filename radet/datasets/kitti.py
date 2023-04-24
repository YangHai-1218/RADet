from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class KittiDataset(CocoDataset):
    CLASSES = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
               'Cyclist', 'Tram', 'Misc')

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        return super(KittiDataset, self).evaluate(results=results,
                                                  metric=metric,
                                                  logger=logger,
                                                  jsonfile_prefix=jsonfile_prefix,
                                                  classwise=True,
                                                  proposal_nums=proposal_nums,
                                                  iou_thrs=iou_thrs,
                                                  metric_items=metric_items)


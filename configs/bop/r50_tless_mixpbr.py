_base_ = [
    '../base/datasets/bop_detection_mix.py',
    '../base/default_runtime.py']


OBJ_NUM = 30
CLASS_NAMES = tuple([i+1 for i in range(OBJ_NUM)])


model = dict(
    type='RADetHead',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='RADetHead',
        num_classes=30,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='TBLRBBoxCoder',
            normalizer=1/8),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
)

train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='vote',
             iou_threshold=0.65,
             cluster_score=['cls', 'iou'],
             vote_score=['iou', 'cls'],
             iou_enable=False,
             sima=0.025,),
    max_per_img=100)


data_root = 'data/tless/'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        dataset_0=dict(
            ann_file=data_root + 'detector_annotations/train_pbr.json',
            img_prefix=data_root + 'train_pbr/',
            seg_prefix=data_root + 'train_pbr/',
            min_visib_frac = 0.1,
            ratio=3,
            classes=CLASS_NAMES,
        ),
        dataset_1=dict(
            ann_file=data_root+'detector_annotations/train_real.json',
            img_prefix=data_root + 'train_primesense/',
            seg_prefix=data_root + 'train_primesense/',
            ratio=1,
            classes=CLASS_NAMES,
        )
    ),
    val=dict(
        ann_file=data_root +'detector_annotations/test_bop19.json',
        img_prefix=data_root + 'test_primesense/',
        classes=CLASS_NAMES,
    ),
    test=dict(
        ann_file=data_root + 'detector_annotations/test_bop19.json',
        img_prefix=data_root + 'test_primesense/',
        classes=CLASS_NAMES,
    )
)


load_from = 'work_dirs/tless_r50_radet_pbr/latest.pth'
work_dir = 'work_dirs/tless_r50_radet_mixpbr'
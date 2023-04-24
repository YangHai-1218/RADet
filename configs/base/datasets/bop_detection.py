dataset_type = 'BOPDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_bop_mask=True),
    dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
    dict(type='RandomBackground', background_dir='data/coco', prob=0.3),
    dict(type='CosyPoseAug', p=0.8,
        pipelines=[
            dict(type='PillowBlur', p=1., factor_interval=(1, 3)),
            dict(type='PillowSharpness', p=0.3, factor_interval=(0., 50.)),
            dict(type='PillowContrast', p=0.3, factor_interval=(0.2, 50.)),
            dict(type='PillowBrightness', p=0.5, factor_interval=(0.1, 6.0)),
            dict(type='PillowColor', p=0.3, factor_interval=(0., 20.)),
    ]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='GenerateDistanceMap'),
    dict(type='LabelAssignment',
        anchor_generator_cfg=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]
        ),
        neg_threshold=0.2,
        positive_num=10,
        adapt_positive_num=False,
        balance_sample=True,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=16),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'points_to_gt_index', 'points_weight'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'detector_annotations/train_pbr.json',
        img_prefix=data_root + 'train_pbr/',
        seg_prefix=data_root + 'train_pbr',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root +'detector_annotations/test_bop19.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'detector_annotations/test_bop19.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        bop_submission=True,
    ),
)

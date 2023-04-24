optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    eps=1e-08,
    amsgrad=False,
)
lr_config = dict(
    policy='OneCycle',
    max_lr=0.0004,
    total_steps=100100,
    pct_start=0.05,
    anneal_strategy='linear')

runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(interval=10000)
evaluation = dict(interval=10000, metric='bbox')
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

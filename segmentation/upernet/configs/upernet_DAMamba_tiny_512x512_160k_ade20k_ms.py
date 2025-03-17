_base_ = [
    '_base_/models/upernet_DAMamba.py',
    '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_160k.py'
]
# optimizer
model = dict(
    backbone=dict(
        pretrained='path/DAMamba-T.pth',
        type='DAMamba_tiny',
    ),
    decode_head=dict(
        in_channels=[80, 160, 320, 512],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=150
    ))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'query_embedding': dict(decay_mult=0.),
                                                 'relative_pos_bias_local': dict(decay_mult=0.),
                                                 'cpb': dict(decay_mult=0.),
                                                 'temperature': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)

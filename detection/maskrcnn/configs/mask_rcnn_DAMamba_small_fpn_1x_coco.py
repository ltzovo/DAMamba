_base_ = [
    '_base_/models/mask_rcnn_DAMamba_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    backbone=dict(
        pretrained='path/DAMamba-S.pth',
        type='DAMamba_small',
        pretrain_size=224,
        img_size=800),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 512]))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'query_embedding': dict(decay_mult=0.),
                                                 'relative_pos_bias_local': dict(decay_mult=0.),
                                                 'cpb': dict(decay_mult=0.),
                                                 'temperature': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
# fp16 = dict(loss_scale=512.)


_base_ = [
    '/SSD/IEEE/src/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/SSD/IEEE/src/mmdetection/configs/_base_/schedules/schedule_1x.py', '/SSD/IEEE/src/mmdetection/configs/_base_/default_runtime.py'
]
dataset_type = 'CocoDataset'
classes_list = ('D00','D10','D20','D40')
num_classes = 4

# model settings
model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# data setting
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    
    # Add advanced augmentations
    dict(type='PhotoMetricDistortion'),
    # dict(type='MixUp'),   
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=40,
    train=dict(
        type='ConcatDataset',
        pipeline=train_pipeline,
        datasets=[
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file='/SSD/IEEE/src/prepare_data/train_labels/India/train/annotations/coco/train_balance.json',
                img_prefix='/SSD/IEEE/data/India/train/images',
                pipeline=train_pipeline),
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file= '/SSD/IEEE/src/prepare_data/train_labels/United_States/train/annotations/coco/train_balance.json',
                img_prefix='/SSD/IEEE/data/United_States/train/images',
                pipeline=train_pipeline),
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file='/SSD/IEEE/src/prepare_data/train_labels/United_States/train/annotations/coco/test_balance.json',
                img_prefix='/SSD/IEEE/data/United_States/train/images',
                pipeline=train_pipeline),
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file='/SSD/IEEE/src/prepare_data/train_labels/China_MotorBike/train/annotations/coco/train_balance.json',
                img_prefix='/SSD/IEEE/data/China_MotorBike/train/images',
                pipeline=train_pipeline),
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file= '/SSD/IEEE/src/prepare_data/train_labels/China_MotorBike/train/annotations/coco/test_balance.json',
                img_prefix='/SSD/IEEE/data/China_MotorBike/train/images',
                pipeline=train_pipeline),
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file= '/SSD/IEEE/src/prepare_data/train_labels/Czech/train/annotations/coco/train_balance.json',
                img_prefix='/SSD/IEEE/data/Czech/train/images',
                pipeline=train_pipeline),
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file= '/SSD/IEEE/src/prepare_data/train_labels/Czech/train/annotations/coco/test_balance.json',
                img_prefix='/SSD/IEEE/data/Czech/train/images',
                pipeline=train_pipeline),
            # NORWAY
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file= '/SSD/IEEE/src/prepare_data/train_labels/Norway/train/annotations/coco/train_balance.json',
                img_prefix='/SSD/IEEE/data/Norway/train/images',
                pipeline=train_pipeline),
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file= '/SSD/IEEE/src/prepare_data/train_labels/Norway/train/annotations/coco/test_balance.json',
                img_prefix='/SSD/IEEE/data/Norway/train/images',
                pipeline=train_pipeline),
            # JAPAN
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file= '/SSD/IEEE/src/prepare_data/train_labels/Japan/train/annotations/coco/train_balance.json',
                img_prefix='/SSD/IEEE/data/Japan/train/images',
                pipeline=train_pipeline),
            dict(
                type='CocoDataset',
                classes=classes_list,
                ann_file= '/SSD/IEEE/src/prepare_data/train_labels/Japan/train/annotations/coco/test_balance.json',
                img_prefix='/SSD/IEEE/data/Japan/train/images',
                pipeline=train_pipeline),
        
        
        ]),
    val=dict(
        type=dataset_type,
        classes = classes_list,
        ann_file='/SSD/IEEE/src/prepare_data/train_labels/India/train/annotations/coco/test_balance.json',
        img_prefix='/SSD/IEEE/data/India/train/images'),
    test=dict(
        type=dataset_type,
        classes = classes_list,
        ann_file='/SSD/IEEE/src/prepare_data/train_labels/India/train/annotations/coco/test_balance.json',
        img_prefix='/SSD/IEEE/data/India/train/images'))

# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[8, 11])

# Evaluation times
evaluation = dict(interval=1, metric='bbox', classwise = True)
checkpoint_config = dict(interval=5)
runner = dict(type='EpochBasedRunner', max_epochs=20)
work_dir = './work_dirs/india_vfnet_resnet101_train_all_aug'
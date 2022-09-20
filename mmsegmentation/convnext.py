FOLD = 0
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth',
            prefix='backbone.')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    test_cfg=dict(mode='whole'))
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'
dataset_type = 'CustomDataset'
data_root = '/content/drive/MyDrive/kaggle/hubmap-organ-segmentation/data/'
classes = [
    'background', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung'
]
palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
           [255, 0, 255]]
img_norm_cfg = dict(
    mean=[196.869, 190.186, 194.802], std=[63.01, 66.765, 65.745], to_rgb=True)
size = 512
offset = 128
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomCutOut',
        prob=0.2,
        cutout_shape=[(5, 5), (10, 10), (20, 20)],
        n_holes=3),
    dict(type='RandomRotate', prob=0.5, degree=45),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[196.869, 190.186, 194.802],
        std=[63.01, 66.765, 65.745],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[196.869, 190.186, 194.802],
                std=[63.01, 66.765, 65.745],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
#test_cfg = dict(mode='slide', crop_size=(768, 768), stride=(128, 128))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_root=data_root,
        img_dir='train',
        ann_dir='masks',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='splits/fold_0.txt',
        classes=[
            'background', 'kidney', 'prostate', 'largeintestine', 'spleen',
            'lung'
        ],
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [255, 255, 0], [255, 0, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(
                type='RandomCutOut',
                prob=0.2,
                cutout_shape=[(5, 5), (10, 10), (20, 20)],
                n_holes=3),
            dict(type='RandomRotate', prob=0.5, degree=45),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[196.869, 190.186, 194.802],
                std=[63.01, 66.765, 65.745],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CustomDataset',
        data_root=data_root,
        img_dir='train',
        ann_dir='masks',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='splits/valid_0.txt',
        classes=[
            'background', 'kidney', 'prostate', 'largeintestine', 'spleen',
            'lung'
        ],
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [255, 255, 0], [255, 0, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[196.869, 190.186, 194.802],
                        std=[63.01, 66.765, 65.745],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        data_root=data_root,
        test_mode=True,
        img_dir='train',
        ann_dir='masks',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=[
            'background', 'kidney', 'prostate', 'largeintestine', 'spleen',
            'lung'
        ],
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [255, 255, 0], [255, 0, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[196.869, 190.186, 194.802],
                        std=[63.01, 66.765, 65.745],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=1000, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
total_iters = 12000
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
find_unused_parameters = True
runner = dict(type='IterBasedRunner', max_iters=12000)
checkpoint_config = dict(by_epoch=False, interval=-1, save_optimizer=False)
evaluation = dict(
    by_epoch=False,
    interval=200,
    metric='mDice',
    pre_eval=True,
    save_best='mDice')
fp16 = dict()
work_dir = f'/content/drive/MyDrive/kaggle/hubmap-organ-segmentation/convnext_checkpoint_fold{FOLD}'
gpu_ids = [0]
auto_resume = False
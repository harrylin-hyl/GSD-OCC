# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


# we follow the online training settings  from solofusion
num_gpus = 8
samples_per_gpu = 4
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu))
num_epochs = 24
checkpoint_epoch_interval = 1
use_custom_eval_hook=True

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order 
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

# Long-Term Fusion Parameters
do_history = True
history_cat_num = 15
history_cat_conv_out_channels = 256


# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

use_checkpoint = True
sync_bn = True


# Model
grid_config = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [2.0, 42.0, 0.5],
}      
depth_categories = 80 #(grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2]

## config for bevformer
grid_config_bevformer={
        'x': [-40, 40, 0.8],
        'y': [-40, 40, 0.8],
        'z': [-1, 5.4, 6.4],
       }
bev_h_ = 100
bev_w_ = 100
numC_Trans=80
_dim_ = 512
_pos_dim_ = 40
_ffn_dim_ = numC_Trans * 4
_num_levels_= 1

num_cls = 18  # 0 others, 1-16 obj, 17 free
img_norm_cfg = None

occ_size = [200, 200, 16]
voxel_out_channel = 256


model = dict(
    type='GSDOCC',
    use_depth_supervision=True,
    do_history = do_history,
    history_cat_num=history_cat_num,
    single_bev_num_channels=numC_Trans,
    history_cat_conv_out_channels=history_cat_conv_out_channels,
    # occupancy_save_path="./GSD-Occ_wo_visible_mask_save",
    readd=True,
    img_backbone=dict(
        pretrained='ckpts/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False, 
        with_cp=use_checkpoint,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        with_cp=use_checkpoint,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet', # camera-aware depth net
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,
        with_cp=use_checkpoint,
        loss_depth_weight=1.0,
        use_dcn=False,
    ),
    forward_projection=dict(
        type='LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=16),
    frpn=None,
    history_fusion=None,
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=256,
        num_layer=[2, 2],
        stride=[2, 2],
        num_channels=[256 * 2, 256 * 4]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        input_feature_index=(0, 1),
        scale_factor=2,
        in_channels=256 * 4 + 256 * 2,
        out_channels=256),
    occupancy_head=dict(
        type='DecoupleOccHead',
        use_focal_loss=True,
        use_dice_loss= False,
        balance_cls_weight=False,
        use_deblock=True,
        use_rep_3dconv=True,
        use_rep_3dconv_kernel=[11,11,1],
        Dz=16,
        final_occ_size=occ_size,
        in_channel=voxel_out_channel,
        out_channel=num_cls,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=100.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    pts_bbox_head=None)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = 'data/nuscenes/gts'


train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='LoadOccupancy', ignore_nonvisible=True, occupancy_path=occupancy_path),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'aux_cam_params', 'gt_bboxes_3d', 'gt_labels_3d',  'gt_occupancy', 'gt_depth', 'visible_mask'
                               ])
]

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs', data_config=data_config),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
            dict(type='LoadOccupancy',  occupancy_path=occupancy_path),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs',  'gt_occupancy', 'gt_depth', 'visible_mask'])
            ]
        )
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
    occupancy_path=occupancy_path,
    use_sequence_group_flag=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    sequences_split_num=test_sequences_split_num,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=6,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        img_info_prototype='bevdet',
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
        filter_empty_gt=filter_empty_gt,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 1e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch*num_epochs,])
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(
    interval=checkpoint_epoch_interval * num_iters_per_epoch)
evaluation = dict(
    interval=24 * num_iters_per_epoch, pipeline=test_pipeline)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=2*num_iters_per_epoch,
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_iter=num_iters_per_epoch *2,
    ),
    dict(
        type='UpdateMixDepth',
        mix_alpha=5,
        max_iter = num_iters_per_epoch * num_epochs,
        mode = "sigmoid",
        # linear_w = 1
    ),
]
load_from = 'ckpts/bevdet-r50-4d-depth-cbgs.pth'
work_dir = 'work_dirs/gsdocc_r50_256x704_16f_8x4_24e.py'
# fp16 = dict(loss_scale='dynamic')


# {'RayIoU': 0.3893244501683875, 'RayIoU@1': 0.3299612740279922, 'RayIoU@2': 0.39716628270465926, 'RayIoU@4': 0.4408457937725109}
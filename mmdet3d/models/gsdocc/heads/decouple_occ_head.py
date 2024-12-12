# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet3d.models.gsdocc.modules.occ_loss_utils import lovasz_softmax, CustomFocalLoss
from mmdet3d.models.gsdocc.modules.occ_loss_utils import nusc_class_frequencies, nusc_class_names
from mmdet3d.models.gsdocc.modules.occ_loss_utils import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from torch.utils.checkpoint import checkpoint as cp
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast
from mmdet3d.models import builder
from mmdet3d.models.gsdocc.modules.rep_3dconv_dense_fusion import UniRepLKNetBlock


@HEADS.register_module()
class DecoupleOccHead(BaseModule):
    def __init__(
        self,
        in_channel,
        out_channel,
        final_occ_size=[256, 256, 20],
        balance_cls_weight=True,
        use_focal_loss=False,
        use_dice_loss= False,
        use_deblock=True,
        Dz=16,
        loss_weight_cfg=None,
        use_rep_3dconv=True,
        use_rep_3dconv_kernel=[7,7,3],
    ):
        super(DecoupleOccHead, self).__init__()
        self.use_rep_3dconv = use_rep_3dconv
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        conv_cfg = conv_cfg=dict(type='Conv2d', bias=False)
        self.use_deblock = use_deblock
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss', use_sigmoid=True, loss_weight=1.0))
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.Dz= Dz
        self.use_dice_loss = use_dice_loss
        
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        
        if self.use_rep_3dconv:
            self.unirep_3dconv_layer = UniRepLKNetBlock(80, kernel_size=(use_rep_3dconv_kernel[0],use_rep_3dconv_kernel[1],use_rep_3dconv_kernel[2]), use_sync_bn=False, attempt_use_lk_impl=True)
        # conv3d_cfg=dict(type='Conv3d', bias=False)
        # self.large_kenerl_3dconv_layer = build_conv_layer(conv3d_cfg, in_channels=80, 
        #                 out_channels=80, kernel_size=(use_rep_3dconv_kernel[0],use_rep_3dconv_kernel[1],use_rep_3dconv_kernel[2]), stride=(1,1,1), padding=[5,5,0], groups=80, bias=True)
        # self.large_kenerl_3dconv_layer = torch.nn.Conv3d(in_channels=80, out_channels=80, kernel_size=(use_rep_3dconv_kernel[0],use_rep_3dconv_kernel[1],use_rep_3dconv_kernel[2]), padding=[5,5,0], groups=80, bias=True)

        if self.use_deblock:
            upsample_cfg=dict(type='deconv', bias=False)
            upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=self.in_channel,
                    out_channels=self.in_channel // 2,
                    kernel_size=3,
                    stride=2,
                    padding=0)

            deconv_cfg=dict(type='Conv2d', bias=False)
            deconv_layer = build_conv_layer(
                    deconv_cfg,
                    in_channels=self.in_channel//2,
                    out_channels=self.in_channel//2,
                    kernel_size=2,
                    stride=1,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, self.in_channel // 2)[1],
                                    nn.ReLU(inplace=True),
                                    deconv_layer)
        else:
            deconv_cfg=dict(type='Conv2d', bias=False)
            deconv_layer = build_conv_layer(
                    deconv_cfg,
                    in_channels=self.in_channel,
                    out_channels=self.in_channel//2,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            self.deblock = nn.Sequential(deconv_layer)

        # recover 3D voxel
        # geo-voxel prediction
        self.recover_geo_occ_conv = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=self.in_channel, 
                    out_channels=self.in_channel//2, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, self.in_channel//2)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=self.in_channel//2, 
                    out_channels=self.Dz//2, kernel_size=3, stride=1, padding=1))

        # semantic-level output
        self.recover_semantic_occ_conv = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=self.in_channel, 
                    out_channels=self.in_channel//2, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, self.in_channel//2)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=self.in_channel//2, 
                    out_channels=80, kernel_size=3, stride=1, padding=1))


        # geo-voxel prediction
        mid_channel = self.in_channel // 2
        self.geo_occ_conv = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=mid_channel, 
                    out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, mid_channel)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=mid_channel, 
                    out_channels=self.Dz, kernel_size=3, stride=1, padding=1))

        # semantic-level output
        self.semantic_occ_conv = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=mid_channel, 
                    out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, mid_channel)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=mid_channel, 
                    out_channels=mid_channel, kernel_size=3, stride=1, padding=1))


        # # BEV-Voxel Linear 
        # self.bev_voxel_linear_1 = nn.Sequential(
        #         nn.Linear(mid_channel, mid_channel * 2),
        #         nn.Softplus(),
        #         nn.Linear(mid_channel* 2, mid_channel * Dz),
        #     )
        
        # # BEV-Voxel Linear 
        # self.bev_voxel_linear_2 = nn.Sequential(
        #         nn.Linear(self.in_channel, self.in_channel * 2),
        #         nn.Softplus(),
        #         nn.Linear(self.in_channel* 2, 80 * Dz//2),
        #     )
        


        # 3D voxel prediction
        conv3d_cfg=dict(type='Conv3d', bias=False)


        # self.voxel_conv_1 = nn.Sequential(
        #     build_conv_layer(conv3d_cfg, in_channels=mid_channel, 
        #             out_channels=mid_channel, kernel_size=3, stride=1, padding=1),)

        # self.voxel_conv_2 = nn.Sequential(
        #     build_conv_layer(conv3d_cfg, in_channels=self.in_channel, 
        #             out_channels=80, kernel_size=3, stride=1, padding=1),)

        self.up0 = nn.Sequential(
                nn.ConvTranspose3d(80, self.in_channel // 2,(3,3,3),padding=(1,1,1)),
                nn.BatchNorm3d(self.in_channel // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(self.in_channel // 2, self.in_channel // 2, (2, 2, 2), stride=(2, 2, 2)),
                nn.BatchNorm3d(self.in_channel // 2),
                nn.ReLU(inplace=True),
            )

        mid_channel = mid_channel
        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv3d_cfg, in_channels=mid_channel, 
                        out_channels=mid_channel//2, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv3d_cfg, in_channels=mid_channel//2, 
                        out_channels=out_channel, kernel_size=1, stride=1, padding=0))

        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
        else:
            self.class_weights = torch.ones(out_channel)/out_channel  

        self.class_names = nusc_class_names    
    
    @force_fp32(apply_to=('voxel_feats')) 
    def forward_coarse_voxel(self, bev_feats, results):
        output_occs = []
        output = {}
        # BEV feat branch
        bev_feats = self.deblock(bev_feats)
        # bev_voxel_linear
        # B,C,H,W = bev_feats.shape
        # semantic_voxel_feats = self.voxel_conv_1(bev_feats.unsqueeze(4).repeat(1,1,1,1,16))
        # semantic_voxel_feats = self.bev_voxel_linear_1(bev_feats.permute(0,2,3,1)).reshape(B,H,W,-1,C).permute(0,4,1,2,3).contiguous()
        geo_occ_outputs = self.geo_occ_conv(bev_feats)
        geo_occ_outputs = geo_occ_outputs.sigmoid()
        semantic_bev_feats = self.semantic_occ_conv(bev_feats)
        semantic_voxel_feats = (geo_occ_outputs.unsqueeze(1) * semantic_bev_feats.unsqueeze(2)).permute(0,1,3,4,2)
        # Voxel feat branch
        # bev_voxel_linear
        # B,C,H,W,D = results['forward_voxel_feat'].shape
        # recover_semantic_voxel_feats = self.voxel_conv_2(results['forward_bev_feat'].unsqueeze(4).repeat(1,1,1,1,8))
        # recover_semantic_voxel_feats = self.bev_voxel_linear_2(results['forward_bev_feat'].permute(0,2,3,1)).reshape(B,H,W,D,C).permute(0,4,1,2,3).contiguous()
        recover_geo_occ_outputs = self.recover_geo_occ_conv(results['forward_bev_feat'])
        recover_geo_occ_outputs = recover_geo_occ_outputs.sigmoid()
        recover_semantic_bev_feats = self.recover_semantic_occ_conv(results['forward_bev_feat'])
        recover_semantic_voxel_feats = (recover_geo_occ_outputs.unsqueeze(1) * recover_semantic_bev_feats.unsqueeze(2)).permute(0,1,3,4,2)
        recover_semantic_voxel_feats = recover_semantic_voxel_feats + results['forward_voxel_feat']
        if self.use_rep_3dconv:
            forward_voxel_feat = self.unirep_3dconv_layer(recover_semantic_voxel_feats)
        else:
            forward_voxel_feat = recover_semantic_voxel_feats
        # forward_voxel_feat = self.large_kenerl_3dconv_layer(recover_semantic_voxel_feats)
        lss_voxel_feats = self.up0(forward_voxel_feat)
        semantic_voxel_feats = semantic_voxel_feats + lss_voxel_feats
        out_voxel = self.occ_pred_conv(semantic_voxel_feats)
        output['out_voxel_feats'] = semantic_voxel_feats
        # output['geo_occ_outputs'] = geo_occ_outputs.permute(0,2,3,1)
        output['geo_occ_outputs'] = None
        output['occ'] = out_voxel

        return output
     
    @force_fp32()
    def forward(self, voxel_feats, results=None, img_feats=None, pts_feats=None, transform=None, **kwargs):
        output = self.forward_coarse_voxel(voxel_feats, results)
        out_voxel_feats = output['out_voxel_feats']
        geo_occ_outputs = output['geo_occ_outputs']
        coarse_occ = output['occ']

        res = {
            'output_voxels': output['occ'],
            'output_geo_voxels': output['geo_occ_outputs'],
            'output_voxels_fine': output.get('fine_output', None),
            'output_coords_fine': output.get('fine_coord', None),
        }


        return res
    
    @force_fp32()
    def forward_train(self, voxel_feats, img_feats=None, results=None, pts_feats=None, transform=None, gt_occupancy=None, gt_occupancy_flow=None, **kwargs):
        res = self.forward(voxel_feats, results=results, img_feats=img_feats, pts_feats=pts_feats, transform=transform, deploy=False, **kwargs)
        loss = self.loss(target_voxels=gt_occupancy,
            output_voxels = res['output_voxels'],
            output_geo_voxels = res['output_geo_voxels'],
            # output_geo_voxels = results['geo_occ'],
            output_coords_fine=res['output_coords_fine'],
            output_voxels_fine=res['output_voxels_fine'])

        return loss


    @force_fp32() 
    def loss_voxel(self, output_voxels, output_geo_voxels, target_voxels, tag):

        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        assert ratio == 1
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. = 255 means not visible (camera visible mask)
        if self.use_focal_loss:
            loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * self.focal_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        else:
            loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, free_index=17, is_geo_pred=False)
        # loss_dict['loss_geo_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_geo_voxels, target_voxels, free_index=17, is_geo_pred=True)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)

        if self.use_dice_loss:
            visible_mask = target_voxels!=255
            visible_pred_voxels = output_voxels.permute(0, 2, 3, 4, 1)[visible_mask]
            visible_target_voxels = target_voxels[visible_mask]
            visible_target_voxels = F.one_hot(visible_target_voxels.to(torch.long), 19)
            loss_dict['loss_voxel_dice_{}'.format(tag)] = self.dice_loss(visible_pred_voxels, visible_target_voxels)

        return loss_dict

    @force_fp32() 
    def loss(self, output_voxels=None, output_geo_voxels=None,
                output_coords_fine=None, output_voxels_fine=None, 
                target_voxels=None, visible_mask=None, **kwargs):
        loss_dict = {}
        target_voxels = target_voxels.long()
        loss_dict.update(self.loss_voxel(output_voxels, output_geo_voxels, target_voxels,  tag='c_0'))
        return loss_dict

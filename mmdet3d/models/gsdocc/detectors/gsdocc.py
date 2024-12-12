# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import force_fp32
import os
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.builder import build_head, build_neck
import numpy as np
import copy 
import spconv.pytorch as spconv
from tqdm import tqdm 
from mmdet3d.models.gsdocc.utils import run_time
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from mmcv.runner import get_dist_info
from mmdet.core import reduce_mean
import mmcv
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
from mmdet3d.core.bbox import box_np_ops # , corner_to_surfaces_3d, points_in_convex_polygon_3d_jit
import numpy as np
from pyquaternion import Quaternion
import time

def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class GSDOCC(CenterPoint):

    def __init__(self, 
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 # BEVFormer components
                 backward_projection=None,
                 history_fusion = None,
                 # FB-BEV components
                 frpn=None,
                 # depth_net
                 depth_net=None,
                 # occupancy head
                 occupancy_head=None,
                 # other settings.
                 use_depth_supervision=False,
                 readd=False,
                 occupancy_save_path=None,
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=16,
                 history_cat_conv_out_channels=None,
                 single_bev_num_channels=80,
                 mix_coefficient=0.,

                  **kwargs):
        super(GSDOCC, self).__init__(**kwargs)
        self.mix_coefficient = mix_coefficient
        # BEVDet init
        self.forward_projection = builder.build_neck(forward_projection) if forward_projection else None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None

        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
    
        # FB-BEV init
        if not self.forward_projection: assert not frpn, 'frpn relies on LSS'
        self.frpn = builder.build_head(frpn) if frpn else None

        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

        # History Fusion init
        self.history_fusion = builder.build_head(history_fusion) if history_fusion else None

        # Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None

        
        self.readd = readd # fuse voxel features and bev features
        
        self.use_depth_supervision = use_depth_supervision
        
        self.occupancy_save_path = occupancy_save_path # for saving data\for submitting to test server

        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        self.interpolation_mode = interpolation_mode
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                         if history_cat_conv_out_channels is not None 
                                         else self.single_bev_num_channels)
        ## Embed each sample with its relative temporal offset with current timestep
        # conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
        conv = nn.Conv2d
        self.history_keyframe_time_conv = nn.Sequential(
             conv(self.single_bev_num_channels + 1,
                     self.single_bev_num_channels,
                     kernel_size=1,
                     padding=0,
                     stride=1),
             nn.SyncBatchNorm(self.single_bev_num_channels),
             nn.ReLU(inplace=True))
        self.history_keyframe_cat_conv = nn.Sequential(
            conv(self.single_bev_num_channels * (self.history_cat_num + 1),
                    history_cat_conv_out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1),
            nn.SyncBatchNorm(history_cat_conv_out_channels),
            nn.ReLU(inplace=True))

        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.count = 0

        # # frozen backbone
        # trainable_params = 0
        # for name, param in self.named_parameters():
        #     print("name:", name)
        #     if "img_backbone" in name:
        #         param.requires_grad = False
        #         trainable_params += param.numel()
        # print("Total img_backbone params_MB:", trainable_params * 4 / (1024 * 1024))
        
    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
      
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        else:
            x = x[-1]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
      
        return x

    @force_fp32()
    def bev_encoder(self, x):
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)
        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)
        if type(x) not in [list, tuple]:
             x = [x]

        return x
        
    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda): # fuse_history for bev feature
        assert len(curr_bev.shape) == 4 # bev feature
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)
        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)

        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        n, c_, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
        grid = torch.stack(
            (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,4,1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = 1
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)
        grid = rt_flow.view(n, 1, 1, 4, 4) @ grid
        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:, :2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0

        tmp_bev = self.history_bev
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)
        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)
        
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        feats_to_return = torch.cat(
        [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
            1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
        ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        # if not self.do_history:
        #     self.history_bev = None
        return feats_to_return.clone()


    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        return_map = {}
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        context = self.image_encoder(img[0])
        # elapsed = time.perf_counter() - start_time
        # print("image_encoder time:", elapsed)
        cam_params = img[1:7]
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(context, mlp_input)
            return_map['depth'] = depth
            return_map['context'] = context
        else:
            context=None
            depth=None
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        if isinstance(kwargs["gt_depth"], list): # for test
            gt_depth = kwargs["gt_depth"][0]
            self.mix_coefficient = 1
        else: # for training
            gt_depth = kwargs["gt_depth"] 
            # self.mix_coefficient = 0
        # print("gt_depth:", gt_depth.shape)
        depth_labels = self.depth_net.get_downsampled_gt_depth(gt_depth)
        B, N, C, H, W = depth.shape
        depth_labels = depth_labels.view(B, N, H, W, C).permute(0,1,4,2,3).contiguous()
        # mix depth 
        # # viz depth
        # depth_range = torch.arange(2,42,0.5)
        # depth_range = depth_range.to(depth.device)
        # for i in range(len(depth[0])):
        #     depth_viz = (depth * depth_range[None,None,:,None,None]).sum(2)[0,i]
        #     depth_labels_viz = (depth_labels * depth_range[None,None,:,None,None]).sum(2)[0,i]
        #     depth_viz = (depth_viz -2) / (42 - 2) * 255.0
        #     depth_viz = depth_viz.cpu().detach().numpy().astype(np.uint8)
        #     depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
        #     cv2.imwrite('test_depth_{}.png'.format(str(i)), depth_viz)

        #     # depth_labels_viz = gt_depth[0,0]
        #     depth_labels_viz = (depth_labels_viz -2) / (42 - 2) * 255.0
        #     depth_labels_viz = depth_labels_viz.cpu().detach().numpy().astype(np.uint8)
        #     depth_labels_viz = cv2.applyColorMap(depth_labels_viz, cv2.COLORMAP_INFERNO)
        #     cv2.imwrite('test_depth_labels_{}.png'.format(str(i)), depth_labels_viz)
        # print(a)
        depth = depth_labels * (1-self.mix_coefficient) + depth * self.mix_coefficient
        if self.with_specific_component('forward_projection'):
            # bev_feat = self.forward_projection(cam_params, context, depth_labels, **kwargs)
            voxel_feat = self.forward_projection(cam_params, context, depth, **kwargs)
            return_map['cam_params'] = cam_params
        else:
            voxel_feat = None
        # elapsed = time.perf_counter() - start_time
        # print("elapsed:", elapsed)
        return_map['forward_voxel_feat'] = voxel_feat
        bev_feat = voxel_feat.mean(-1)
        # print("voxel_feat:", (voxel_feat==0).sum() / len(voxel_feat.reshape(-1)))
        # print("bev_feat:", (bev_feat==0).sum() / len(bev_feat.reshape(-1)))
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        bev_feat = self.fuse_history(bev_feat, img_metas, img[6])
        return_map['forward_bev_feat'] = bev_feat

        bev_feat = self.bev_encoder(bev_feat)
        bev_feat = torch.cat(bev_feat, dim=1)
        return_map['img_bev_feat'] = bev_feat
        # elapsed = time.perf_counter() - start_time
        # print("time_bev_encoder_elapsed:", elapsed)
        return return_map


    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        results={}
        if img is not None and self.with_specific_component('image_encoder'):
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))

        return results


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occupancy_flow=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        results= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if  self.with_pts_bbox:
            losses_pts = self.forward_pts_train(results['img_bev_feat'], gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
            losses.update(losses_pts)
        if self.with_specific_component('occupancy_head'):
            bev_gt_occupancy = kwargs['gt_occupancy'].max(-1)[0]
            losses_occupancy = self.occupancy_head.forward_train(results['img_bev_feat'], results=results, gt_occupancy=kwargs['gt_occupancy'], gt_occupancy_flow=gt_occupancy_flow)
            losses.update(losses_occupancy)
        if self.with_specific_component('frpn'):
            losses_mask = self.frpn.get_bev_mask_loss(kwargs['gt_bev_mask'], results['bev_mask_logit'])
            losses.update(losses_mask)

        if self.use_depth_supervision and self.with_specific_component('depth_net'):
            loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        self.do_history = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
                if not isinstance(var, list) :
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))        
            num_augs = len(img_inputs)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))

            if num_augs==1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                outputs = self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
                # elapsed = time.perf_counter() - start_time
                # print("total elapsed 2:", elapsed)
                return outputs
            else:
                return self.aug_test(points, img_metas, img_inputs, **kwargs)
        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        
    def aug_test(self,points,
                    img_metas,
                    img_inputs=None,
                    visible_mask=[None],
                    **kwargs):
        """Test function without augmentaiton."""
        assert False
        return None

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        

        bbox_list = [dict() for _ in range(len(img_metas))]
        
        if  self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        if self.with_specific_component('occupancy_head'):
            pred_occupancy = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)['output_voxels']
            # elapsed = time.perf_counter() - start_time
            # print("occ elapsed:", elapsed)
            # return 
        #     # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0] 
            pred_occupancy = pred_occupancy.softmax(-1)
            # torch.cuda.synchronize()
            # convert to CVPR2023 Format
            pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
            pred_occupancy = torch.flip(pred_occupancy, [2])
            pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
            pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
            
            if return_raw_occ:
                pred_occupancy_category = pred_occupancy
            else:
                pred_occupancy_category = pred_occupancy.argmax(-1) 
            
            # # do not change the order
            # if self.occupancy_save_path is not None:
            #     scene_name = img_metas[0]['scene_name']
            #     sample_token = img_metas[0]['sample_idx']
            #     mask_camera = visible_mask[0][0]
            #     pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
            #     pred_occupancy[mask_camera.cpu().numpy()==0] = 17
            #     save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', scene_name+'_'+sample_token)
            #     np.savez_compressed(save_path, pred=pred_occupancy, sample_token=sample_token) 

             # do not change the order
            if self.occupancy_save_path is not None:
                sample_token = img_metas[0]['sample_idx']
                pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                save_path = os.path.join(self.occupancy_save_path, sample_token + '.npz')
                np.savez_compressed(save_path, pred=pred_occupancy) 

            pred_occupancy_category = pred_occupancy_category.cpu().numpy()
            # elapsed = time.perf_counter() - start_time
            # print("pred elapsed:", elapsed) 

        else:
            pred_occupancy_category =  None
        if results.get('bev_mask_logit', None) is not None:
            pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
            iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
        else:
            iou = None

        assert len(img_metas) == 1

        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
        return bbox_list

        # if img_metas[0]['scene_name'] in ['scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 
        #     'scene-1067', 'scene-1068', 'scene-1069', 'scene-1070']:
        #     for i, result_dict in enumerate(bbox_list):
        #         result_dict['pts_bbox'] = bbox_pts[i]
        #         result_dict['iou'] = iou
        #         result_dict['pred_occupancy'] = pred_occupancy_category
        #         result_dict['index'] = img_metas[0]['index']
        #     return bbox_list
        # else:
        #     for i, result_dict in enumerate(bbox_list):
        #         result_dict['pts_bbox'] = bbox_pts[i]
        #         result_dict['iou'] = iou
        #         result_dict['pred_occupancy'] = None
        #         result_dict['index'] = img_metas[0]['index']
        #     return bbox_list
        # return pred_occupancy_category


    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs

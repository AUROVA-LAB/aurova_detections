import pytorch_lightning as pl
import argparse

import pytorch_lightning.utilities.distributed
import torch
import yaml
from easydict import EasyDict
import os
import sys
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
import pyquaternion

from datasets import points_utils, data_classes
from models import get_model
from nuscenes.utils import geometry_utils
from vision_msgs.msg import BoundingBox3D
from sensor_msgs.msg import PointCloud2
from pyquaternion import Quaternion
import time, copy



def predict(net, prev_frame, this_frame, ref_box):
    ref_box = rosBB2netBBox(ref_box)
    prev_frame = data_classes.PointCloud(prev_frame)
    this_frame = data_classes.PointCloud(this_frame)
    prev_frame_pc = points_utils.generate_subwindow(prev_frame, ref_box,
                                                        scale=net.config.bb_scale,
                                                        offset=net.config.bb_offset,oriented=False)
    this_frame_pc = points_utils.generate_subwindow(this_frame, ref_box,
                                                    scale=net.config.bb_scale,
                                                    offset=net.config.bb_offset,oriented=False)

    canonical_box = points_utils.transform_box(ref_box, ref_box)
    prev_points, idx_prev = points_utils.regularize_pc(prev_frame_pc.points.T,
                                                        net.config.point_sample_size,
                                                        seed=1)

    this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T,
                                                        net.config.point_sample_size,
                                                        seed=1)
    seg_mask_prev = geometry_utils.points_in_box(canonical_box, prev_points.T, 1.25).astype(float)

    # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
    # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
    seg_mask_prev[seg_mask_prev == 0] = 0.2
    seg_mask_prev[seg_mask_prev == 1] = 0.8
    seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

    timestamp_prev = np.full((net.config.point_sample_size, 1), fill_value=0)
    timestamp_this = np.full((net.config.point_sample_size, 1), fill_value=0.1)
    prev_points = np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
    this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

    stack_points = np.concatenate([prev_points, this_points], axis=0)

    data_dict = {"points": torch.tensor(stack_points[None, :], device=net.device, dtype=torch.float32),
                     }
    if getattr(net.config, 'box_aware', False):
        candidate_bc_prev = points_utils.get_point_to_box_distance(
            stack_points[:net.config.point_sample_size, :3], canonical_box)
        candidate_bc_this = np.zeros_like(candidate_bc_prev)
        candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)
        data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                            device=net.device)})
    
    end_points = net(data_dict)
    estimation_box = end_points['estimation_boxes']
    estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

    if len(estimation_box.shape) == 3:
        best_box_idx = estimation_box_cpu[:, 4].argmax()
        estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

    candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=net.config.degrees,
                                                use_z=net.config.use_z,
                                                limit_box=net.config.limit_box)
    
    candidate_box =  netBB2rosBB(candidate_box)
    return candidate_box

def get_net(dir):
    config_file = os.path.join(dir, 'cfgs','M2_track_kitti.yaml')
    checkpoint_file=os.path.join(dir, 'pretrained_models','mmtrack_kitti_pedestrian.ckpt')
    cfg = load_yaml(config_file)
    cfg.update({'checkpoint' : checkpoint_file})
    cfg = EasyDict(cfg)

    # init model
    net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)
    net.eval()
    return net

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

def rosBB2netBBox(rosBB : BoundingBox3D):
    center = [rosBB.center.position.x, rosBB.center.position.y, rosBB.center.position.z]
    size = [rosBB.size.x,rosBB.size.y, rosBB.size.z]
    orientation = Quaternion(rosBB.center.orientation.w, rosBB.center.orientation.x, rosBB.center.orientation.y, rosBB.center.orientation.z)
    netBB=data_classes.Box(center, size, orientation)
    return netBB

def netBB2rosBB(netBB : data_classes.Box):
    rosBB = BoundingBox3D()
    rosBB.center.position.x = netBB.center[0]; rosBB.center.position.y = netBB.center[1]; rosBB.center.position.z = netBB.center[2]
    rosBB.center.orientation.w = netBB.orientation.scalar; rosBB.center.orientation.x=netBB.orientation.vector[0]
    rosBB.center.orientation.y = netBB.orientation.vector[1]; rosBB.center.orientation.z=netBB.orientation.vector[2]
    rosBB.size.x=netBB.wlh[0]; rosBB.size.y=netBB.wlh[1]; rosBB.size.z=netBB.wlh[2]
    return rosBB

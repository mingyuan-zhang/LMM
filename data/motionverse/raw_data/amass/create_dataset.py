import os
import json
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
from mogen.datasets.utils import  (
    create_data_item,
    ang2joint
)
from mogen.models.utils.imagebind_wrapper import (
    imagebind_huge
)
from scipy.spatial.transform import Rotation as R

root_path = "../../datasets/amass"
amass_dir = 'amass_data'
motion_dir = 'pose_data'
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_motions_dir = os.path.join(root_path, 'eval_motions')
os.makedirs(eval_motions_dir, exist_ok=True)
    
    
skeleton_info = np.load("../../body_models/smpl_skeleton.npz")
p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
parents = skeleton_info['parents']
parent = {}
for i in range(len(parents)):
    parent[i] = parents[i]
cnt = 0
for line in tqdm(open('sequence_name.txt')):
    filename = line.strip()
    basename = '%06d' % cnt
    cnt += 1
    pose_data = np.load(os.path.join(motion_dir, filename + '.npy'))
    amass_data = np.load(os.path.join(amass_dir, filename + '.npz'))
    N = pose_data.shape[0]
    framerate = float(amass_data['mocap_framerate'])
    meta_data = {
        'framerate': framerate,
        'has_root': True,
        'has_head': True,
        'has_stem': True,
        'has_larm': True,
        'has_rarm': True,
        'has_lleg': True,
        'has_rleg': True,
        'has_lhnd': True,
        'has_rhnd': True,
        'has_face': False,
        'num_frames': int(pose_data.shape[0])
    }
    create_data_item(
        keypoints3d=pose_data, 
        meta_data=meta_data,
        root_dir=root_path,
        basename=basename)
    if 'BioMotionLab_NTroje' in filename:
        sample_rate = int(framerate // 25)
        sampled_index = np.arange(0, N, sample_rate)
        amass_motion_poses = amass_data['poses']
        amass_motion_poses = amass_motion_poses[sampled_index]
        T = amass_motion_poses.shape[0]
        amass_motion_poses = R.from_rotvec(amass_motion_poses.reshape(-1, 3)).as_rotvec()
        amass_motion_poses = amass_motion_poses.reshape(T, 52, 3)
        amass_motion_poses[:, 0] = 0
        p3d0_tmp = p3d0.repeat([amass_motion_poses.shape[0], 1, 1])
        amass_motion_poses = ang2joint(p3d0_tmp, torch.tensor(amass_motion_poses).float(), parent).reshape(-1, 52, 3)[:, 4:22]
        
        eval_motion_path = os.path.join(eval_motions_dir, basename + '.npy')
        np.save(eval_motion_path, amass_motion_poses.reshape((T, 18, 3)))
# ------------------------------------------------------------------------------------------------
# Copyright (c) Chuan Guo.
# ------------------------------------------------------------------------------------------------
# This code were adapted from the following open-source project:
# https://github.com/EricGuo5513/HumanML3D
# ------------------------------------------------------------------------------------------------

import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from mogen.datasets.utils import BodyModelWrapper

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Choose the device to run the body model on.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = []
folders = []
dataset_names = []
for root, dirs, files in os.walk('./amass_data'):
    folders.append(root)
    for name in files:
        if not 'npz' in name:
            continue
        dataset_name = root.split('/')[2]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        paths.append(os.path.join(root, name))
        
save_root = './pose_data'
save_folders = [folder.replace('./amass_data', './pose_data') for folder in folders]
for folder in save_folders:
    os.makedirs(folder, exist_ok=True)
group_path = [[path for path in paths if name in path and "npz" in path] for name in dataset_names]

trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])
# ex_fps = 20
model = BodyModelWrapper(device)
def amass_to_pose(src_path, save_path):
    bdata = np.load(src_path, allow_pickle=True)
    if 'mocap_framerate' not in bdata:
        return
    pose_seq_np = model.process_smplh(bdata)
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    np.save(save_path, pose_seq_np_n)


group_path = group_path
all_count = sum([len(paths) for paths in group_path])
cur_count = 0

import time
for paths in group_path:
    dataset_name = paths[0].split('/')[2]
    pbar = tqdm(paths)
    pbar.set_description('Processing: %s'%dataset_name)
    for path in pbar:
        if os.path.splitext(path)[-1] != '.npz':
            continue
        save_path = path.replace('./amass_data', './pose_data')
        save_path = save_path[:-3] + 'npy'
        amass_to_pose(path, save_path)
        
    cur_count += len(paths)
    print('Processed / All: %d/%d'% (cur_count, all_count))
    time.sleep(0.5)

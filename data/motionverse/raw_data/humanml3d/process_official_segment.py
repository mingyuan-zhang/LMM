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
for root, dirs, files in os.walk('../amass/amass_data'):
    folders.append(root)
    for name in files:
        if not 'npz' in name:
            continue
        dataset_name = root.split('/')[3]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        paths.append(os.path.join(root, name))
        
save_root = './pose_data'
save_folders = [folder.replace('../amass/amass_data', './pose_data') for folder in folders]
for folder in save_folders:
    os.makedirs(folder, exist_ok=True)
group_path = [[path for path in paths if name in path and "npz" in path] for name in dataset_names]

trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])
ex_fps = 20
model = BodyModelWrapper(device)
def amass_to_pose(src_path, save_path):
    bdata = np.load(src_path, allow_pickle=True)
    if 'mocap_framerate' not in bdata:
        return
    fps = bdata['mocap_framerate']
    down_sample = int(fps / ex_fps)
    pose_seq_np = model.process_smplh(bdata, down_sample)
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    np.save(save_path, pose_seq_np_n)
    return fps


group_path = group_path
all_count = sum([len(paths) for paths in group_path])
cur_count = 0

import time
for paths in group_path:
    dataset_name = paths[0].split('/')[3]
    pbar = tqdm(paths)
    pbar.set_description('Processing: %s'%dataset_name)
    for path in pbar:
        if os.path.splitext(path)[-1] != '.npz':
            continue
        save_path = path.replace('../amass/amass_data', './pose_data')
        save_path = save_path[:-3] + 'npy'
        fps = amass_to_pose(path, save_path)
        
    cur_count += len(paths)
    print('Processed / All: %d/%d'% (cur_count, all_count) )
    time.sleep(0.5)


import os
import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
import pickle as pkl

def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

index_path = './index.csv'
save_dir = './official_joints'
os.makedirs(save_dir, exist_ok=True)

index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]
fps = 20
framerate_dict = {}

for i in tqdm(range(total_amount)):
    source_path = index_file.loc[i]['source_path']
    new_name = index_file.loc[i]['new_name']
    data = np.load(source_path)
    start_frame = index_file.loc[i]['start_frame']
    end_frame = index_file.loc[i]['end_frame']
    if 'humanact12' not in source_path:
        if 'Eyes_Japan_Dataset' in source_path:
            data = data[3 * fps:]
        if 'MPI_HDM05' in source_path:
            data = data[3 * fps:]
        if 'TotalCapture' in source_path:
            data = data[1 * fps:]
        if 'MPI_Limits' in source_path:
            data = data[1 * fps:]
        if 'Transitions_mocap' in source_path:
            data = data[int(0.5 * fps):]
        data = data[start_frame:end_frame]
        data[..., 0] *= -1
    
    data_m = swap_left_right(data)
    np.save(pjoin(save_dir, new_name), data)
    np.save(pjoin(save_dir, 'M'+new_name), data_m)
    

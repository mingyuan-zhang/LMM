# ------------------------------------------------------------------------------------------------
# Copyright (c) Chuan Guo.
# ------------------------------------------------------------------------------------------------
# This code were adapted from the following open-source project:
# https://github.com/EricGuo5513/HumanML3D
# ------------------------------------------------------------------------------------------------

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
save_dir = './joints'
os.makedirs(save_dir, exist_ok=True)

index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]
fps = 20
framerate_dict = {}

for i in tqdm(range(total_amount)):
    source_path = index_file.loc[i]['source_path']
    new_name = index_file.loc[i]['new_name']
    if 'humanact12' not in source_path:
        amass_path = source_path.replace('./pose_data', '../amass/amass_data').replace('npy', 'npz')
        amass_data = np.load(amass_path)
        framerate = amass_data['mocap_framerate']
        down_sample = int(framerate / fps)
        pose_path = source_path.replace('./pose_data', '../amass/pose_data')
        data = np.load(pose_path)
    else:
        framerate = 20
        down_sample = 1
        data = np.load(source_path)
    start_frame = index_file.loc[i]['start_frame'] * down_sample
    end_frame = index_file.loc[i]['end_frame'] * down_sample
    if 'humanact12' not in source_path:
        if 'Eyes_Japan_Dataset' in source_path:
            data = data[3 * fps * down_sample:]
        if 'MPI_HDM05' in source_path:
            data = data[3 * fps * down_sample:]
        if 'TotalCapture' in source_path:
            data = data[1 * fps * down_sample:]
        if 'MPI_Limits' in source_path:
            data = data[1 * fps * down_sample:]
        if 'Transitions_mocap' in source_path:
            data = data[int(0.5 * fps * down_sample):]
        data = data[start_frame:end_frame]
        data[..., 0] *= -1
    
    data_m = swap_left_right(data)
    np.save(pjoin(save_dir, new_name), data)
    np.save(pjoin(save_dir, 'M'+new_name), data_m)
    framerate_dict[new_name] = framerate
    framerate_dict['M' + new_name] = framerate

with open('framerate_dict.pkl', 'wb') as handle:
    pkl.dump(framerate_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
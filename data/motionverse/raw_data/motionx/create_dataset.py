import os
import json
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
from mogen.datasets.utils import  (
    create_data_item,
    extract_text_feature,
    BodyModelWrapper
)
from mogen.models.utils.imagebind_wrapper import (
    imagebind_huge
)



device = "cuda" if torch.cuda.is_available() else "cpu"
model = imagebind_huge(pretrained=True)
model.eval()
model.to(device)

root_path = "../../datasets/motionx"
text_tgt_dir = os.path.join(root_path, 'texts')
os.makedirs(text_tgt_dir, exist_ok=True)
text_feats_dir = os.path.join(root_path, 'text_feats')
os.makedirs(text_feats_dir, exist_ok=True)

motion_dir = 'motion_data'
all_paths = []
def traverse(root_path):
    for filename in os.listdir(root_path):
        new_path = os.path.join(root_path, filename)
        if filename[-3:] == 'npy':
            if not 'aist' in new_path: # aist is used for music2dance
                all_paths.append(new_path)
        elif os.path.isdir(new_path):
            traverse(new_path)

traverse(motion_dir)
trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])
body_model = BodyModelWrapper(device)

fout = open('seq_name.txt', 'w')
for idx, filepath in enumerate(tqdm(all_paths)):
    basename = '%06d' % idx
    fout.write(filepath + '\n')
    motion_data = np.load(filepath)
    framerate = 30
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
        'has_face': True,
        'num_frames': int(motion_data.shape[0])
    }
    bdata = {}
    bdata['poses'] = motion_data[:, :156]
    bdata['trans'] = motion_data[:, 309: 312]
    bdata['betas'] = motion_data[0, 312:]
    pose_seq_np = body_model.process_smplh(bdata)
    # pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    
    text_src_path = filepath.replace('motion_data/smplx_322', 'motionx_seq_text_v1.1')
    text_src_path = text_src_path.replace('npy', 'txt')
    create_data_item(
        keypoints3d=pose_seq_np,
        expression=motion_data[:, 159: 159 + 50],
        smpl_rot=motion_data[:, :156],
        meta_data=meta_data,
        root_dir=root_path,
        basename=basename)
    all_anno = []
    seq_caption = []
    seq_start = 0
    seq_end = int(motion_data.shape[0])
    
    text_word_feats = []
    text_seq_feats = []
    
    for line in open(text_src_path):
        line = line.strip()
        seq_caption.append(line)
            
    if len(seq_caption) > 0:
        all_anno.append(
            {
                'start_frame': seq_start,
                'end_frame': seq_end,
                'body_text': seq_caption,
            }
        )
        text_word_feat, text_seq_feat = extract_text_feature(model, seq_caption, device)
        text_word_feats.append(text_word_feat.cpu().numpy())
        text_seq_feats.append(text_seq_feat.cpu().numpy())
    
    text_tgt_path = os.path.join(text_tgt_dir, basename + '.json')
    json.dump(all_anno, open(text_tgt_path, 'w'), indent=4)
    text_feats_path = os.path.join(text_feats_dir, basename + '.pkl')
    text_feats_dict = {
        'text_word_feats': text_word_feats,
        'text_seq_feats': text_seq_feats
    }
    with open(text_feats_path, 'wb') as handle:
        pkl.dump(text_feats_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
import os
import json
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
from mogen.datasets.utils import  (
    create_data_item,
    extract_text_feature
)
from mogen.models.utils.imagebind_wrapper import (
    imagebind_huge
)

motion_dir = 'joints'
framerate_dict = pkl.load(open('framerate_dict.pkl', 'rb'))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = imagebind_huge(pretrained=True)
model.eval()
model.to(device)

root_path = "../../datasets/humanml3d"
text_tgt_dir = os.path.join(root_path, 'texts')
os.makedirs(text_tgt_dir, exist_ok=True)
text_feats_dir = os.path.join(root_path, 'text_feats')
os.makedirs(text_feats_dir, exist_ok=True)
    
for filename in tqdm(os.listdir(motion_dir)):
    basename = filename.split(".")[0]
    motion_data = np.load(os.path.join(motion_dir, filename))
    framerate = float(framerate_dict[filename])
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
        'num_frames': int(motion_data.shape[0])
    }
    if motion_data.shape[1] == 24:
        meta_data['has_lhnd'] = False
        meta_data['has_rhnd'] = False
    create_data_item(
        keypoints3d=motion_data, 
        meta_data=meta_data,
        root_dir=root_path,
        basename=basename)
    text_src_path = os.path.join('texts', basename + '.txt')
    all_anno = []
    seq_caption = []
    seq_tokens = []
    seq_start = 0
    seq_end = int(motion_data.shape[0])
    
    text_word_feats = []
    text_seq_feats = []
    for line in open(text_src_path):
        line = line.strip()
        content = line.split("#")
        caption = content[0]
        tokens = content[1]
        if content[2] == 'nan' or content[3] == 'nan':
            continue
        start_time = float(content[2])
        end_time = float(content[3])
        if start_time == 0.0 and end_time == 0.0:
            seq_caption.append(caption)
            seq_tokens.append(tokens)
        else:
            anno = {
                'start_frame': int(start_time * framerate),
                'end_frame': min(int(end_time * framerate), seq_end),
                'body_text': [caption],
                'body_tokens': [tokens]
            }
            all_anno.append(anno)
            text_word_feat, text_seq_feat = extract_text_feature(model, [caption], device)
            text_word_feats.append(text_word_feat.cpu().numpy())
            text_seq_feats.append(text_seq_feat.cpu().numpy())
            
    if len(seq_caption) > 0:
        all_anno.append(
            {
                'start_frame': seq_start,
                'end_frame': seq_end,
                'body_text': seq_caption,
                'body_tokens': seq_tokens
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
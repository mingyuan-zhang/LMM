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

root_path = "../../datasets/babel"
text_tgt_dir = os.path.join(root_path, 'texts')
os.makedirs(text_tgt_dir, exist_ok=True)
text_feats_dir = os.path.join(root_path, 'text_feats')
os.makedirs(text_feats_dir, exist_ok=True)

pose_dir = '../amass/pose_data'
motion_dir = '../amass/amass_data'
fout = open('train.txt', 'w')
for split in ['train', 'val']:
    anno_file = os.path.join('babel_v1.0_release', split + '.json')
    anno_data = json.load(open(anno_file))
    print('Process ' + str(split))
    for k, v in tqdm(anno_data.items()):
        basename = k
        rel_path = '/'.join(v['feat_p'].split('/')[1:])
        amass_path = os.path.join(motion_dir, rel_path)
        amass_data = np.load(amass_path)
        
        pose_path = os.path.join(pose_dir, rel_path[:-1] + 'y')
        pose_data = np.load(pose_path)
        pose_data[:, :, 2] *= -1
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
            smpl_rot=amass_data['poses'].reshape((pose_data.shape[0], -1)),
            meta_data=meta_data,
            root_dir=root_path,
            basename=basename)

        all_anno = []
        
        text_word_feats = []
        text_seq_feats = []
        
        if v['seq_ann']['mul_act'] and v['frame_ann'] is None:
            continue
        
        fout.write(k + '\n')
        if not v['seq_ann']['mul_act']:
            seq_caption = [v['seq_ann']['labels'][0]['proc_label']]
            all_anno.append(
                {
                    'start_frame': 0,
                    'end_frame': int(pose_data.shape[0]),
                    'body_text': seq_caption,
                }
            )
            text_word_feat, text_seq_feat = extract_text_feature(model, seq_caption, device)
            text_word_feats.append(text_word_feat.cpu().numpy())
            text_seq_feats.append(text_seq_feat.cpu().numpy())
            
        if v['frame_ann'] is not None:
            for anno in v['frame_ann']['labels']:
                start_frame = int(anno['start_t'] * framerate)
                end_frame = int(anno['end_t'] * framerate)
                caption = [anno['proc_label']]
                all_anno.append(
                    {
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'body_text': caption,
                    }
                )
                text_word_feat, text_seq_feat = extract_text_feature(model, caption, device)
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
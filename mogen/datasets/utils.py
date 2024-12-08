import os
import json
import numpy as np
import torch
from imagebind import data
from imagebind.models.imagebind_model import ModalityType
from mogen.datasets.human_body_prior.body_model.body_model import BodyModel
from mogen.datasets.quaternion import qrot, qinv
from pytorch3d.transforms import axis_angle_to_matrix


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4, )).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3, )).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    '''Add Y-axis rotation to local joints'''
    rot = qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4, ))
    positions = qrot(rot, positions)
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def create_data_item(meta_data,
                     root_dir,
                     basename,
                     tomato_repr=None,
                     keypoints3d=None,
                     expression=None,
                     smpl_rot=None,
                     bvh_rot=None):
    assert os.path.exists(root_dir)
    meta_dir = os.path.join(root_dir, 'metas')
    motion_dir = os.path.join(root_dir, 'motions')
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)
    
    motion_data = {}
    
    if tomato_repr is not None:
        motion_data['tomato_repr'] = tomato_repr
    if keypoints3d is not None:
        motion_data['keypoints3d'] = keypoints3d
        num_frames = keypoints3d.shape[0]
        keypoints3d = keypoints3d.reshape((num_frames, -1))
    if expression is not None:
        motion_data['expression'] = expression
    if smpl_rot is not None:
        motion_data['smpl_rot'] = smpl_rot
    if bvh_rot is not None:
        motion_data['bvh_rot'] = bvh_rot
        
    motion_path = os.path.join(motion_dir, basename + '.npz')
    meta_path = os.path.join(meta_dir, basename + '.json')
    np.savez_compressed(motion_path, **motion_data)
    json.dump(meta_data, open(meta_path, 'w'), indent=4)


def extract_text_feature(model, text, device):
    text_list = text
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    }
    with torch.no_grad():
        text_word_feat, text_seq_feat = model(inputs)
    return text_word_feat, text_seq_feat


def extract_image_feature(model, image_paths, device):
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }
    with torch.no_grad():
        _, embeddings = model(inputs)
    return embeddings


def extract_audio_feature(model, audio_paths, device):
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device, clips_per_video=1),
    }
    with torch.no_grad():
       audio_word_feat, audio_seq_feat = model(inputs)
    return audio_word_feat, audio_seq_feat


def copy_repr_data(src_data, src_idx, num_src_joints, tgt_data, tgt_idx, num_tgt_joints):
    # ric_data
    tgt_base1 = 4 + (tgt_idx - 1) * 3
    src_base1 = 4 + (src_idx - 1) * 3
    tgt_data[:, tgt_base1: tgt_base1 + 3] = \
        src_data[:, src_base1: src_base1 + 3]
    # rot_data
    tgt_base2 = 4 + (num_tgt_joints - 1) * 3 + (tgt_idx - 1) * 6
    src_base2 = 4 + (num_src_joints - 1) * 3 + (src_idx - 1) * 6
    tgt_data[:, tgt_base2: tgt_base2 + 6] = \
        src_data[:, src_base2: src_base2 + 6]
    # local velocity
    tgt_base3 = 4 + (num_tgt_joints - 1) * 9 + tgt_idx * 3
    src_base3 = 4 + (num_src_joints - 1) * 9 + src_idx * 3
    tgt_data[:, tgt_base3: tgt_base3 + 3] = \
        src_data[:, src_base3: src_base3 + 3]


def extract_repr_data(data, idx, num_joints):
    assert idx > 0
    base1 = 4 + (idx - 1) * 3
    ric_data = data[:, base1: base1 + 3]
    base2 = 4 + (num_joints - 1) * 3 + (idx - 1) * 6
    rot_data = data[:, base2: base2 + 6]
    base3 = 4  + (num_joints - 1) * 9 + idx * 3
    local_vel = data[:, base3: base3 + 3]
    if isinstance(data, torch.Tensor):
        output = torch.cat((ric_data, rot_data, local_vel), dim=-1)
    else:
        output = np.concatenate((ric_data, rot_data, local_vel), axis=-1)
    return output


def move_repr_data(data, idx, num_joints, output):
    assert idx > 0
    assert data.shape[1] == 12
    base1 = 4 + (idx - 1) * 3
    output[:, base1: base1 + 3] = data[:, :3]
    base2 = 4 + (num_joints - 1) * 3 + (idx - 1) * 6
    output[:, base2: base2 + 6] = data[:, 3: 9]
    base3 = 4  + (num_joints - 1) * 9 + idx * 3
    output[:, base3: base3 + 3] = data[:, 9:]


def estimate_repr_data(data, idx1, idx2, tgt, ratio, num_joints):
    # direction: same as idx1
    # position: |idx1 - tgt| / |idx1 - idx2| = ratio
    assert 0 <= ratio <= 1, "ratio should be between 0 and 1"
    assert 1 <= idx1 <= num_joints, "idx1 out of range"
    assert 1 <= idx2 <= num_joints, "idx2 out of range"
    assert 1 <= tgt <= num_joints, "tgt out of range"
    
    # ric data
    base1 = 4 + (idx1 - 1) * 3
    base2 = 4 + (idx2 - 1) * 3
    baset = 4 + (tgt - 1) * 3
    pose1 = data[:, base1: base1 + 3]
    pose2 = data[:, base2: base2 + 3]
    poset = pose1 * (1 - ratio) + pose2 * ratio
    data[:, baset: baset + 3] = poset
    
    # rot_data
    base1 = 4 + (num_joints - 1) * 3 + (idx1 - 1) * 6
    baset = 4 + (num_joints - 1) * 3 + (tgt - 1) * 6
    data[:, baset: baset + 6] = data[:, base1: base1 + 6]
    
    # local velocity
    base1 = 4 + (num_joints - 1) * 9 + idx1 * 3
    base2 = 4 + (num_joints - 1) * 9 + idx2 * 3
    baset = 4 + (num_joints - 1) * 9 + tgt * 3
    vel1 = data[:, base1: base1 + 3]
    vel2 = data[:, base2: base2 + 3]
    velt = vel1 * (1 - ratio) + vel2 * ratio
    data[:, baset: baset + 3] = velt
    

class BodyModelWrapper:
    
    def __init__(self, device):
        file_path = os.path.abspath(os.path.dirname(__file__))
        body_model_dir = os.path.join(file_path, '../../data/motionverse/body_models')
        male_bm_path = os.path.join(body_model_dir, 'smplh/male/model.npz')
        male_dmpl_path = os.path.join(body_model_dir, 'dmpls/male/model.npz')
        female_bm_path = os.path.join(body_model_dir, 'smplh/female/model.npz')
        female_dmpl_path = os.path.join(body_model_dir, 'dmpls/female/model.npz')
        neutral_bm_path = os.path.join(body_model_dir, 'smplh/neutral/model.npz')
        neutral_dmpl_path = os.path.join(body_model_dir, 'dmpls/neutral/model.npz')

        self.num_betas = 10 # number of body parameters
        self.num_dmpls = 8 # number of DMPL parameters

        self.male_bm = BodyModel(
            bm_fname=male_bm_path,
            num_betas=self.num_betas,
            num_dmpls=self.num_dmpls,
            dmpl_fname=male_dmpl_path).to(device)
        self.female_bm = BodyModel(
            bm_fname=female_bm_path,
            num_betas=self.num_betas,
            num_dmpls=self.num_dmpls,
            dmpl_fname=female_dmpl_path).to(device)
        self.neutral_bm = BodyModel(
            bm_fname=neutral_bm_path,
            num_betas=self.num_betas,
            num_dmpls=self.num_dmpls,
            dmpl_fname=neutral_dmpl_path).to(device)
        self.device = device
        
    def process_smplh(self, smplh_data, downsample=1):
        poses = smplh_data['poses'][::downsample]
        trans = smplh_data['trans'][::downsample]
        betas = smplh_data['betas']
        if len(betas.shape) == 1:
            betas = betas[:self.num_betas][np.newaxis]
            betas = np.repeat(betas, repeats=len(trans), axis=0)
        else:
            betas = betas[:, :self.num_betas]
        body_parms = {
            'root_orient': torch.Tensor(poses[:, :3]).to(self.device),
            'pose_body': torch.Tensor(poses[:, 3:66]).to(self.device),
            'pose_hand': torch.Tensor(poses[:, 66:]).to(self.device),
            'trans': torch.Tensor(trans).to(self.device),
            'betas': torch.Tensor(betas).to(self.device),
        }
        gender = smplh_data.get('gender', 'neutral')
        if gender == 'male' or gender == 'm':
            bm = self.male_bm
        elif gender == 'female' or gender == 'f':
            bm = self.female_bm
        else:
            bm = self.neutral_bm
        with torch.no_grad():
            body = bm(**body_parms)
        pose_seq_np = body.Jtr.detach().cpu().numpy()
        return pose_seq_np


def ang2joint(p3d0, pose,
              parent={0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
                      15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}):
    """

    :param p3d0:[batch_size, joint_num, 3]
    :param pose:[batch_size, joint_num, 3]
    :param parent:
    :return:
    """
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

        Parameter:
        ---------
        x: Tensor to be appended.

        Return:
        ------
        Tensor after appending of shape [4,4]

        """
        ones = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
        ).expand(x.shape[0], -1, -1).to(x.device)
        ret = torch.cat((x, ones), dim=1)
        return ret
    batch_num = p3d0.shape[0]
    jnum = len(parent.keys())
    J = p3d0
    R_cube_big = axis_angle_to_matrix(pose.contiguous().view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)
    results = []
    results.append(
        with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )
    for i in range(1, jnum):
        results.append(
            torch.matmul(
                results[parent[i]],
                with_zeros(
                    torch.cat(
                        (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
                        dim=2
                    )
                )
            )
        )

    stacked = torch.stack(results, dim=1)
    J_transformed = stacked[:, :, :3, 3]
    return J_transformed
import numpy as np
import torch

from ..builder import PIPELINES
from ..quaternion import qbetween_np, qinv_np, qmul_np, qrot_np

face_joint_indx = [2, 1, 17, 16]
fid_l = [7, 10]
fid_r = [8, 11]

trans_matrix = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                             [0.0, -1.0, 0.0]])


def rigid_transform(relative, data):

    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    global_vel = data[..., 22 * 3:22 * 6].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4, ))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)
    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    global_positions[..., [0, 2]] += relative_t
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1, ))
    global_vel = qrot_np(qinv_np(relative_r_rot_quat), global_vel)
    data[..., 22 * 3:22 * 6] = global_vel.reshape(data.shape[:-1] + (-1, ))

    return data


@PIPELINES.register_module()
class SwapSiameseMotion(object):
    r"""Swap motion sequences.

    Args:
        prob (float): The probability of swapping siamese motions
    """

    def __init__(self, prob=0.5):
        self.prob = prob
        assert prob >= 0 and prob <= 1.0

    def __call__(self, results):
        if np.random.rand() <= self.prob:
            motion1 = results['motion1']
            motion2 = results['motion2']
            results['motion1'] = motion2
            results['motion2'] = motion1
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class ProcessSiameseMotion(object):
    r"""Process siamese motion sequences.
        The code is borrowed from
            https://github.com/tr3e/InterGen/blob/master/utils/utils.py
    """

    def __init__(self, feet_threshold, prev_frames, n_joints, prob):
        self.feet_threshold = feet_threshold
        self.prev_frames = prev_frames
        self.n_joints = n_joints
        self.prob = prob

    def process_single_motion(self, motion):
        feet_thre = self.feet_threshold
        prev_frames = self.prev_frames
        n_joints = self.n_joints
        '''Uniform Skeleton'''
        # positions = uniform_skeleton(positions, tgt_offsets)

        positions = motion[:, :n_joints * 3].reshape(-1, n_joints, 3)
        rotations = motion[:, n_joints * 3:]

        positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)
        '''Put on Floor'''
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height
        '''XZ at origin'''
        root_pos_init = positions[prev_frames]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz
        '''All initially face Z+'''
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across = root_pos_init[r_hip] - root_pos_init[l_hip]
        across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init**2).sum(axis=-1))
        forward_init = forward_init[..., np.newaxis]

        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init_for_all = \
            np.ones(positions.shape[:-1] + (4,)) * root_quat_init

        positions = qrot_np(root_quat_init_for_all, positions)
        """ Get Foot Contacts """

        def foot_detect(positions, thres):
            velfactor, heightfactor = \
                np.array([thres, thres]), np.array([0.12, 0.05])

            feet_l_x = \
                (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = \
                (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = \
                (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1, fid_l, 1]
            feet_l_sum = feet_l_x + feet_l_y + feet_l_z
            feet_l = ((feet_l_sum < velfactor) & (feet_l_h < heightfactor))
            feet_l = feet_l.astype(np.float32)

            feet_r_x = \
                (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = \
                (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = \
                (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1, fid_r, 1]
            feet_r_sum = feet_r_x + feet_r_y + feet_r_z
            feet_r = ((feet_r_sum < velfactor) & (feet_r_h < heightfactor))
            feet_r = feet_r.astype(np.float32)
            return feet_l, feet_r

        feet_l, feet_r = foot_detect(positions, feet_thre)
        '''Get Joint Rotation Representation'''
        rot_data = rotations
        '''Get Joint Rotation Invariant Position Represention'''
        joint_positions = positions.reshape(len(positions), -1)
        joint_vels = positions[1:] - positions[:-1]
        joint_vels = joint_vels.reshape(len(joint_vels), -1)

        data = joint_positions[:-1]
        data = np.concatenate([data, joint_vels], axis=-1)
        data = np.concatenate([data, rot_data[:-1]], axis=-1)
        data = np.concatenate([data, feet_l, feet_r], axis=-1)

        return data, root_quat_init, root_pose_init_xz[None]

    def __call__(self, results):
        motion1, root_quat_init1, root_pos_init1 = \
            self.process_single_motion(results['motion1'])
        motion2, root_quat_init2, root_pos_init2 = \
            self.process_single_motion(results['motion2'])
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:,
                                                                       [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)
        if np.random.rand() <= self.prob:
            motion2, motion1 = motion1, motion2
        motion = np.concatenate((motion1, motion2), axis=-1)
        results['motion'] = motion
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(feet_threshold={self.feet_threshold})'
        repr_str += f'(feet_threshold={self.feet_threshold})'
        repr_str += f'(n_joints={self.n_joints})'
        repr_str += f'(prob={self.prob})'
        return repr_str

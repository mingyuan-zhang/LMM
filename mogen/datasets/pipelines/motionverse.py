import random
from typing import Optional, Union, List
import pytorch3d.transforms as geometry

import numpy as np
import torch

from ..builder import PIPELINES
from ..skeleton import Skeleton
from ..paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from ..quaternion import (
    qbetween_np,
    qrot_np,
    qinv_np,
    qmul_np,
    quaternion_to_cont6d_np
)


@PIPELINES.register_module()
class LoadMotion(object):
    r"""Load motion data from a file.

    This pipeline component loads motion data from a specified file path provided in the `results` dictionary.
    It randomly selects a motion type from the given `motion_types` list and updates the `results` dictionary
    with the corresponding motion data based on the selected motion type.

    Args:
        motion_types (List[str]): A list of motion types to choose from. Possible values include:
            - `'tomato_repr'`: Load the 'tomato_repr' motion representation.
            - `'smpl_rot'`: Load SMPL rotation data.
            - `'bvh_rot'`: Load BVH rotation data.
            - `'h3d_rot'`: Calculate H3D rotation data(in another pipeline).
        max_size (int): The maximum size of the cropped motion
            sequence (inclusive). This is only used for `tomato_repr`
    """

    def __init__(self, motion_types: List[str], max_size: int = -1):
        self.motion_types = motion_types
        self.max_size = max_size

    def __call__(self, results):
        """Load motion data and update the results dictionary.

        Args:
            results (dict): A dictionary containing the key `'motion_path'`, which specifies the path to the motion data file.

        Returns:
            dict: The updated results dictionary with loaded motion data and the selected motion type.
        """
        # Load motion data from the specified file
        motion_path = results['motion_path']
        motion_data = np.load(motion_path)

        # Randomly select a motion type from the provided list
        motion_type = np.random.choice(self.motion_types)
        results['motion_type'] = motion_type

        if motion_type == 'tomato_repr':
            motion = motion_data['tomato_repr']
            length = motion.shape[0]
            assert self.max_size != -1
            actual_length = min(self.max_size, length)
            padding_length = self.max_size - actual_length
            if padding_length > 0:
                D = motion.shape[1:]
                padding_zeros = np.zeros((padding_length, *D), dtype=np.float32)
                motion = np.concatenate([motion, padding_zeros], axis=0)
            else:
                motion = motion[:actual_length]
            results['motion_length'] = actual_length
            results['motion'] = motion
            results['motion_shape'] = motion.shape
            motion_mask = torch.cat(
                (torch.ones(actual_length),
                    torch.zeros(padding_length)),
                dim=0).numpy()
            motion_mask = np.expand_dims(motion_mask, axis=1)
            motion_mask = np.repeat(motion_mask, 10, axis=1)
            meta_data = results['meta_data']
            if not meta_data['has_root']:
                motion_mask[:, 0] = 0
            if not meta_data['has_head']:
                motion_mask[:, 1] = 0
            if not meta_data['has_stem']:
                motion_mask[:, 2] = 0
            if not meta_data['has_larm']:
                motion_mask[:, 3] = 0
            if not meta_data['has_rarm']:
                motion_mask[:, 4] = 0
            if not meta_data['has_lleg']:
                motion_mask[:, 5] = 0
            if not meta_data['has_rleg']:
                motion_mask[:, 6] = 0
            if not meta_data['has_lhnd']:
                motion_mask[:, 7] = 0
            if not meta_data['has_rhnd']:
                motion_mask[:, 8] = 0
            if not meta_data['has_face']:
                motion_mask[:, 9] = 0
            results['motion_mask'] = motion_mask
            results['meta_data']['rotation_type'] = 'h3d_rot'
            if not 'video_word_feat' in results:
                results['video_word_feat'] = np.zeros((self.max_size, 1024))
        else:
            keypoints3d = motion_data['keypoints3d']
            if keypoints3d.shape[0] == 0:
                print(results['motion_path'])
            start_frame = results['meta_data'].get('start_frame', 0)
            end_frame = results['meta_data'].get('end_frame', keypoints3d.shape[0])
            keypoints3d = keypoints3d.reshape(keypoints3d.shape[0], -1, 3)
            if keypoints3d.shape[1] == 24:
                keypoints3d = np.concatenate(
                    (keypoints3d[:, :22, :], np.zeros((keypoints3d.shape[0], 30, 3))),
                    axis=1
                )
            elif keypoints3d.shape[1] == 22:
                keypoints3d = np.concatenate(
                    (keypoints3d, np.zeros((keypoints3d.shape[0], 30, 3))),
                    axis=1
                )
            keypoints3d = keypoints3d[start_frame: end_frame]
            assert not np.isnan(keypoints3d).any()
            results['keypoints3d'] = keypoints3d
            if motion_type == 'smpl_rot':
                results['rotation'] = motion_data['smpl_rot'][start_frame: end_frame]
                assert not np.isnan(results['rotation']).any()
                results['meta_data']['rotation_type'] = 'smpl_rot'
            elif motion_type == 'bvh_rot':
                results['rotation'] = motion_data['bvh_rot'][start_frame: end_frame]
                assert not np.isnan(results['rotation']).any()
                results['meta_data']['rotation_type'] = 'bvh_rot'
            else:
                results['meta_data']['rotation_type'] = 'h3d_rot'
            if 'expression' in motion_data:
                results['expression'] = motion_data['expression'][start_frame: end_frame]
                assert not np.isnan(results['expression']).any()
            if 'video_word_feat' in results:
                results['video_word_feat'] = results['video_word_feat'][start_frame: end_frame]
            else:
                results['video_word_feat'] = np.zeros((keypoints3d.shape[0], 1024))
        return results

    def __repr__(self):
        """Return a string representation of the class."""
        return f"{self.__class__.__name__}(motion_types={self.motion_types}, max_size={self.max_size})"


@PIPELINES.register_module()
class RandomCropKeypoints(object):
    r"""Random crop keypoints sequences.

    Args:
        min_size (int or None): The minimum size of the cropped motion
            sequence (inclusive).
        max_size (int or None): The maximum size of the cropped motion
            sequence (inclusive).
    """

    def __init__(self,
                 min_size: Optional[Union[int, None]] = None,
                 max_size: Optional[Union[int, None]] = None):
        self.min_size = min_size
        self.max_size = max_size
        assert self.min_size is not None
        assert self.max_size is not None

    def __call__(self, results):
        keypoints3d = results['keypoints3d']
        length = len(keypoints3d)
        crop_size = random.randint(self.min_size, self.max_size)
        if length > crop_size:
            idx = random.randint(0, length - crop_size)
            keypoints3d = keypoints3d[idx: idx + crop_size]
            if 'rotation' in results:
                results['rotation'] = results['rotation'][idx: idx + crop_size]
            if 'expression' in results:
                results['expression'] = results['expression'][idx: idx + crop_size]
            if 'video_word_feat' in results:
                results['video_word_feat'] = results['video_word_feat'][idx: idx + crop_size]
            results['keypoints3d'] = keypoints3d
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(min_size={self.min_size}'
        repr_str += f', max_size={self.max_size})'
        return repr_str


@PIPELINES.register_module()
class RetargetSkeleton(object):
    """Retarget motion data to a target skeleton.

    Adjusts motion data from a source skeleton to match a target skeleton structure by scaling and retargeting.

    Args:
        tgt_skel_file (str): Path to the file containing the target skeleton data.

    Note:
        This code is adapted from:
        https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb.
    """

    def __init__(self, tgt_skel_file: str):
        skeleton_data = np.load(tgt_skel_file)
        skeleton_data = skeleton_data.reshape(len(skeleton_data), -1, 3)
        skeleton_data = torch.from_numpy(skeleton_data)
        self.raw_offsets = torch.from_numpy(t2m_raw_offsets)
        tgt_skeleton = Skeleton(self.raw_offsets, t2m_body_hand_kinematic_chain, 'cpu')
        self.tgt_offsets = tgt_skeleton.get_offsets_joints(skeleton_data[0])
        self.tgt_skel_file = tgt_skel_file

    def __call__(self, results):
        """Retarget the motion data to the target skeleton.

        Args:
            results (dict): Contains 'keypoints3d' with source motion data.

        Returns:
            dict: Updated results with retargeted motion data in 'keypoints3d'.
        """
        positions = results['keypoints3d']
        src_skel = Skeleton(self.raw_offsets, t2m_body_hand_kinematic_chain, 'cpu')
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
        src_offset = src_offset.numpy()
        tgt_offset = self.tgt_offsets.numpy()

        # Calculate scale ratio based on leg lengths
        l_idx1, l_idx2 = 5, 8  # Indices for leg joints
        eps = 1e-5
        src_leg_len = np.linalg.norm(src_offset[l_idx1]) + np.linalg.norm(src_offset[l_idx2])
        tgt_leg_len = np.linalg.norm(tgt_offset[l_idx1]) + np.linalg.norm(tgt_offset[l_idx2])
        if src_leg_len < eps:
            a_idx1, a_idx2 = 19, 21
            src_arm_len = np.linalg.norm(src_offset[a_idx1]) + np.linalg.norm(src_offset[a_idx2])
            tgt_arm_len = np.linalg.norm(tgt_offset[a_idx1]) + np.linalg.norm(tgt_offset[a_idx2])
            if src_arm_len < eps:
                scale_rt = 1.0
            else:
                scale_rt = tgt_arm_len / src_arm_len
        else:
            scale_rt = tgt_leg_len / src_leg_len

        # Scale root positions
        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt

        # Perform inverse kinematics to get rotation parameters
        face_joint_idx = [2, 1, 17, 16]  # Indices for face-related joints
        quat_params = src_skel.inverse_kinematics_np(positions, face_joint_idx)
        # Set offsets to target skeleton and perform forward kinematics
        src_skel.set_offset(self.tgt_offsets)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        new_joints[np.isnan(new_joints)] = 0
        
        if not results['meta_data'].get('has_lhnd', False):
            new_joints[:, 22:, :] = 0
        results['keypoints3d'] = new_joints

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(tgt_skel_file='{self.tgt_skel_file}')"
    

@PIPELINES.register_module()
class MotionDownsample(object):

    def __init__(self, framerate_list: List[float]):
        self.framerate_list = framerate_list

    def __call__(self, results):
        framerate = np.random.choice(self.framerate_list)
        downsample_rate = int(results['meta_data']['framerate'] / framerate)
        results['meta_data']['framerate'] = framerate
        if 'keypoints3d' in results:
            results['keypoints3d'] = results['keypoints3d'][::downsample_rate]
        if 'rotation' in results:
            results['rotation'] = results['rotation'][::downsample_rate]
        if 'expression' in results:
            results['expression'] = results['expression'][::downsample_rate]       
        if 'video_word_feat' in results:
            results['video_word_feat'] = results['video_word_feat'] [::downsample_rate]
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    

@PIPELINES.register_module()
class PutOnFloor(object):
    """Shift motion data so that the skeleton stands on the floor.

    This pipeline component adjusts the motion data by translating it vertically,
    ensuring that the lowest point of the skeleton aligns with the floor level (y=0).

    Note:
        This code is adapted from:
        https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb.
    """

    def __init__(self):
        pass  # No initialization parameters required

    def __call__(self, results):
        """Adjust the motion data to place the skeleton on the floor.

        Args:
            results (dict): Contains 'keypoints3d' with motion data.

        Returns:
            dict: Updated results with adjusted 'keypoints3d'.
        """
        positions = results['keypoints3d']
        # Calculate the minimum y-coordinate among the first 22 joints over all frames
        floor_height = positions[:, :22, 1].min()
        # Shift the y-coordinates so that the lowest point is at y=0
        positions[:, :, 1] -= floor_height
        results['keypoints3d'] = positions
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    

@PIPELINES.register_module()
class MoveToOrigin(object):
    """Translate motion data so the root joint starts at the origin.

    This pipeline component adjusts the motion data by translating it so that
    the initial position of the root joint aligns with the origin.

    Note:
        This code is adapted from:
        https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb.
    """

    def __init__(self, origin: str):
        assert origin in ['xz', 'xyz']
        if origin == 'xz':
            self.weight = np.array([1, 0, 1])
        elif origin == 'xyz':
            self.weight = np.array([1, 1, 1])

    def __call__(self, results):
        """Adjust the motion data to move the root joint to the origin.

        Args:
            results (dict): Contains 'keypoints3d' with motion data.

        Returns:
            dict: Updated results with adjusted 'keypoints3d'.
        """
        positions = results['keypoints3d']
        # Get the initial root joint position (frame 0, joint 0)
        root_pos_init = positions[0, 0]

        root_pos_init = root_pos_init * self.weight
        positions = positions - root_pos_init
        results['keypoints3d'] = positions
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    
@PIPELINES.register_module()
class RotateToZ(object):
    """Rotate motion data so the initial facing direction aligns with the Z-axis.

    This pipeline component rotates the motion data such that the character's initial
    facing direction is aligned with the positive Z-axis, standardizing the orientation
    of the motion data.

    Note:
        This code is adapted from:
        https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb.
    """

    def __init__(self):
        pass  # No initialization parameters required

    def __call__(self, results):
        """Rotate the motion data to align the initial facing direction with the Z-axis.

        Args:
            results (dict): Contains 'keypoints3d' with motion data.

        Returns:
            dict: Updated results with rotated 'keypoints3d'.
        """
        positions = results['keypoints3d']
        # Indices for specific joints used to determine facing direction
        face_joint_idx = [2, 1, 17, 16]  # Right hip, left hip, right shoulder, left shoulder
        r_hip, l_hip, sdr_r, sdr_l = face_joint_idx

        # Calculate the initial across vector from hips and shoulders
        pos_init = positions[0]
        across1 = pos_init[r_hip] - pos_init[l_hip]
        across2 = pos_init[sdr_r] - pos_init[sdr_l]
        across = across1 + across2
        eps = 1e-8
        across = across / (np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis] + eps)

        # Calculate the initial forward vector using a cross product with the up vector
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / (np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis] + eps)

        # Compute the rotation quaternion between the initial forward vector and target vector (Z-axis)
        target_vector = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target_vector)

        # Apply the rotation to all joints across all frames
        root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
        positions = qrot_np(root_quat_init, positions)
        positions[np.isnan(positions)] = 0
        results['keypoints3d'] = positions
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@PIPELINES.register_module()
class KeypointsToTomato(object):
    """Convert keypoint motion data to the TOMATO representation.

    This pipeline component transforms 3D keypoints into the TOMATO motion representation,
    suitable for motion generation models.

    Note:
        Adapted from:
        https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb.
    """

    def __init__(self, smooth_forward=True):
        self.raw_offsets = torch.from_numpy(t2m_raw_offsets)
        self.smooth_forward = smooth_forward
                    
    def get_cont6d_params(self, positions):
        """Compute continuous 6D rotation parameters and root velocities."""
        skel = Skeleton(self.raw_offsets, t2m_body_hand_kinematic_chain, "cpu")
        face_joint_idx = [2, 1, 17, 16]

        quat_params = skel.inverse_kinematics_np(positions, face_joint_idx, smooth_forward=self.smooth_forward)
        quat_params[np.isnan(quat_params)] = 0
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()

        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)

        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

        return cont_6d_params, r_velocity, velocity, r_rot
        
    def get_rifke(self, r_rot, positions):
        """Compute rotation-invariant joint positions."""
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions
            
    def __call__(self, results):
        """Convert keypoints to TOMATO motion representation."""
        positions = results['keypoints3d']
        global_positions = positions.copy()

        cont_6d_params, r_velocity, velocity, r_rot = self.get_cont6d_params(positions)
        positions = self.get_rifke(r_rot, positions)

        root_y = positions[:-1, 0, 1:2]
        r_velocity = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]

        root_data = np.concatenate([r_velocity, l_velocity, root_y], axis=-1)

        rot_data = cont_6d_params[:-1, 1:].reshape(len(cont_6d_params) - 1, -1)
        motion_type = results['motion_type']
        if motion_type == 'smpl_rot':
            rot_data = results['rotation'][:-1]
            if rot_data.shape[1] == 72:
                num_frames = rot_data.shape[0]
                rot_data = rot_data.reshape((num_frames, 24, 3))
                rot_data = np.concatenate((
                    rot_data[:, 1: 22, :],
                    np.zeros((num_frames, 30, 3))
                ), axis=1)
                rot_data = torch.from_numpy(rot_data)
                rot_data = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(rot_data))
                rot_data = rot_data.numpy().reshape((num_frames, -1))
            elif rot_data.shape[1] == 156:
                num_frames = rot_data.shape[0]
                rot_data = rot_data.reshape((num_frames, 52, 3))[:, 1:, :]
                rot_data = torch.from_numpy(rot_data)
                rot_data = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(rot_data))
                rot_data = rot_data.numpy().reshape((num_frames, -1))
            else:
                raise NotImplementedError()
        ric_data = positions[:-1, 1:].reshape(len(positions) - 1, -1)

        local_vel = qrot_np(
            np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
            global_positions[1:] - global_positions[:-1]
        )
        local_vel = local_vel.reshape(len(local_vel), -1)
        if not results['meta_data']['has_lhnd']:
            ric_data[:, 21 * 3: 51 * 3] = 0
            rot_data[:, 21 * 6: 51 * 6] = 0
            local_vel[:, 22 * 3: 52 * 3] = 0
        data = np.concatenate([root_data, ric_data, rot_data, local_vel], axis=-1)

        if 'expression' in results:
            data = np.concatenate([data, results['expression'][:-1]], axis=-1)
        else:
            data = np.concatenate([data, np.zeros((data.shape[0], 50))], axis=-1)
            
        if 'video_word_feat' in results:
            results['video_word_feat'] = results['video_word_feat'][:-1]

        data[np.isnan(data)] = 0
        results['motion'] = data
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@PIPELINES.register_module()
class MaskedRandomCrop(object):
    r"""Masked Random crop motion sequences. Each sequence will be padded with zeros
    to the maximum length.

    Args:
        min_size (int or None): The minimum size of the cropped motion
            sequence (inclusive).
        max_size (int or None): The maximum size of the cropped motion
            sequence (inclusive).
    """

    def __init__(self,
                 min_size: Optional[Union[int, None]] = None,
                 max_size: Optional[Union[int, None]] = None,
                 pad_size: Optional[Union[int, None]] = None):
        self.min_size = min_size
        self.max_size = max_size
        assert self.min_size is not None
        assert self.max_size is not None
        self.pad_size = max_size if pad_size is None else pad_size

    def __call__(self, results):
        motion = results['motion']
        length = len(motion)
        crop_size = random.randint(self.min_size, self.max_size)
        if length > crop_size:
            idx = random.randint(0, length - crop_size)
            motion = motion[idx: idx + crop_size]
            results['motion_length'] = crop_size
            if 'video_word_feat' in results:
                results['video_word_feat'] = results['video_word_feat'][idx: idx + crop_size]
        else:
            results['motion_length'] = length
        padding_length = self.pad_size - min(crop_size, length)
        if padding_length > 0:
            D = motion.shape[1:]
            padding_zeros = np.zeros((padding_length, *D), dtype=np.float32)
            motion = np.concatenate([motion, padding_zeros], axis=0)
            if 'video_word_feat' in results:
                D = results['video_word_feat'].shape[1]
                results['video_word_feat'] = np.concatenate(
                    [results['video_word_feat'], np.zeros((padding_length, D))],
                    axis=0
                )
        results['motion'] = motion
        results['motion_shape'] = motion.shape
        if length >= self.pad_size and crop_size == self.pad_size:
            motion_mask = torch.ones(self.pad_size).numpy()
        else:
            motion_mask = torch.cat(
                (torch.ones(min(length, crop_size)),
                 torch.zeros(self.pad_size - min(length, crop_size))),
                dim=0).numpy()
        motion_mask = np.expand_dims(motion_mask, axis=1)
        motion_mask = np.repeat(motion_mask, 10, axis=1)
        meta_data = results['meta_data']
        if not meta_data['has_root']:
            motion_mask[:, 0] = 0
        if not meta_data['has_head']:
            motion_mask[:, 1] = 0
        if not meta_data['has_stem']:
            motion_mask[:, 2] = 0
        if not meta_data['has_larm']:
            motion_mask[:, 3] = 0
        if not meta_data['has_rarm']:
            motion_mask[:, 4] = 0
        if not meta_data['has_lleg']:
            motion_mask[:, 5] = 0
        if not meta_data['has_rleg']:
            motion_mask[:, 6] = 0
        if not meta_data['has_lhnd']:
            motion_mask[:, 7] = 0
        if not meta_data['has_rhnd']:
            motion_mask[:, 8] = 0
        if not meta_data['has_face']:
            motion_mask[:, 9] = 0
        results['motion_mask'] = motion_mask
        assert len(motion) == self.pad_size
        
        if 'video_word_feat' in results:
            assert len(results['video_word_feat']) == self.pad_size
        else:
            results['video_word_feat'] = np.zeros((self.pad_size, 1024))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(min_size={self.min_size}'
        repr_str += f', max_size={self.max_size})'
        return repr_str


@PIPELINES.register_module()
class MaskedCrop(object):
    r"""Masked crop motion sequences. Each sequence will be padded with zeros
    to the maximum length.

    Args:
        min_size (int or None): The minimum size of the cropped motion
            sequence (inclusive).
        max_size (int or None): The maximum size of the cropped motion
            sequence (inclusive).
    """

    def __init__(self,
                 crop_size: Optional[Union[int, None]] = None,
                 pad_size: Optional[Union[int, None]] = None):
        self.crop_size = crop_size
        assert self.crop_size is not None
        if pad_size is None:
            self.pad_size = self.crop_size
        else:
            self.pad_size = pad_size

    def __call__(self, results):
        motion = results['motion']
        length = len(motion)
        crop_size = self.crop_size
        pad_size = self.pad_size
        if length > crop_size:
            idx = random.randint(0, length - crop_size)
            motion = motion[idx: idx + crop_size]
            results['motion_length'] = crop_size
            if 'video_word_feat' in results:
                results['video_word_feat'] = results['video_word_feat'][idx: idx + crop_size]
        else:
            results['motion_length'] = length
        actual_length = min(crop_size, length)
        padding_length = pad_size - actual_length
        if padding_length > 0:
            D = motion.shape[1:]
            padding_zeros = np.zeros((padding_length, *D), dtype=np.float32)
            motion = np.concatenate([motion, padding_zeros], axis=0)
            if 'video_word_feat' in results:
                D = results['video_word_feat'].shape[1]
                results['video_word_feat'] = np.concatenate(
                    [results['video_word_feat'], np.zeros((padding_length, D))],
                    axis=0
                )
        results['motion'] = motion
        results['motion_shape'] = motion.shape
        motion_mask = torch.cat(
            (torch.ones(actual_length),
                torch.zeros(padding_length)),
            dim=0).numpy()
        motion_mask = np.expand_dims(motion_mask, axis=1)
        motion_mask = np.repeat(motion_mask, 10, axis=1)
        meta_data = results['meta_data']
        if not meta_data['has_root']:
            motion_mask[:, 0] = 0
        if not meta_data['has_head']:
            motion_mask[:, 1] = 0
        if not meta_data['has_stem']:
            motion_mask[:, 2] = 0
        if not meta_data['has_larm']:
            motion_mask[:, 3] = 0
        if not meta_data['has_rarm']:
            motion_mask[:, 4] = 0
        if not meta_data['has_lleg']:
            motion_mask[:, 5] = 0
        if not meta_data['has_rleg']:
            motion_mask[:, 6] = 0
        if not meta_data['has_lhnd']:
            motion_mask[:, 7] = 0
        if not meta_data['has_rhnd']:
            motion_mask[:, 8] = 0
        if not meta_data['has_face']:
            motion_mask[:, 9] = 0
        results['motion_mask'] = motion_mask
        assert len(motion) == pad_size
        if 'video_word_feat' in results:
            assert len(results['video_word_feat']) == pad_size
        else:
            results['video_word_feat'] = np.zeros((pad_size, 1024))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        return repr_str
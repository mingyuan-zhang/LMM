import copy
import os
import os.path
from typing import Optional, Union, List, Dict

import numpy as np
import torch
import json

from .base_dataset import BaseMotionDataset
from .builder import DATASETS


@DATASETS.register_module()
class TextMotionDataset(BaseMotionDataset):
    """
    TextMotion dataset for handling motion data paired with text descriptions.

    Args:
        data_prefix (str): Path to the base directory containing the dataset.
        pipeline (list): List of data transformations to apply.
        dataset_name (Optional[str]): Name of the dataset.
        fixed_length (Optional[int]): Fixed length of data samples (if applicable).
        ann_file (Optional[str]): Path to the annotation file.
        motion_dir (Optional[str]): Path to the directory containing motion data.
        text_dir (Optional[str]): Path to the directory containing text data.
        token_dir (Optional[str]): Path to the directory containing token data.
        clip_feat_dir (Optional[str]): Path to the directory containing clip feature data.
        meta_dir (Optional[str]): Path to the directory containing metadata.
        eval_cfg (Optional[dict]): Configuration for evaluation metrics.
        test_mode (Optional[bool]): Whether the dataset is in test mode. Defaults to False.
        siamese_mode (Optional[bool]): Whether to use Siamese mode (motion1 vs. motion2 comparison). Defaults to False.
        tcomb_mode (Optional[bool]): Mode for specific processing (tcomb). Defaults to False.
        fine_mode (Optional[bool]): Whether to use fine-grained text processing. Defaults to False.
        balanced_sampling (Optional[int]): Number of categories for balanced sampling. If not None, enables balanced sampling.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: List[Dict],
                 dataset_name: Optional[Union[str, None]] = None,
                 fixed_length: Optional[Union[int, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 motion_dir: Optional[Union[str, None]] = None,
                 text_dir: Optional[Union[str, None]] = None,
                 token_dir: Optional[Union[str, None]] = None,
                 clip_feat_dir: Optional[Union[str, None]] = None,
                 meta_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False,
                 siamese_mode: Optional[bool] = False,
                 tcomb_mode: Optional[bool] = False,
                 fine_mode: Optional[bool] = False,
                 balanced_sampling: Optional[Union[int, None]] = None):
        self.text_dir = os.path.join(data_prefix, 'datasets', dataset_name, text_dir)
        self.token_dir = os.path.join(data_prefix, 'datasets', dataset_name, token_dir) if token_dir else None
        self.clip_feat_dir = os.path.join(data_prefix, 'datasets', dataset_name, clip_feat_dir) if clip_feat_dir else None
        self.meta_dir = os.path.join(data_prefix, 'datasets', dataset_name, meta_dir) if meta_dir else None
        self.siamese_mode = siamese_mode
        self.tcomb_mode = tcomb_mode
        self.fine_mode = fine_mode
        self.balanced_sampling = balanced_sampling is not None

        if self.balanced_sampling:
            self.category_list = [[] for _ in range(balanced_sampling)]

        super(TextMotionDataset, self).__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            dataset_name=dataset_name,
            fixed_length=fixed_length,
            ann_file=ann_file,
            motion_dir=motion_dir,
            eval_cfg=eval_cfg,
            test_mode=test_mode
        )

    def load_anno(self, idx: int, name: str) -> Dict:
        """
        Load a single annotation based on the given index and name.

        Args:
            idx (int): Index of the data sample.
            name (str): Name of the data sample (typically used as a file identifier).

        Returns:
            dict: A dictionary containing the loaded data and relevant information.
        """
        results = {}
        if self.siamese_mode:
            motion_path = os.path.join(self.motion_dir, name + '.npz')
            motion_data = np.load(motion_path)
            results['motion1'] = motion_data['motion1']
            results['motion2'] = motion_data['motion2']
            assert results['motion1'].shape == results['motion2'].shape
        else:
            motion_path = os.path.join(self.motion_dir, name + '.npy')
            motion_data = np.load(motion_path)
            results['motion'] = motion_data

        if self.fine_mode:
            text_path = os.path.join(self.text_dir, name + '.json')
            text_data = json.load(open(text_path))
            for entry in text_data:
                entry.pop('start_frame', None)
                entry.pop('end_frame', None)
                entry.pop('num_frames', None)
            results['text'] = text_data
        else:
            text_path = os.path.join(self.text_dir, name + '.txt')
            results['text'] = [line.strip() for line in open(text_path, 'r')]

        if self.token_dir:
            token_path = os.path.join(self.token_dir, name + '.txt')
            results['token'] = [line.strip() for line in open(token_path, 'r')]

        if self.clip_feat_dir:
            clip_feat_path = os.path.join(self.clip_feat_dir, name + '.npy')
            results['clip_feat_path'] = clip_feat_path
            # if self.fine_mode:
            #     results['clip_feat_path'] = clip_feat_path
            # else:
            #     clip_feat = torch.from_numpy(np.load(clip_feat_path))
            #     if len(clip_feat.shape) == 2:
            #         clip_feat = clip_feat.unsqueeze(0)
            #     results['clip_feat'] = clip_feat

        if self.meta_dir:
            score_path = os.path.join(self.meta_dir, name + '_score.npy')
            results['score'] = torch.from_numpy(np.load(score_path))

        if self.balanced_sampling:
            assert self.meta_dir is not None
            category_path = os.path.join(self.meta_dir, name + '.json')
            category = json.load(open(category_path))['category']
            self.category_list[category].append(idx)

        return results

    def prepare_data(self, idx: int) -> Dict:
        """
        Prepare raw data for the given index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: Processed data after applying the pipeline.
        """
        results = copy.deepcopy(self.data_infos[idx])
        text_list = results['text']
        selected_idx = np.random.randint(0, len(text_list))
        results['text'] = text_list[selected_idx]

        if 'clip_feat' in results:
            results['clip_feat'] = results['clip_feat'][selected_idx]

        if 'clip_feat_path' in results:
            clip_feat = torch.from_numpy(np.load(results['clip_feat_path']))
            if len(clip_feat.shape) == 2:
                clip_feat = clip_feat.unsqueeze(0)
            results['clip_feat'] = clip_feat[selected_idx]

        if 'token' in results:
            results['token'] = results['token'][selected_idx]

        results['dataset_name'] = self.dataset_name
        results = self.pipeline(results)
        return results

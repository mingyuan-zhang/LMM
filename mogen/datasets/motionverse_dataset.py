import copy
import os
import pickle as pkl
from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
from .builder import DATASETS
from .pipelines import Compose, RetargetSkeleton
import random
import pytorch3d.transforms as geometry
from scipy.ndimage import gaussian_filter
from mogen.core.evaluation import build_evaluator
from mogen.core.evaluation.utils import compute_similarity_transform, transform_pose_sequence
from mogen.models.builder import build_submodule
from .utils import copy_repr_data, extract_repr_data, move_repr_data, recover_from_ric

class SingleMotionVerseDataset(Dataset):
    """
    A dataset class for handling single MotionVerse datasets.

    Args:
        dataset_name (str): Name of the dataset and task to load.
        data_prefix (str): Path to the directory containing the dataset.
        ann_file (str): Path to the annotation file.
        pipeline (list): A list of transformations to apply on the data.
        mode (str): the mode of current work. Choices: ['pretrain', 'train', 'test'].
        eval_cfg (dict): Configuration for evaluation metrics.
    """

    def __init__(self,
                 dataset_path: Optional[str] = None,
                 task_name: Optional[str] = None,
                 data_prefix: Optional[str] = None,
                 ann_file: Optional[str] = None,
                 pipeline: Optional[List[dict]] = None,
                 
                 # for text2motion and speech2gesture
                 tgt_min_motion_length: int = 20,
                 tgt_max_motion_length: int = 200,
                 
                 # for video2motion
                 v2m_window_size: int = 20,
                 
                 # for motion prediction
                 mp_input_length: int = 50,
                 mp_output_length: int = 25,
                 mp_stride_step: int = 5,
                 
                 # for general test
                 test_rotation_type: str = 'h3d_rot',
                 target_framerate: float = 20,
                 eval_cfg: Optional[dict] = None,
                 test_mode: Optional[bool] = False):
        data_prefix = os.path.join(data_prefix, 'datasets', dataset_path)
        self.dataset_path = dataset_path
        assert task_name in ['mocap', 't2m', 'v2m', 's2g', 'm2d']
        self.task_name = task_name
        self.dataset_name = dataset_path + '_' + task_name

        # define subdirectories
        self.meta_dir = os.path.join(data_prefix, 'metas')
        self.motion_dir = os.path.join(data_prefix, 'motions')
        self.eval_motion_dir = os.path.join(data_prefix, 'eval_motions')
        self.text_dir = os.path.join(data_prefix, 'texts')
        self.text_feat_dir = os.path.join(data_prefix, 'text_feats')
        self.speech_dir = os.path.join(data_prefix, 'speeches')
        self.speech_feat_dir = os.path.join(data_prefix, 'speech_feats')
        self.music_dir = os.path.join(data_prefix, 'musics')
        self.music_feat_dir = os.path.join(data_prefix, 'music_feats')
        self.video_feat_dir = os.path.join(data_prefix, 'video_feats')
        self.anno_file = os.path.join(data_prefix, 'splits', ann_file)

        self.pipeline = Compose(pipeline)

        self.tgt_min_motion_length = tgt_min_motion_length
        self.tgt_max_motion_length = tgt_max_motion_length
        
        self.v2m_window_size = v2m_window_size
        
        self.mp_input_length = mp_input_length
        self.mp_output_length = mp_output_length
        self.mp_stride_step = mp_stride_step
        
        self.target_framerate = target_framerate
        self.test_rotation_type = test_rotation_type
        self.test_mode = test_mode
        self.load_annotations()
        self.eval_cfg = copy.deepcopy(eval_cfg)
        if self.test_mode:
            self.prepare_evaluation()

    def __len__(self) -> int:
        """Return the length of the current dataset."""
        if self.test_mode:
            return len(self.eval_indexes)
        return len(self.name_list)

    def __getitem__(self, idx: int) -> dict:
        """Prepare data for the given index."""
        if self.test_mode:
            idx = self.eval_indexes[idx]
        return self.prepare_data(idx)
    
    def load_annotations(self):
        if self.task_name == 'mocap':
            self.load_annotations_mocap()
        elif self.task_name == 't2m':
            self.load_annotations_t2m()
        elif self.task_name == 'v2m':
            self.load_annotations_v2m()
        elif self.task_name == 's2g':
            self.load_annotations_s2g()
        elif self.task_name == 'm2d':
            self.load_annotations_m2d()
        else:
            raise NotImplementedError()
    
    def load_annotations_mocap(self):
        if self.test_mode:
            self.name_list = []
            self.src_start_frame = []
            self.src_end_frame = []
            self.tgt_start_frame = []
            self.tgt_end_frame = []
            tgt_motion_length = self.mp_input_length + self.mp_output_length
            for name in open(self.anno_file):
                name = name.strip()
                meta_path = os.path.join(self.meta_dir, name + ".json")
                meta_data = json.load(open(meta_path))
                num_frames = meta_data['num_frames']
                downrate = int(meta_data['framerate'] / self.target_framerate + 0.1)
                if num_frames < (self.mp_input_length + self.mp_output_length) * downrate:
                    continue
                lim = num_frames // downrate - tgt_motion_length
                for start_frame in range(0, lim, self.mp_stride_step):
                    self.name_list.append(name)
                    self.src_start_frame.append((start_frame + 1) * downrate)
                    self.src_end_frame.append((start_frame + tgt_motion_length + 1) * downrate)
                    self.tgt_start_frame.append(start_frame + self.mp_input_length)
                    self.tgt_end_frame.append(start_frame + tgt_motion_length)
        else:
            self.name_list = []
            for name in open(self.anno_file):
                name = name.strip()
                self.name_list.append(name)
    
    def load_annotations_t2m(self):
        self.name_list = []
        self.text_idx = []
        for name in open(self.anno_file):
            name = name.strip()
            meta_path = os.path.join(self.meta_dir, name + ".json")
            meta_data = json.load(open(meta_path))
            downrate = int(meta_data['framerate'] / self.target_framerate + 0.1)
            text_path = os.path.join(self.text_dir, name + ".json")
            text_data = json.load(open(text_path))
            for i, anno in enumerate(text_data):
                start_frame = anno['start_frame'] // downrate
                end_frame = min(anno['end_frame'], meta_data['num_frames']) // downrate
                num_frame = end_frame - start_frame
                if num_frame < self.tgt_min_motion_length or num_frame > self.tgt_max_motion_length:
                    continue
                if len(anno['body_text']) > 0:
                    self.name_list.append(name)
                    self.text_idx.append(i)
    
    def load_annotations_v2m(self):
        if not self.test_mode:
            self.name_list = []
            for name in open(self.anno_file):
                name = name.strip()
                self.name_list.append(name)
        else:
            self.name_list = []
            self.start_frame = []
            self.end_frame = []
            self.valid_start_frame = []
            self.valid_end_frame = []
            for name in open(self.anno_file):
                name = name.strip()
                meta_path = os.path.join(self.meta_dir, name + ".json")
                meta_data = json.load(open(meta_path))
                num_frames = meta_data['num_frames']
                assert num_frames >= self.v2m_window_size
                cur_idx = 0
                while cur_idx < num_frames:
                    if cur_idx + self.v2m_window_size < num_frames:
                        self.name_list.append(name)
                        self.start_frame.append(cur_idx)
                        self.end_frame.append(cur_idx + self.v2m_window_size)
                        self.valid_start_frame.append(cur_idx)
                        self.valid_end_frame.append(cur_idx + self.v2m_window_size)
                        cur_idx += self.v2m_window_size
                    else:
                        self.name_list.append(name)
                        self.start_frame.append(num_frames - self.v2m_window_size)
                        self.end_frame.append(num_frames)
                        self.valid_start_frame.append(cur_idx)
                        self.valid_end_frame.append(num_frames)
                        break
    
    def load_annotations_s2g(self):
        self.name_list = []
        self.speech_idx = []
        for name in open(self.anno_file):
            name = name.strip()
            meta_path = os.path.join(self.meta_dir, name + ".json")
            meta_data = json.load(open(meta_path))
            downrate = int(meta_data['framerate'] / self.target_framerate + 0.1)
            speech_path = os.path.join(self.speech_dir, name + ".json")
            speech_data = json.load(open(speech_path))
            for i, anno in enumerate(speech_data):
                start_frame = anno['start_frame'] // downrate
                end_frame = min(anno['end_frame'], meta_data['num_frames']) // downrate
                num_frame = end_frame - start_frame
                if num_frame < self.tgt_min_motion_length or num_frame > self.tgt_max_motion_length:
                    continue
                self.name_list.append(name)
                self.speech_idx.append(i)
    
    def load_annotations_m2d(self):
        self.name_list = []
        self.music_idx = []
        for name in open(self.anno_file):
            name = name.strip()
            meta_path = os.path.join(self.meta_dir, name + ".json")
            meta_data = json.load(open(meta_path))
            downrate = int(meta_data['framerate'] / self.target_framerate + 0.1)
            music_path = os.path.join(self.music_dir, name + ".json")
            music_data = json.load(open(music_path))
            for i, anno in enumerate(music_data):
                start_frame = anno['start_frame'] // downrate
                end_frame = min(anno['end_frame'], meta_data['num_frames']) // downrate
                num_frame = end_frame - start_frame
                if num_frame < self.tgt_min_motion_length or num_frame > self.tgt_max_motion_length:
                    continue
                self.name_list.append(name)
                self.music_idx.append(i)

    def prepare_data_base(self, idx: int) -> dict:
        results = {}
        name = self.name_list[idx]
        results['motion_path'] = os.path.join(self.motion_dir, name + ".npz")
        meta_path = os.path.join(self.meta_dir, name + ".json")
        meta_data = json.load(open(meta_path))
        meta_data['dataset_name'] = self.dataset_name
        results['meta_data'] = meta_data
        results['meta_data']['sample_idx'] = idx
        results.update({
            'text_word_feat': np.zeros((77, 1024)).astype(np.float32),
            'text_seq_feat': np.zeros((1024)).astype(np.float32),
            'text_cond': 0,
            'music_word_feat': np.zeros((229, 768)).astype(np.float32),
            'music_seq_feat': np.zeros((1024)).astype(np.float32),
            'music_cond': 0,
            'speech_word_feat': np.zeros((229, 768)).astype(np.float32),
            'speech_seq_feat': np.zeros((1024)).astype(np.float32),
            'speech_cond': 0,
            'video_seq_feat': np.zeros((1024)).astype(np.float32),
            'video_cond': 0,
        })
        return results
    
    def prepare_data(self, idx: int) -> dict:
        if self.task_name == 'mocap':
            results = self.prepare_data_mocap(idx)
        elif self.task_name == 't2m':
             results = self.prepare_data_t2m(idx)
        elif self.task_name == 'v2m':
             results = self.prepare_data_v2m(idx)
        elif self.task_name == 's2g':
             results = self.prepare_data_s2g(idx)
        elif self.task_name == 'm2d':
             results = self.prepare_data_m2d(idx)
        else:
            raise NotImplementedError()
        results = self.pipeline(results)
        return results
        
    def prepare_data_mocap(self, idx: int) -> dict:
        results = self.prepare_data_base(idx)
        if self.test_mode:
            results['meta_data']['start_frame'] = self.src_start_frame[idx]
            results['meta_data']['end_frame'] = self.src_end_frame[idx]
            results['context_mask'] = np.concatenate(
                (np.ones((self.mp_input_length - 1)), np.zeros((self.mp_output_length))),
                axis=-1
            )
        return results
    
    def prepare_data_t2m(self, idx: int) -> dict:
        results = self.prepare_data_base(idx)
        name = self.name_list[idx]
        text_idx = self.text_idx[idx]
        text_path = os.path.join(self.text_dir, name + ".json")
        text_data = json.load(open(text_path))[text_idx]
        text_feat_path = os.path.join(self.text_feat_dir, name + ".pkl")
        text_feat_data = pkl.load(open(text_feat_path, "rb"))
        text_list = text_data['body_text']
        tid = np.random.randint(len(text_list))
        text = text_list[tid]
        text_word_feat = text_feat_data['text_word_feats'][text_idx][tid]
        text_seq_feat = text_feat_data['text_seq_feats'][text_idx][tid]
        assert text_word_feat.shape[0] == 77
        assert text_word_feat.shape[1] == 1024
        assert text_seq_feat.shape[0] == 1024

        if self.test_mode:
            motion_path = os.path.join(self.eval_motion_dir, name + ".npy")
            motion_data = np.load(motion_path)
            assert not np.isnan(motion_data).any()
            downrate = int(results['meta_data']['framerate'] / self.target_framerate + 0.1)
            start_frame = text_data['start_frame'] // downrate
            end_frame = text_data['end_frame'] // downrate
            motion_data = motion_data[start_frame: end_frame]
            results['meta_data']['framerate'] = self.target_framerate
            results['meta_data']['rotation_type'] = self.test_rotation_type
            assert motion_data.shape[0] > 0
            if 'body_tokens' in text_data:
                token = text_data['body_tokens'][tid]
            else:
                token = ""
            text_cond = 1
            results.update({
                'motion': motion_data,
                'text_word_feat': text_word_feat,
                'text_seq_feat': text_seq_feat,
                'text_cond': text_cond,
                'text': text,
                'token': token
            })
        else:
            results['meta_data']['start_frame'] = text_data['start_frame']
            results['meta_data']['end_frame'] = text_data['end_frame']
            text_cond = 1
            results.update({
                'text_word_feat': text_word_feat,
                'text_seq_feat': text_seq_feat,
                'text_cond': text_cond
            })
        return results
            
    def prepare_data_v2m(self, idx: int) -> dict:
        results = self.prepare_data_base(idx)
        name = self.name_list[idx]
        video_feat_path = os.path.join(self.video_feat_dir, name + ".pkl")
        video_feat_data = pkl.load(open(video_feat_path, "rb"))
        video_word_feat = video_feat_data['video_word_feats']
        video_seq_feat = video_feat_data['video_seq_feats']
        assert video_word_feat.shape[0] == results['meta_data']['num_frames']
        assert video_word_feat.shape[1] == 1024
        assert video_seq_feat.shape[0] == 1024
        video_cond = 1
        if self.test_mode:
            results['meta_data']['start_frame'] = self.start_frame[idx]
            results['meta_data']['end_frame'] = self.end_frame[idx]
            motion_path = os.path.join(self.eval_motion_dir, name + ".npy")
            motion_data = np.load(motion_path)
            assert not np.isnan(motion_data).any()
            
            start_frame = self.start_frame[idx]
            end_frame = self.end_frame[idx]
            motion_data = motion_data[start_frame: end_frame]
            video_word_feat = video_word_feat[start_frame: end_frame]
            results['meta_data']['framerate'] = self.target_framerate
            results['meta_data']['rotation_type'] = self.test_rotation_type
            assert motion_data.shape[0] > 0
            results.update({
                'motion': motion_data,
                'video_word_feat': video_word_feat,
                'video_seq_feat': video_seq_feat,
                'video_cond': video_cond
            })
        else:
            results.update({
                'video_word_feat': video_word_feat,
                'video_seq_feat': video_seq_feat,
                'video_cond': video_cond
            })
        return results
    
    def prepare_data_s2g(self, idx: int) -> dict:
        results = self.prepare_data_base(idx)
        name = self.name_list[idx]
        speech_idx = self.speech_idx[idx]
        speech_path = os.path.join(self.speech_dir, name + ".json")
        speech_data = json.load(open(speech_path))[speech_idx]
        speech_feat_path = os.path.join(self.speech_feat_dir, name + ".pkl")
        speech_feat_data = pkl.load(open(speech_feat_path, "rb"))
        try:
            speech_word_feat = speech_feat_data['audio_word_feats'][speech_idx]
            speech_seq_feat = speech_feat_data['audio_seq_feats'][speech_idx]
        except:
            speech_word_feat = speech_feat_data['speech_word_feats'][speech_idx]
            speech_seq_feat = speech_feat_data['speech_seq_feats'][speech_idx]
        del speech_feat_data
        assert speech_word_feat.shape[0] == 229
        assert speech_word_feat.shape[1] == 768
        assert speech_seq_feat.shape[0] == 1024
        
        results['meta_data']['start_frame'] = speech_data['start_frame']
        results['meta_data']['end_frame'] = speech_data['end_frame']
        speech_cond = 1
        results.update({
            'speech_word_feat': speech_word_feat,
            'speech_seq_feat': speech_seq_feat,
            'speech_cond': speech_cond
        })
        if self.test_mode:
            results['meta_data']['framerate'] = self.target_framerate
            results['meta_data']['rotation_type'] = self.test_rotation_type
            eval_data_path = os.path.join(self.eval_motion_dir, name + ".npz")
            eval_data = np.load(eval_data_path)
            motion_data = eval_data["bvh_rot_beat141"]
            sem_data = eval_data["sem"]
            wav_data = eval_data["wave16k"]
            assert not np.isnan(motion_data).any()
            
            start_frame = results['meta_data']['start_frame']
            end_frame = results['meta_data']['end_frame']
            wav_start_frame = start_frame / results['meta_data']['framerate'] * 16000
            wav_end_frame = end_frame / results['meta_data']['framerate'] * 16000
            motion_data = motion_data[start_frame: end_frame]
            sem_data = sem_data[start_frame: end_frame]
            wav_data = wav_data[wav_start_frame: wav_end_frame]
            assert motion_data.shape[0] > 0
            results.update({
                'motion': motion_data,
                'sem_score': sem_data,
                'wav_feat': wav_data
            })
        return results
    
    def prepare_data_m2d(self, idx: int) -> dict:
        results = self.prepare_data_base(idx)
        name = self.name_list[idx]
        music_idx = self.music_idx[idx]
        music_path = os.path.join(self.music_dir, name + ".json")
        music_data = json.load(open(music_path))[music_idx]
        music_feat_path = os.path.join(self.music_feat_dir, name + ".pkl")
        music_feat_data = pkl.load(open(music_feat_path, "rb"))
        music_word_feat = music_feat_data['audio_word_feats'][music_idx]
        music_seq_feat = music_feat_data['audio_seq_feats'][music_idx]
        assert music_word_feat.shape[0] == 229
        assert music_word_feat.shape[1] == 768
        assert music_seq_feat.shape[0] == 1024

        results['meta_data']['start_frame'] = music_data['start_frame']
        results['meta_data']['end_frame'] = music_data['end_frame']
        music_cond = 1
        results.update({
            'music_word_feat': music_word_feat,
            'music_seq_feat': music_seq_feat,
            'music_cond': music_cond
        })
        return results

    def prepare_evaluation(self):
        """
        Prepare the dataset for evaluation by initializing evaluators and creating evaluation indexes.
        """
        self.evaluators = []
        self.eval_indexes = []
        self.evaluator_model = build_submodule(self.eval_cfg.get('evaluator_model', None))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.evaluator_model is not None:
            self.evaluator_model = self.evaluator_model.to(device)
            self.evaluator_model.eval()
        self.eval_cfg['evaluator_model'] = self.evaluator_model

        for _ in range(self.eval_cfg['replication_times']):
            eval_indexes = np.arange(len(self.name_list))
            if self.eval_cfg.get('shuffle_indexes', False):
                np.random.shuffle(eval_indexes)
            self.eval_indexes.append(eval_indexes)

        for metric in self.eval_cfg['metrics']:
            evaluator, self.eval_indexes = build_evaluator(
                metric, self.eval_cfg, len(self.name_list), self.eval_indexes)
            self.evaluators.append(evaluator)

        self.eval_indexes = np.concatenate(self.eval_indexes)
        
    def process_outputs(self, results):
        return results

    def evaluate(self, results: List[dict], work_dir: str, logger=None) -> dict:
        """
        Evaluate the model performance based on the results.

        Args:
            results (list): A list of result dictionaries.
            work_dir (str): Directory where evaluation logs will be stored.
            logger: Logger object to record evaluation results (optional).

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        metrics = {}
        results = self.process_outputs(results)
        for evaluator in self.evaluators:
            metrics.update(evaluator.evaluate(results))
        if logger is not None:
            logger.info(metrics)
        eval_output = os.path.join(work_dir, 'eval_results.log')
        with open(eval_output, 'w') as f:
            for k, v in metrics.items():
                f.write(k + ': ' + str(v) + '\n')
        return metrics
    

def create_single_dataset(cfg: dict):
    dataset_path = cfg['dataset_path']
    if dataset_path == 'amass':
        return MotionVerseAMASS(**cfg)
    elif dataset_path == 'humanml3d':
        return MotionVerseH3D(**cfg)
    elif dataset_path == 'kitml':
        return MotionVerseKIT(**cfg)
    elif dataset_path == 'babel':
        return MotionVerseBABEL(**cfg)
    elif dataset_path == 'motionx':
        return MotionVerseMotionX(**cfg)
    elif dataset_path == 'humanact12':
        return MotionVerseACT12(**cfg)
    elif dataset_path == 'uestc':
        return MotionVerseUESTC(**cfg)
    elif dataset_path == 'ntu':
        return MotionVerseNTU(**cfg)
    elif dataset_path == 'h36m':
        return MotionVerseH36M(**cfg)
    elif dataset_path == 'mpi':
        return MotionVerseMPI(**cfg)
    elif dataset_path == 'pw3d':
        return MotionVersePW3D(**cfg)
    elif dataset_path == 'aist':
        return MotionVerseAIST(**cfg)
    elif dataset_path == 'beat':
        return MotionVerseBEAT(**cfg)
    elif dataset_path == 'tedg':
        return MotionVerseTEDG(**cfg)
    elif dataset_path == 'tedex':
        return MotionVerseTEDEx(**cfg)
    elif dataset_path == 's2g3d':
        return MotionVerseS2G3D(**cfg)
    else:
        raise NotImplementedError()
    

@DATASETS.register_module()
class MotionVerse(Dataset):
    """
    A dataset class that handles multiple MotionBench datasets.

    Args:
        dataset_cfgs (list[str]): List of dataset configurations.
        partitions (list[float]): List of partition weights corresponding to the datasets.
        num_data (Optional[int]): Number of data samples to load. Defaults to None.
        data_prefix (str): Path to the directory containing the dataset.
    """

    def __init__(self,
                 dataset_cfgs: List[dict],
                 partitions: List[int],
                 num_data: Optional[int] = None,
                 data_prefix: Optional[str] = None):
        """Load data from multiple datasets."""
        assert min(partitions) >= 0
        assert len(dataset_cfgs) == len(partitions)
        datasets = []
        new_partitions = []
        for idx, cfg in enumerate(dataset_cfgs):
            if partitions[idx] == 0:
                continue
            new_partitions.append(partitions[idx])
            cfg.update({
                'data_prefix': data_prefix
            })
            datasets.append(create_single_dataset(cfg))
        self.dataset = ConcatDataset(datasets)
        if num_data is not None:
            self.length = num_data
        else:
            self.length = max(len(ds) for ds in datasets)
        partitions = new_partitions
        weights = [np.ones(len(ds)) * p / len(ds) for (p, ds) in zip(partitions, datasets)]
        weights = np.concatenate(weights, axis=0)
        self.weights = weights
        self.task_proj = {
            'mocap': 0,
            't2m': 1,
            'v2m': 2,
            's2g': 3,
            'm2d': 4
        }
        self.task_idx_list = []
        for ds in datasets:
            self.task_idx_list += [self.task_proj[ds.task_name]] * len(ds)

    def __len__(self) -> int:
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> dict:
        """Given an index, sample data from multiple datasets with the specified proportion."""
        return self.dataset[idx]

    def get_task_idx(self, idx: int) -> int:
        return self.task_idx_list[idx]


@DATASETS.register_module()
class MotionVerseEval(Dataset):

    def __init__(self,
                 eval_cfgs: dict,
                 testset: str,
                 test_mode: bool = True):
        """Load data from multiple datasets."""
        assert testset in eval_cfgs
        dataset_path, task_name = testset.split('_')
        dataset_cfg = eval_cfgs[testset]
        dataset_cfg['dataset_path'] = dataset_path
        dataset_cfg['task_name'] = task_name
        dataset_cfg['test_mode'] = test_mode
        self.dataset = create_single_dataset(dataset_cfg)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]
    
    def load_annotation(self):
        self.dataset.load_annotation()

    def prepare_data(self, idx: int) -> dict:
        return self.dataset.prepare_data(idx)

    def prepare_evaluation(self):
        self.dataset.prepare_evaluation()
        
    def process_outputs(self, results):
        return self.dataset.process_outputs(results)

    def evaluate(self, results: List[dict], work_dir: str, logger=None) -> dict:
        return self.dataset.evaluate(results=results, work_dir=work_dir, logger=logger)


@DATASETS.register_module()
class MotionVerseAMASS(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'amass'
        task_name = kwargs['task_name']
        assert task_name in ['mocap']
        super().__init__(**kwargs)


@DATASETS.register_module()
class MotionVerseH3D(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'humanml3d'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 't2m']
        super().__init__(**kwargs)


@DATASETS.register_module()
class MotionVerseKIT(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'kitml'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 't2m']
        super().__init__(**kwargs)


@DATASETS.register_module()
class MotionVerseBABEL(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'babel'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 't2m']
        super().__init__(**kwargs)


@DATASETS.register_module()
class MotionVerseMotionX(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'motionx'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 't2m']
        super().__init__(**kwargs)


@DATASETS.register_module()
class MotionVerseACT12(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'humanact12'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 't2m']
        super().__init__(**kwargs)
    

@DATASETS.register_module()
class MotionVerseUESTC(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'uestc'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 't2m']
        super().__init__(**kwargs)


@DATASETS.register_module()
class MotionVerseNTU(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'ntu'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 't2m']
        super().__init__(**kwargs)
        
        
@DATASETS.register_module()
class MotionVerseH36M(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'h36m'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 'v2m']
        super().__init__(**kwargs)
        

@DATASETS.register_module()
class MotionVerseMPI(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'mpi'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 'v2m']
        super().__init__(**kwargs)
        

@DATASETS.register_module()
class MotionVersePW3D(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = '3dpw'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 'v2m']
        super().__init__(**kwargs)

        
@DATASETS.register_module()
class MotionVerseAIST(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'aist'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 'm2d']
        super().__init__(**kwargs)


@DATASETS.register_module()
class MotionVerseBEAT(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'beat'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 's2g']
        super().__init__(**kwargs)

        
@DATASETS.register_module()
class MotionVerseTEDG(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'tedg'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 's2g']
        super().__init__(**kwargs)
        
        
@DATASETS.register_module()
class MotionVerseTEDEx(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 'tedex'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 's2g']
        super().__init__(**kwargs)
        
        
@DATASETS.register_module()
class MotionVerseS2G3D(SingleMotionVerseDataset):

    def __init__(self, **kwargs):
        if 'dataset_path' not in kwargs:
            kwargs['dataset_path'] = 's2g3d'
        task_name = kwargs['task_name']
        assert task_name in ['mocap', 's2g']
        super().__init__(**kwargs)
        
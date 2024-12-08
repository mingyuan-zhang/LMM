import copy
import os
import json
from abc import abstractmethod
from typing import Optional, Union, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from mogen.core.evaluation import build_evaluator
from mogen.models.builder import build_submodule
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class BaseMotionDataset(Dataset):
    """
    Base class for motion datasets.

    Args:
        data_prefix (str): The prefix of the data path.
        pipeline (list): A list of dicts, where each element represents an operation 
                         defined in `mogen.datasets.pipelines`.
        dataset_name (Optional[Union[str, None]]): The name of the dataset. Used to 
                         identify the type of evaluation metric.
        fixed_length (Optional[Union[int, None]]): The fixed length of the dataset for 
                         iteration. If None, the dataset length is based on the number 
                         of annotations.
        ann_file (Optional[Union[str, None]]): The annotation file. If it is a string, 
                         it is expected to be read from the file. If None, it will be 
                         read from `data_prefix`.
        motion_dir (Optional[Union[str, None]]): The directory containing motion data.
        eval_cfg (Optional[Union[dict, None]]): Configuration for evaluation metrics.
        test_mode (Optional[bool]): Whether the dataset is in test mode. Default is False.

    Attributes:
        data_infos (list): Loaded dataset annotations.
        evaluators (list): List of evaluation objects.
        eval_indexes (np.ndarray): Array of indices used for evaluation.
        evaluator_model (torch.nn.Module): Model used for evaluation.
        pipeline (Compose): Data processing pipeline.
    """
    
    def __init__(self,
                 data_prefix: str,
                 pipeline: List[Dict],
                 dataset_name: Optional[Union[str, None]] = None,
                 fixed_length: Optional[Union[int, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 motion_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False):
        super(BaseMotionDataset, self).__init__()

        self.data_prefix = data_prefix
        self.pipeline = Compose(pipeline)
        self.dataset_name = dataset_name
        self.fixed_length = fixed_length
        self.ann_file = os.path.join(data_prefix, 'datasets', dataset_name, ann_file)
        self.motion_dir = os.path.join(data_prefix, 'datasets', dataset_name, motion_dir)
        self.eval_cfg = copy.deepcopy(eval_cfg)
        self.test_mode = test_mode

        self.load_annotations()
        if self.test_mode:
            self.prepare_evaluation()

    @abstractmethod
    def load_anno(self, name: str) -> dict:
        """
        Abstract method to load a single annotation.

        Args:
            name (str): Name or identifier of the annotation to load.

        Returns:
            dict: Loaded annotation as a dictionary.
        """
        pass

    def load_annotations(self):
        """Load annotations from `ann_file` to `data_infos`."""
        self.data_infos = []
        idx = 0
        for line in open(self.ann_file, 'r').readlines():
            line = line.strip()
            self.data_infos.append(self.load_anno(idx, line))
            idx += 1

    def prepare_data(self, idx: int) -> dict:
        """
        Prepare raw data for the given index.

        Args:
            idx (int): Index of the data to prepare.

        Returns:
            dict: Processed data for the given index.
        """
        results = copy.deepcopy(self.data_infos[idx])
        results['dataset_name'] = self.dataset_name
        results['sample_idx'] = idx
        return self.pipeline(results)

    def __len__(self) -> int:
        """Return the length of the current dataset.

        Returns:
            int: Length of the dataset.
        """
        if self.test_mode:
            return len(self.eval_indexes)
        elif self.fixed_length is not None:
            return self.fixed_length
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> dict:
        """
        Prepare data for the given index.

        Args:
            idx (int): Index of the data.

        Returns:
            dict: Data for the specified index.
        """
        if self.test_mode:
            idx = self.eval_indexes[idx]
        elif self.fixed_length is not None:
            idx = idx % len(self.data_infos)
        elif self.balanced_sampling:
            cid = np.random.randint(0, len(self.category_list))
            idx = np.random.randint(0, len(self.category_list[cid]))
            idx = self.category_list[cid][idx]
        return self.prepare_data(idx)

    def prepare_evaluation(self):
        """Prepare evaluation settings, including evaluators and evaluation indices."""
        self.evaluators = []
        self.eval_indexes = []
        self.evaluator_model = build_submodule(self.eval_cfg.get('evaluator_model', None))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator_model = self.evaluator_model.to(device)
        self.evaluator_model.eval()
        self.eval_cfg['evaluator_model'] = self.evaluator_model

        for _ in range(self.eval_cfg['replication_times']):
            eval_indexes = np.arange(len(self.data_infos))
            if self.eval_cfg.get('shuffle_indexes', False):
                np.random.shuffle(eval_indexes)
            self.eval_indexes.append(eval_indexes)

        for metric in self.eval_cfg['metrics']:
            evaluator, self.eval_indexes = build_evaluator(
                metric, self.eval_cfg, len(self.data_infos), self.eval_indexes)
            self.evaluators.append(evaluator)

        self.eval_indexes = np.concatenate(self.eval_indexes)

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
        for evaluator in self.evaluators:
            metrics.update(evaluator.evaluate(results))
        if logger is not None:
            logger.info(metrics)
        eval_output = os.path.join(work_dir, 'eval_results.log')
        with open(eval_output, 'w') as f:
            for k, v in metrics.items():
                f.write(k + ': ' + str(v) + '\n')
        return metrics

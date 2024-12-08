import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from typing import Optional, Union, Iterator
import numpy as np


class DistributedSampler(_DistributedSampler):
    """
    A custom distributed sampler that supports shuffling, round-up of the sample size, 
    and ensures deterministic shuffling across epochs.

    Args:
        dataset: The dataset from which samples are drawn.
        num_replicas: Optional; the number of processes participating in the distributed training.
        rank: Optional; the rank of the current process among num_replicas.
        shuffle: Optional; whether to shuffle the dataset every epoch. Defaults to True.
        round_up: Optional; whether to round up the total size to make it divisible among replicas.
                  Defaults to True.

    Attributes:
        shuffle (bool): Whether to shuffle the dataset.
        round_up (bool): Whether to round up the total size to make it evenly divisible among replicas.
        total_size (int): The total number of samples.
    """
    
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 round_up: bool = True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the indices of the dataset, shuffled if required, 
        with optional rounding up to make the number of samples divisible among replicas.

        Returns:
            Iterator[int]: An iterator over the indices for the current rank.
        """
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == self.num_samples

        return iter(indices)
    
    
class DistributedWeightedRandomSampler(_DistributedSampler):    
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 round_up: bool = True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        weights = self.dataset.weights
        indices = np.random.choice(len(weights), size=len(self.dataset), replace=True, p=weights)
        indices = indices.tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == self.num_samples

        return iter(indices)
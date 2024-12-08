from typing import Iterator, List

from torch.utils.data import BatchSampler, Sampler


class MonoTaskBatchSampler(BatchSampler):

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 num_tasks: int,
                 drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._task_buckets = [[] for _ in range(num_tasks)]
        self.num_tasks = num_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for idx in self.sampler:
            bucket_id = self.sampler.dataset.get_task_idx(idx)
            bucket = self._task_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        left_data = []
        for i in range(self.num_tasks):
            if len(self._task_buckets[i]) > 0:
                left_data.append(self._task_buckets[i])

        self._task_buckets = [[] for _ in range(self.num_tasks)]
        for data in left_data:
            yield data
        # while len(left_data) > 0:
        #     if len(left_data) <= self.batch_size:
        #         if not self.drop_last:
        #             yield left_data[:]
        #         left_data = []
        #     else:
        #         yield left_data[:self.batch_size]
        #         left_data = left_data[self.batch_size:]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
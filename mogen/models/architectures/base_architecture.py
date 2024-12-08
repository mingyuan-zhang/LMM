from collections import OrderedDict
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule
from typing import Dict, Tuple, List


def to_cpu(x: torch.Tensor) -> torch.Tensor:
    """Move a tensor to CPU and detach it from the computation graph.
    
    Args:
        x (torch.Tensor): The input tensor.
    
    Returns:
        torch.Tensor: The tensor detached and moved to CPU.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


class BaseArchitecture(BaseModule):
    """Base class for mogen architecture.

    Args:
        init_cfg (dict, optional): Initialization config for the module.
    """

    def __init__(self, init_cfg: dict = None):
        super(BaseArchitecture, self).__init__(init_cfg)

    def forward_train(self, **kwargs):
        """Forward computation during training."""
        pass

    def forward_test(self, **kwargs):
        """Forward computation during testing."""
        pass

    def _parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Parse the raw outputs (losses) of the network.
        
        Args:
            losses (dict): Raw output of the network, which usually contains
                losses and other necessary information.
        
        Returns:
            tuple[Tensor, dict]: (loss, log_vars)
                - loss is the loss tensor which may be a weighted sum of all losses,
                - log_vars contains all the variables to be logged.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """The iteration step during training.

        This method defines an iteration step during training, excluding backpropagation
        and optimizer updating, which are handled by an optimizer hook.

        Args:
            data (dict): The output of the dataloader.
            optimizer (torch.optim.Optimizer): The optimizer object (unused).

        Returns:
            dict: A dictionary containing the loss, log_vars for logging, and the number of samples.
                - ``loss``: A tensor for backpropagation, which may be a weighted sum of multiple losses.
                - ``log_vars``: All the variables to be logged.
                - ``num_samples``: The number of samples in the batch.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['motion']))
        return outputs

    def val_step(self, data: Dict, optimizer: torch.optim.Optimizer = None) -> Dict:
        """The iteration step during validation.

        Args:
            data (dict): The output of the dataloader.
            optimizer (torch.optim.Optimizer, optional): The optimizer object (unused).

        Returns:
            dict: A dictionary containing the loss, log_vars for logging, and the number of samples.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['motion']))
        return outputs

    def forward(self, **kwargs):
        """Forward computation based on the training or testing mode."""
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def split_results(self, results: Dict[str, torch.Tensor]) -> List[Dict]:
        """Split batched results into individual outputs.

        Args:
            results (dict): The batched results from the model containing 'motion', 'pred_motion', etc.

        Returns:
            list: A list of dictionaries where each dictionary contains results for a single instance.
        """
        B = results['motion'].shape[0]
        output = []
        for i in range(B):
            batch_output = dict()
            batch_output['motion'] = to_cpu(results['motion'][i])
            batch_output['pred_motion'] = to_cpu(results['pred_motion'][i])
            batch_output['motion_length'] = to_cpu(results['motion_length'][i])
            batch_output['motion'][batch_output['motion_length']:, :] = 0
            batch_output['motion_mask'] = to_cpu(results['motion_mask'][i])
            if 'pred_motion_length' in results:
                batch_output['pred_motion_length'] = to_cpu(results['pred_motion_length'][i])
            else:
                batch_output['pred_motion_length'] = to_cpu(results['motion_length'][i])
            batch_output['pred_motion'][batch_output['pred_motion_length']:, :] = 0
            if 'pred_motion_mask' in results:
                batch_output['pred_motion_mask'] = to_cpu(results['pred_motion_mask'][i])
            else:
                batch_output['pred_motion_mask'] = to_cpu(results['motion_mask'][i])
            if 'motion_metas' in results:
                motion_metas = results['motion_metas'][i]
                if 'text' in motion_metas:
                    batch_output['text'] = motion_metas['text']
                if 'token' in motion_metas:
                    batch_output['token'] = motion_metas['token']
                if 'meta_data' in motion_metas and 'category_id' in motion_metas['meta_data']:
                    batch_output['category_id'] = motion_metas['meta_data']['category_id']
                batch_output['motion_metas'] = motion_metas
            output.append(batch_output)
        return output

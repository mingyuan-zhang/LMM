import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Union

from ..builder import ARCHITECTURES, build_loss, build_submodule
from ..utils.gaussian_diffusion import create_named_schedule_sampler, build_diffusion
from ..utils.mask_helper import expand_mask_to_all
from .base_architecture import BaseArchitecture


def set_requires_grad(nets: Union[torch.nn.Module, List[torch.nn.Module]], requires_grad: bool = False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single network.
        requires_grad (bool): Whether the networks require gradients or not.
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


@ARCHITECTURES.register_module()
class MotionDiffusion(BaseArchitecture):
    """
    Motion Diffusion architecture for modeling and generating motion sequences using diffusion models.

    Args:
        dataset_name (Optional[str]): Name of the dataset being used (e.g., 'kit_ml', 'human_ml3d').
        model (dict): Configuration for the submodule (e.g., the motion generation model).
        loss_recon (dict): Configuration for the reconstruction loss.
        loss_reduction (str): Specifies the reduction method for the loss. Defaults to 'frame'.
        use_loss_score (bool): Whether to use a scoring mechanism for loss calculation. Defaults to False.
        diffusion_train (dict): Configuration for the diffusion model during training.
        diffusion_test (dict): Configuration for the diffusion model during testing.
        sampler_type (str): The type of sampler to use. Defaults to 'uniform'.
        init_cfg (dict): Initialization config for the module.
        inference_type (str): Type of inference to use ('ddpm' or 'ddim'). Defaults to 'ddpm'.
    """

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 model: dict = None,
                 loss_recon: dict = None,
                 loss_reduction: str = "frame",
                 use_loss_score: bool = False,
                 diffusion_train: dict = None,
                 diffusion_test: dict = None,
                 sampler_type: str = 'uniform',
                 init_cfg: dict = None,
                 inference_type: str = 'ddpm',
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.model = build_submodule(model)
        self.loss_recon = build_loss(loss_recon)
        self.diffusion_train = build_diffusion(diffusion_train)
        self.diffusion_test = build_diffusion(diffusion_test)
        self.sampler = create_named_schedule_sampler(sampler_type, self.diffusion_train)
        self.inference_type = inference_type
        self.loss_reduction = loss_reduction
        self.use_loss_score = use_loss_score
        self.dataset_name = dataset_name

        if self.dataset_name == "kit_ml":
            self.mean = np.load("data/datasets/kit_ml/mean.npy")
            self.std = np.load("data/datasets/kit_ml/std.npy")
        elif self.dataset_name == "human_ml3d":
            self.mean = np.load("data/datasets/human_ml3d/mean.npy")
            self.std = np.load("data/datasets/human_ml3d/std.npy")
        elif self.dataset_name is not None:
            raise NotImplementedError()


    def forward(self, **kwargs) -> Union[Dict, List]:
        """Forward pass of the model.

        Depending on whether the model is in training mode, this method performs the forward pass
        during training or inference, and calculates the relevant losses.

        Args:
            **kwargs: Keyword arguments containing the input data for the model.

        Returns:
            dict or list: The calculated losses during training or the generated motion during inference.
        """
        motion = kwargs['motion'].float()
        motion_mask = kwargs['motion_mask'].float()
        motion_length = kwargs['motion_length']
        num_intervals = kwargs.get('num_intervals', 1)
        sample_idx = kwargs.get('sample_idx', None)
        clip_feat = kwargs.get('clip_feat', None)
        B, T = motion.shape[:2]
        text = [kwargs['motion_metas'][i]['text'] for i in range(B)]

        if self.training:
            t, _ = self.sampler.sample(B, motion.device)
            output = self.diffusion_train.training_losses(
                model=self.model,
                x_start=motion,
                t=t,
                model_kwargs={
                    'motion_mask': motion_mask,
                    'motion_length': motion_length,
                    'text': text,
                    'clip_feat': clip_feat,
                    'sample_idx': sample_idx,
                    'num_intervals': num_intervals
                }
            )
            pred, target = output['pred'], output['target']
            recon_loss = self.loss_recon(pred, target, reduction_override='none')

            if self.use_loss_score:
                loss_score = kwargs['score']
                recon_loss = recon_loss * loss_score.view(B, 1, -1)

            recon_loss = recon_loss.mean(dim=-1) * motion_mask
            recon_loss_batch = recon_loss.sum(dim=1) / motion_mask.sum(dim=1)
            recon_loss_frame = recon_loss.sum() / motion_mask.sum()

            if self.loss_reduction == "frame":
                recon_loss = recon_loss_frame
            else:
                recon_loss = recon_loss_batch

            if hasattr(self.sampler, "update_with_local_losses"):
                self.sampler.update_with_local_losses(t, recon_loss_batch)

            loss = {'recon_loss': recon_loss.mean()}
            if hasattr(self.model, 'aux_loss'):
                loss.update(self.model.aux_loss())
            return loss

        else:
            dim_pose = kwargs['motion'].shape[-1]
            model_kwargs = self.model.get_precompute_condition(
                device=motion.device, text=text, **kwargs
            )
            model_kwargs.update({
                'motion_mask': motion_mask,
                'sample_idx': sample_idx,
                'motion_length': motion_length,
                'num_intervals': num_intervals
            })

            inference_kwargs = kwargs.get('inference_kwargs', {})
            if self.inference_type == 'ddpm':
                output = self.diffusion_test.p_sample_loop(
                    self.model, (B, T, dim_pose), clip_denoised=False, progress=False,
                    model_kwargs=model_kwargs, **inference_kwargs
                )
            else:
                output = self.diffusion_test.ddim_sample_loop(
                    self.model, (B, T, dim_pose), clip_denoised=False, progress=False,
                    model_kwargs=model_kwargs, eta=0, **inference_kwargs
                )

            results = kwargs
            if getattr(self.model, "post_process") is not None:
                output = self.model.post_process(output)

            results['pred_motion'] = output
            results = self.split_results(results)
            return results


@ARCHITECTURES.register_module()
class UnifiedMotionDiffusion(BaseArchitecture):
    """
    Unified Motion Diffusion architecture for generating motion sequences using diffusion models.
    
    Args:
        model (dict): Configuration for the motion generation model.
        loss_recon (dict): Configuration for the reconstruction loss.
        loss_reduction (str): Specifies the reduction method for the loss. Defaults to 'frame'.
        random_mask (float): Probability or scaling factor for applying random masking. Defaults to 0.
        diffusion_train (dict): Configuration for the diffusion model during training.
        diffusion_test (dict): Configuration for the diffusion model during testing.
        sampler_type (str): The type of sampler to use. Defaults to 'uniform'.
        init_cfg (dict): Initialization config for the module.
        inference_type (str): Type of inference to use ('ddpm' or 'ddim'). Defaults to 'ddpm'.
        body_scale (float): Scaling factor for the body motion mask. Defaults to 1.0.
        hand_scale (float): Scaling factor for the hand motion mask. Defaults to 1.0.
        face_scale (float): Scaling factor for the face motion mask. Defaults to 1.0.
    """

    def __init__(self,
                 model: dict = None,
                 loss_recon: dict = None,
                 loss_reduction: str = "frame",
                 random_mask: float = 0,
                 diffusion_train: dict = None,
                 diffusion_test_dict: dict = None,
                 sampler_type: str = 'uniform',
                 init_cfg: dict = None,
                 inference_type: str = 'ddpm',
                 body_scale: float = 1.0,
                 hand_scale: float = 1.0,
                 face_scale: float = 1.0,
                 train_repeat: int = 1,
                 loss_weight: str = None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.model = build_submodule(model)
        self.loss_recon = build_loss(loss_recon)
        self.diffusion_train = build_diffusion(diffusion_train)
        self.diffusion_test_dict = diffusion_test_dict
        self.sampler = create_named_schedule_sampler(sampler_type, self.diffusion_train)
        self.inference_type = inference_type
        self.loss_reduction = loss_reduction
        self.random_mask = random_mask
        self.body_scale = body_scale
        self.hand_scale = hand_scale
        self.face_scale = face_scale
        self.train_repeat = train_repeat
        if loss_weight is not None:
            self.loss_weight = np.load(loss_weight)
        else:
            self.loss_weight = None
        if init_cfg is not None:
            self.init_weights()

    def repeat_data(self, **kwargs):
        if self.train_repeat == 1:
            return kwargs
        N = self.train_repeat
        motion = kwargs['motion'].float().repeat(N, 1, 1)
        B = motion.shape[0]
        kwargs['motion'] = motion
        
        motion_mask = kwargs['motion_mask'].float().repeat(N, 1, 1)
        kwargs['motion_mask'] = motion_mask
        
        motion_length = kwargs['motion_length'].repeat(N, 1)
        kwargs['motion_length'] = motion_length

        motion_metas = kwargs['motion_metas'] * N
        kwargs['motion_metas'] = motion_metas

        if 'text_seq_feat' in kwargs:
            kwargs['text_seq_feat'] = kwargs['text_seq_feat'].repeat(N, 1)
        if 'text_word_feat' in kwargs:
            kwargs['text_word_feat'] = kwargs['text_word_feat'].repeat(N, 1, 1)
        if 'text_cond' in kwargs:
            kwargs['text_cond'] = kwargs['text_cond'].repeat(N, 1)
            
        if 'music_seq_feat' in kwargs:
            kwargs['music_seq_feat'] = kwargs['music_seq_feat'].repeat(N, 1)
        if 'music_word_feat' in kwargs:
            kwargs['music_word_feat'] = kwargs['music_word_feat'].repeat(N, 1, 1)
        if 'music_cond' in kwargs:
            kwargs['music_cond'] = kwargs['music_cond'].repeat(N, 1)

        if 'speech_seq_feat' in kwargs:
            kwargs['speech_seq_feat'] = kwargs['speech_seq_feat'].repeat(N, 1)
        if 'speech_word_feat' in kwargs:
            kwargs['speech_word_feat'] = kwargs['speech_word_feat'].repeat(N, 1, 1)
        if 'speech_cond' in kwargs:
            kwargs['speech_cond'] = kwargs['speech_cond'].repeat(N, 1)

        if 'video_seq_feat' in kwargs:
            kwargs['video_seq_feat'] = kwargs['video_seq_feat'].repeat(N, 1)
        if 'video_word_feat' in kwargs:
            kwargs['video_word_feat'] = kwargs['video_word_feat'].repeat(N, 1, 1)
        if 'video_cond' in kwargs:
            kwargs['video_cond'] = kwargs['video_cond'].repeat(N, 1)
        return kwargs


    def forward(self, **kwargs) -> Dict:
        """Forward pass for training or inference in the unified motion diffusion model.
        
        Args:
            **kwargs: Keyword arguments containing the input data for the model.
        
        Returns:
            dict: The calculated losses during training or the generated motion during inference.
        """
        if self.training:
            kwargs = self.repeat_data(**kwargs)
            
        motion = kwargs['motion'].float()
        B, T = motion.shape[:2]
        motion_mask = kwargs['motion_mask'].float()
        motion_length = kwargs['motion_length']
        num_intervals = kwargs.get('num_intervals', 1)
        sample_idx = kwargs.get('sample_idx', None)
        motion_metas = kwargs['motion_metas']

        # Conditioning features (text, music, speech, video)
        text_word_feat = kwargs.get('text_word_feat', None)
        text_seq_feat = kwargs.get('text_seq_feat', None)
        text_cond = kwargs.get('text_cond', torch.zeros(B).type_as(motion))

        music_word_feat = kwargs.get('music_word_feat', None)
        music_seq_feat = kwargs.get('music_seq_feat', None)
        music_cond = kwargs.get('music_cond', torch.zeros(B).type_as(motion))

        speech_word_feat = kwargs.get('speech_word_feat', None)
        speech_seq_feat = kwargs.get('speech_seq_feat', None)
        speech_cond = kwargs.get('speech_cond', torch.zeros(B).type_as(motion))

        video_word_feat = kwargs.get('video_word_feat', None)
        video_seq_feat = kwargs.get('video_seq_feat', None)
        video_cond = kwargs.get('video_cond', torch.zeros(B).type_as(motion))

        if self.training:
            # Random masking during training
            t, _ = self.sampler.sample(B, motion.device)
            
            # rand_mask = torch.rand_like(motion_mask)
            # new_motion_mask = motion_mask.clone()
            # threshold = torch.rand(B).type_as(rand_mask)
            # threshold = threshold.view(B, 1, 1).repeat(1, T, 10)
            # new_motion_mask[rand_mask < threshold] = 0
            # motion_mask = new_motion_mask
            
            output = self.diffusion_train.training_losses(
                model=self.model,
                x_start=motion,
                t=t,
                model_kwargs={
                    'motion_mask': motion_mask,
                    'motion_length': motion_length,
                    'num_intervals': num_intervals,
                    'motion_metas': motion_metas,
                    'text_word_feat': text_word_feat,
                    'text_seq_feat': text_seq_feat,
                    'text_cond': text_cond,
                    'music_word_feat': music_word_feat,
                    'music_seq_feat': music_seq_feat,
                    'music_cond': music_cond,
                    'speech_word_feat': speech_word_feat,
                    'speech_seq_feat': speech_seq_feat,
                    'speech_cond': speech_cond,
                    'video_word_feat': video_word_feat,
                    'video_seq_feat': video_seq_feat,
                    'video_cond': video_cond,
                })
            pred, target = output['pred'], output['target']
            recon_loss = self.loss_recon(pred, target, reduction_override='none')
            # Apply expanded motion mask
            motion_mask = expand_mask_to_all(
                motion_mask, self.body_scale, self.hand_scale, self.face_scale
            )
            if self.loss_weight is not None:
                loss_weight = torch.from_numpy(self.loss_weight).type_as(motion_mask)
                dataset_idx = self.model.dataset_idx
                loss_weight = loss_weight.index_select(0, dataset_idx).unsqueeze(1)
                motion_mask = motion_mask * loss_weight
                recon_loss = (recon_loss * motion_mask).sum(dim=-1)
                motion_mask = motion_mask.sum(dim=-1)
            else:
                recon_loss = (recon_loss * motion_mask).mean(dim=-1)
                motion_mask = motion_mask.mean(dim=-1)

            recon_loss_batch = recon_loss.sum(dim=1) / motion_mask.sum(dim=1)
            recon_loss_frame = recon_loss.sum() / motion_mask.sum()

            # Determine final reconstruction loss
            if self.loss_reduction == "frame":
                recon_loss = recon_loss_frame
            else:
                recon_loss = recon_loss_batch

            if hasattr(self.sampler, "update_with_local_losses"):
                self.sampler.update_with_local_losses(t, recon_loss_batch)

            loss = {'recon_loss': recon_loss.mean()}

            # Add auxiliary loss if applicable
            if hasattr(self.model, 'aux_loss'):
                loss.update(self.model.aux_loss())

            return loss
        else:
            # Inference (DDPM or DDIM sampling)
            dim_pose = 669  # Fixed dimension for the motion output
            model_kwargs = self.model.get_precompute_condition(
                device=motion.device, **kwargs
            )
            model_kwargs.update({
                'motion_mask': motion_mask,
                'sample_idx': sample_idx,
                'motion_length': motion_length,
                'num_intervals': num_intervals,
                'motion_metas': motion_metas,
                'text_word_feat': text_word_feat,
                'text_seq_feat': text_seq_feat,
                'text_cond': text_cond,
                'music_word_feat': music_word_feat,
                'music_seq_feat': music_seq_feat,
                'music_cond': music_cond,
                'speech_word_feat': speech_word_feat,
                'speech_seq_feat': speech_seq_feat,
                'speech_cond': speech_cond,
                'video_word_feat': video_word_feat,
                'video_seq_feat': video_seq_feat,
                'video_cond': video_cond,
            })
            inference_kwargs = kwargs.get('inference_kwargs', {})
            inference_kwargs['gt_motion'] = motion
            inference_kwargs['context_mask'] = kwargs.get('context_mask', None)
            dataset_name = motion_metas[0]['meta_data']['dataset_name']
            diffusion_test_cfg = self.diffusion_test_dict['base']
            diffusion_test_cfg.update(dict(respace=self.diffusion_test_dict[dataset_name]))
            diffusion_test = build_diffusion(diffusion_test_cfg)
            if self.inference_type == 'ddpm':
                output = diffusion_test.p_sample_loop(
                    self.model, (B, T, dim_pose), clip_denoised=False,
                    progress=False, model_kwargs=model_kwargs, **inference_kwargs
                )
            else:
                output = diffusion_test.ddim_sample_loop(
                    self.model, (B, T, dim_pose), clip_denoised=False,
                    progress=False, model_kwargs=model_kwargs, eta=0,
                    **inference_kwargs
                )

            results = kwargs
            if getattr(self.model, "post_process") is not None:
                output = self.model.post_process(output)

            results['pred_motion'] = output
            results = self.split_results(results)

            return results

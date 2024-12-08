import numpy as np
import torch

from typing import Optional, Dict, List

from ..builder import SUBMODULES
from .motion_transformer import MotionTransformer


@SUBMODULES.register_module()
class MotionDiffuseTransformer(MotionTransformer):
    """
    MotionDiffuseTransformer is a subclass of DiffusionTransformer designed for motion generation.
    It uses a diffusion-based approach with optional guidance during training and inference.

    Args:
        guidance_cfg (dict, optional): Configuration for guidance during inference and training.
                                        'type' can be 'constant' or dynamically calculated based on timesteps.
        kwargs: Additional keyword arguments for the DiffusionTransformer base class.
    """

    def __init__(self, guidance_cfg: Optional[dict] = None, **kwargs):
        """
        Initialize the MotionDiffuseTransformer.

        Args:
            guidance_cfg (Optional[dict]): Configuration for the guidance.
            kwargs: Additional arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self.guidance_cfg = guidance_cfg

    def scale_func(self, timestep: int) -> dict:
        """
        Compute the scaling coefficients for text-based guidance and no-guidance.

        Args:
            timestep (int): The current diffusion timestep.

        Returns:
            dict: A dictionary containing 'text_coef' and 'none_coef' that control the mix of text-conditioned and 
                  non-text-conditioned outputs.
        """
        if self.guidance_cfg['type'] == 'constant':
            w = self.guidance_cfg['scale']
            return {'text_coef': w, 'none_coef': 1 - w}
        else:
            scale = self.guidance_cfg['scale']
            w = (1 - (1000 - timestep) / 1000) * scale + 1
            output = {'text_coef': w, 'none_coef': 1 - w}
        return output

    def get_precompute_condition(self,
                                 text: Optional[torch.Tensor] = None,
                                 xf_proj: Optional[torch.Tensor] = None,
                                 xf_out: Optional[torch.Tensor] = None,
                                 device: Optional[torch.device] = None,
                                 clip_feat: Optional[torch.Tensor] = None,
                                 **kwargs) -> dict:
        """
        Precompute the conditions for text-based guidance using a text encoder.

        Args:
            text (Optional[torch.Tensor]): The input text data.
            xf_proj (Optional[torch.Tensor]): Precomputed text projection.
            xf_out (Optional[torch.Tensor]): Precomputed output from the text encoder.
            device (Optional[torch.device]): The device on which the model is running.
            clip_feat (Optional[torch.Tensor]): CLIP features for text guidance.
            kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the text projection and output from the encoder.
        """
        if xf_out is None:
            if self.use_text_proj:
                xf_proj, xf_out = self.encode_text(text, clip_feat, device)
            else:
                xf_out = self.encode_text(text, clip_feat, device)
        return {'xf_proj': xf_proj, 'xf_out': xf_out}

    def post_process(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Post-process the generated motion data by re-normalizing it using mean and standard deviation.

        Args:
            motion (torch.Tensor): The generated motion data.

        Returns:
            torch.Tensor: Post-processed motion data.
        """
        if self.post_process_cfg is not None:
            if self.post_process_cfg.get("unnormalized_infer", False):
                mean = torch.from_numpy(np.load(self.post_process_cfg['mean_path']))
                mean = mean.type_as(motion)
                std = torch.from_numpy(np.load(self.post_process_cfg['std_path']))
                std = std.type_as(motion)
            motion = motion * std + mean
        return motion

    def forward_train(self,
                      h: torch.Tensor,
                      src_mask: Optional[torch.Tensor] = None,
                      emb: Optional[torch.Tensor] = None,
                      xf_out: Optional[torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        """
        Forward pass during training.

        Args:
            h (torch.Tensor): Input motion tensor of shape (B, T, D).
            src_mask (Optional[torch.Tensor]): Source mask for masking the input.
            emb (torch.Tensor): Time-step embeddings.
            xf_out (Optional[torch.Tensor]): Precomputed output from the text encoder.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output motion data after processing by the temporal decoder blocks.
        """
        B, T = h.shape[0], h.shape[1]
        if self.guidance_cfg is None:
            for module in self.temporal_decoder_blocks:
                h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask)
        else:
            cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
            for module in self.temporal_decoder_blocks:
                h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=cond_type)
        output = self.out(h).view(B, T, -1).contiguous()
        return output

    def forward_test(self,
                     h: torch.Tensor,
                     src_mask: Optional[torch.Tensor] = None,
                     emb: Optional[torch.Tensor] = None,
                     xf_out: Optional[torch.Tensor] = None,
                     timesteps: Optional[torch.Tensor] = None,
                     **kwargs) -> torch.Tensor:
        """
        Forward pass during testing/inference.

        Args:
            h (torch.Tensor): Input motion tensor of shape (B, T, D).
            src_mask (Optional[torch.Tensor]): Source mask for masking the input.
            emb (torch.Tensor): Time-step embeddings.
            xf_out (Optional[torch.Tensor]): Precomputed output from the text encoder.
            timesteps (Optional[torch.Tensor]): Current diffusion timesteps.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output motion data after processing by the temporal decoder blocks.
        """
        B, T = h.shape[0], h.shape[1]
        if self.guidance_cfg is None:
            for module in self.temporal_decoder_blocks:
                h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask)
            output = self.out(h).view(B, T, -1).contiguous()
        else:
            text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
            none_cond_type = torch.zeros(B, 1, 1).to(h.device)
            all_cond_type = torch.cat((text_cond_type, none_cond_type), dim=0)
            h = h.repeat(2, 1, 1)
            xf_out = xf_out.repeat(2, 1, 1)
            emb = emb.repeat(2, 1)
            src_mask = src_mask.repeat(2, 1, 1)
            for module in self.temporal_decoder_blocks:
                h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=all_cond_type)
            out = self.out(h).view(2 * B, T, -1).contiguous()
            out_text = out[:B].contiguous()
            out_none = out[B:].contiguous()
            coef_cfg = self.scale_func(int(timesteps[0]))
            text_coef = coef_cfg['text_coef']
            none_coef = coef_cfg['none_coef']
            output = out_text * text_coef + out_none * none_coef
        return output

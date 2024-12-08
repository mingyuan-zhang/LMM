import numpy as np
import torch
from torch import nn

from typing import Optional, Dict, List

from mogen.models.utils.misc import zero_module

from ..builder import SUBMODULES, build_attention
from ..utils.stylization_block import StylizationBlock
from .motion_transformer import MotionTransformer


def get_kit_slice(idx: int) -> List[int]:
    """
    Get the slice indices for the KIT skeleton.

    Args:
        idx (int): The index of the skeleton part.

    Returns:
        List[int]: Slice indices for the specified skeleton part.
    """
    if idx == 0:
        return [0, 1, 2, 3, 184, 185, 186, 247, 248, 249, 250]
    return [
        4 + (idx - 1) * 3,
        4 + (idx - 1) * 3 + 1,
        4 + (idx - 1) * 3 + 2,
        64 + (idx - 1) * 6,
        64 + (idx - 1) * 6 + 1,
        64 + (idx - 1) * 6 + 2,
        64 + (idx - 1) * 6 + 3,
        64 + (idx - 1) * 6 + 4,
        64 + (idx - 1) * 6 + 5,
        184 + idx * 3,
        184 + idx * 3 + 1,
        184 + idx * 3 + 2,
    ]


def get_t2m_slice(idx: int) -> List[int]:
    """
    Get the slice indices for the T2M skeleton.

    Args:
        idx (int): The index of the skeleton part.

    Returns:
        List[int]: Slice indices for the specified skeleton part.
    """
    if idx == 0:
        return [0, 1, 2, 3, 193, 194, 195, 259, 260, 261, 262]
    return [
        4 + (idx - 1) * 3,
        4 + (idx - 1) * 3 + 1,
        4 + (idx - 1) * 3 + 2,
        67 + (idx - 1) * 6,
        67 + (idx - 1) * 6 + 1,
        67 + (idx - 1) * 6 + 2,
        67 + (idx - 1) * 6 + 3,
        67 + (idx - 1) * 6 + 4,
        67 + (idx - 1) * 6 + 5,
        193 + idx * 3,
        193 + idx * 3 + 1,
        193 + idx * 3 + 2,
    ]


def get_part_slice(idx_list: List[int], func) -> List[int]:
    """
    Get the slice indices for a list of indices.

    Args:
        idx_list (List[int]): List of part indices.
        func (Callable): Function to get slice indices for each part.

    Returns:
        List[int]: Concatenated list of slice indices for the parts.
    """
    result = []
    for idx in idx_list:
        result.extend(func(idx))
    return result


class PoseEncoder(nn.Module):
    """
    Pose Encoder to process motion data and encode body parts into latent representations.
    """

    def __init__(self,
                 dataset_name: str = "human_ml3d",
                 latent_dim: int = 64,
                 input_dim: int = 263):
        super().__init__()
        self.dataset_name = dataset_name
        if dataset_name == "human_ml3d":
            func = get_t2m_slice
            self.head_slice = get_part_slice([12, 15], func)
            self.stem_slice = get_part_slice([3, 6, 9], func)
            self.larm_slice = get_part_slice([14, 17, 19, 21], func)
            self.rarm_slice = get_part_slice([13, 16, 18, 20], func)
            self.lleg_slice = get_part_slice([2, 5, 8, 11], func)
            self.rleg_slice = get_part_slice([1, 4, 7, 10], func)
            self.root_slice = get_part_slice([0], func)
            self.body_slice = get_part_slice([_ for _ in range(22)], func)
        elif dataset_name == "kit_ml":
            func = get_kit_slice
            self.head_slice = get_part_slice([4], func)
            self.stem_slice = get_part_slice([1, 2, 3], func)
            self.larm_slice = get_part_slice([8, 9, 10], func)
            self.rarm_slice = get_part_slice([5, 6, 7], func)
            self.lleg_slice = get_part_slice([16, 17, 18, 19, 20], func)
            self.rleg_slice = get_part_slice([11, 12, 13, 14, 15], func)
            self.root_slice = get_part_slice([0], func)
            self.body_slice = get_part_slice([_ for _ in range(21)], func)
        else:
            raise ValueError()

        self.head_embed = nn.Linear(len(self.head_slice), latent_dim)
        self.stem_embed = nn.Linear(len(self.stem_slice), latent_dim)
        self.larm_embed = nn.Linear(len(self.larm_slice), latent_dim)
        self.rarm_embed = nn.Linear(len(self.rarm_slice), latent_dim)
        self.lleg_embed = nn.Linear(len(self.lleg_slice), latent_dim)
        self.rleg_embed = nn.Linear(len(self.rleg_slice), latent_dim)
        self.root_embed = nn.Linear(len(self.root_slice), latent_dim)
        self.body_embed = nn.Linear(len(self.body_slice), latent_dim)

        assert len(set(self.body_slice)) == input_dim

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for encoding the motion into body part embeddings.

        Args:
            motion (torch.Tensor): Input motion tensor of shape (B, T, D).

        Returns:
            torch.Tensor: Concatenated latent representations of body parts.
        """
        head_feat = self.head_embed(motion[:, :, self.head_slice].contiguous())
        stem_feat = self.stem_embed(motion[:, :, self.stem_slice].contiguous())
        larm_feat = self.larm_embed(motion[:, :, self.larm_slice].contiguous())
        rarm_feat = self.rarm_embed(motion[:, :, self.rarm_slice].contiguous())
        lleg_feat = self.lleg_embed(motion[:, :, self.lleg_slice].contiguous())
        rleg_feat = self.rleg_embed(motion[:, :, self.rleg_slice].contiguous())
        root_feat = self.root_embed(motion[:, :, self.root_slice].contiguous())
        body_feat = self.body_embed(motion[:, :, self.body_slice].contiguous())
        feat = torch.cat((head_feat, stem_feat, larm_feat, rarm_feat,
                          lleg_feat, rleg_feat, root_feat, body_feat),
                         dim=-1)
        return feat


class PoseDecoder(nn.Module):
    """
    Pose Decoder to decode the latent representations of body parts back into motion.
    """

    def __init__(self,
                 dataset_name: str = "human_ml3d",
                 latent_dim: int = 64,
                 output_dim: int = 263):
        super().__init__()
        self.dataset_name = dataset_name
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        if dataset_name == "human_ml3d":
            func = get_t2m_slice
            self.head_slice = get_part_slice([12, 15], func)
            self.stem_slice = get_part_slice([3, 6, 9], func)
            self.larm_slice = get_part_slice([14, 17, 19, 21], func)
            self.rarm_slice = get_part_slice([13, 16, 18, 20], func)
            self.lleg_slice = get_part_slice([2, 5, 8, 11], func)
            self.rleg_slice = get_part_slice([1, 4, 7, 10], func)
            self.root_slice = get_part_slice([0], func)
            self.body_slice = get_part_slice([_ for _ in range(22)], func)
        elif dataset_name == "kit_ml":
            func = get_kit_slice
            self.head_slice = get_part_slice([4], func)
            self.stem_slice = get_part_slice([1, 2, 3], func)
            self.larm_slice = get_part_slice([8, 9, 10], func)
            self.rarm_slice = get_part_slice([5, 6, 7], func)
            self.lleg_slice = get_part_slice([16, 17, 18, 19, 20], func)
            self.rleg_slice = get_part_slice([11, 12, 13, 14, 15], func)
            self.root_slice = get_part_slice([0], func)
            self.body_slice = get_part_slice([_ for _ in range(21)], func)
        else:
            raise ValueError()

        self.head_out = nn.Linear(latent_dim, len(self.head_slice))
        self.stem_out = nn.Linear(latent_dim, len(self.stem_slice))
        self.larm_out = nn.Linear(latent_dim, len(self.larm_slice))
        self.rarm_out = nn.Linear(latent_dim, len(self.rarm_slice))
        self.lleg_out = nn.Linear(latent_dim, len(self.lleg_slice))
        self.rleg_out = nn.Linear(latent_dim, len(self.rleg_slice))
        self.root_out = nn.Linear(latent_dim, len(self.root_slice))
        self.body_out = nn.Linear(latent_dim, len(self.body_slice))

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to decode the latent body part features back to motion.

        Args:
            motion (torch.Tensor): Input tensor of shape (B, T, D).

        Returns:
            torch.Tensor: Output motion tensor of shape (B, T, output_dim).
        """
        B, T = motion.shape[:2]
        D = self.latent_dim
        head_feat = self.head_out(motion[:, :, :D].contiguous())
        stem_feat = self.stem_out(motion[:, :, D:2 * D].contiguous())
        larm_feat = self.larm_out(motion[:, :, 2 * D:3 * D].contiguous())
        rarm_feat = self.rarm_out(motion[:, :, 3 * D:4 * D].contiguous())
        lleg_feat = self.lleg_out(motion[:, :, 4 * D:5 * D].contiguous())
        rleg_feat = self.rleg_out(motion[:, :, 5 * D:6 * D].contiguous())
        root_feat = self.root_out(motion[:, :, 6 * D:7 * D].contiguous())
        body_feat = self.body_out(motion[:, :, 7 * D:].contiguous())
        output = torch.zeros(B, T, self.output_dim).type_as(motion)
        output[:, :, self.head_slice] = head_feat
        output[:, :, self.stem_slice] = stem_feat
        output[:, :, self.larm_slice] = larm_feat
        output[:, :, self.rarm_slice] = rarm_feat
        output[:, :, self.lleg_slice] = lleg_feat
        output[:, :, self.rleg_slice] = rleg_feat
        output[:, :, self.root_slice] = root_feat
        output = (output + body_feat) / 2.0
        return output


class SFFN(nn.Module):
    """
    A Stylized Feed-Forward Network (SFFN) module for transformer layers.

    Args:
        latent_dim (int): Dimensionality of the input.
        ffn_dim (int): Dimensionality of the feed-forward layer.
        dropout (float): Dropout probability.
        time_embed_dim (int): Dimensionality of the time embedding.
        norm (str): Normalization type ('None').
        activation (str): Activation function ('GELU').
    """

    def __init__(self,
                 latent_dim: int,
                 ffn_dim: int,
                 dropout: float,
                 time_embed_dim: int,
                 norm: str = "None",
                 activation: str = "GELU",
                 **kwargs):
        super().__init__()
        self.linear1_list = nn.ModuleList()
        self.linear2_list = nn.ModuleList()

        channel_mul = 1
        if activation == "GELU":
            self.activation = nn.GELU()

        for i in range(8):
            self.linear1_list.append(nn.Linear(latent_dim, ffn_dim * channel_mul))
            self.linear2_list.append(nn.Linear(ffn_dim, latent_dim))

        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim * 8, time_embed_dim, dropout)

        if norm == "None":
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the SFFN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).
            emb (torch.Tensor): Embedding tensor for time step modulation.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).
        """
        B, T, D = x.shape
        x = self.norm(x)
        x = x.reshape(B, T, 8, -1)
        output = []
        for i in range(8):
            feat = x[:, :, i].contiguous()
            feat = self.dropout(self.activation(self.linear1_list[i](feat)))
            feat = self.linear2_list[i](feat)
            output.append(feat)
        y = torch.cat(output, dim=-1)
        y = x.reshape(B, T, D) + self.proj_out(y, emb)
        return y


class DecoderLayer(nn.Module):
    """
    A transformer decoder layer with cross-attention and feed-forward network (SFFN).
    
    Args:
        ca_block_cfg (Optional[Dict]): Configuration for the cross-attention block.
        ffn_cfg (Optional[Dict]): Configuration for the feed-forward network (SFFN).
    """

    def __init__(self, ca_block_cfg: Optional[Dict] = None, ffn_cfg: Optional[Dict] = None):
        super().__init__()
        self.ca_block = build_attention(ca_block_cfg)
        self.ffn = SFFN(**ffn_cfg)

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass of the decoder layer.

        Args:
            kwargs: Keyword arguments for attention and feed-forward layers.

        Returns:
            torch.Tensor: Output of the decoder layer.
        """
        if self.ca_block is not None:
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


@SUBMODULES.register_module()
class FineMoGenTransformer(MotionTransformer):
    """
    A transformer model for motion generation using fine-grained control with Diffusion.

    Args:
        scale_func_cfg (Optional[Dict]): Configuration for scaling function.
        pose_encoder_cfg (Optional[Dict]): Configuration for the PoseEncoder.
        pose_decoder_cfg (Optional[Dict]): Configuration for the PoseDecoder.
        moe_route_loss_weight (float): Weight for the Mixture of Experts (MoE) routing loss.
        template_kl_loss_weight (float): Weight for the KL loss in template generation.
        fine_mode (bool): Whether to enable fine mode for control over body parts.
    """

    def __init__(self,
                 scale_func_cfg: Optional[Dict] = None,
                 pose_encoder_cfg: Optional[Dict] = None,
                 pose_decoder_cfg: Optional[Dict] = None,
                 moe_route_loss_weight: float = 1.0,
                 template_kl_loss_weight: float = 0.0001,
                 fine_mode: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.scale_func_cfg = scale_func_cfg
        self.joint_embed = PoseEncoder(**pose_encoder_cfg)
        self.out = zero_module(PoseDecoder(**pose_decoder_cfg))
        self.moe_route_loss_weight = moe_route_loss_weight
        self.template_kl_loss_weight = template_kl_loss_weight
        self.mean = np.load("data/datasets/kit_ml/mean.npy")
        self.std = np.load("data/datasets/kit_ml/std.npy")
        self.fine_mode = fine_mode

    def build_temporal_blocks(self, sa_block_cfg: Optional[Dict], ca_block_cfg: Optional[Dict], ffn_cfg: Optional[Dict]):
        """
        Build temporal decoder blocks for the model.

        Args:
            sa_block_cfg (Optional[Dict]): Configuration for self-attention blocks.
            ca_block_cfg (Optional[Dict]): Configuration for cross-attention blocks.
            ffn_cfg (Optional[Dict]): Configuration for feed-forward networks.
        """
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if isinstance(ffn_cfg, list):
                ffn_cfg_block = ffn_cfg[i]
            else:
                ffn_cfg_block = ffn_cfg
            self.temporal_decoder_blocks.append(DecoderLayer(ca_block_cfg=ca_block_cfg, ffn_cfg=ffn_cfg_block))

    def scale_func(self, timestep: int) -> Dict[str, float]:
        """
        Scaling function for text and none coefficient based on timestep.

        Args:
            timestep (int): Current diffusion timestep.

        Returns:
            Dict[str, float]: Scaling factors for text and non-text conditioning.
        """
        scale = self.scale_func_cfg['scale']
        w = (1 - (1000 - timestep) / 1000) * scale + 1
        return {'text_coef': w, 'none_coef': 1 - w}

    def aux_loss(self) -> Dict[str, torch.Tensor]:
        """
        Auxiliary loss computation for MoE routing and KL loss.

        Returns:
            Dict[str, torch.Tensor]: Computed auxiliary losses.
        """
        aux_loss = 0
        kl_loss = 0
        for module in self.temporal_decoder_blocks:
            if hasattr(module.ca_block, 'aux_loss'):
                aux_loss = aux_loss + module.ca_block.aux_loss
            if hasattr(module.ca_block, 'kl_loss'):
                kl_loss = kl_loss + module.ca_block.kl_loss
        losses = {}
        if aux_loss > 0:
            losses['moe_route_loss'] = aux_loss * self.moe_route_loss_weight
        if kl_loss > 0:
            losses['template_kl_loss'] = kl_loss * self.template_kl_loss_weight
        return losses

    def get_precompute_condition(self,
                                 text: Optional[str] = None,
                                 motion_length: Optional[torch.Tensor] = None,
                                 xf_out: Optional[torch.Tensor] = None,
                                 re_dict: Optional[Dict] = None,
                                 device: Optional[torch.device] = None,
                                 sample_idx: Optional[int] = None,
                                 clip_feat: Optional[torch.Tensor] = None,
                                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Precompute conditioning features for text or other modalities.

        Args:
            text (Optional[str]): Text input for conditioning.
            motion_length (Optional[torch.Tensor]): Length of the motion sequence.
            xf_out (Optional[torch.Tensor]): Precomputed text features.
            re_dict (Optional[Dict]): Additional features dictionary.
            device (Optional[torch.device]): Target device for the model.
            sample_idx (Optional[int]): Sample index for specific conditioning.
            clip_feat (Optional[torch.Tensor]): Precomputed CLIP features.

        Returns:
            Dict[str, torch.Tensor]: Precomputed conditioning features.
        """
        if xf_out is None:
            xf_out = self.encode_text(text, clip_feat, device)
        output = {'xf_out': xf_out}
        return output

    def post_process(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Post-process motion data by unnormalizing if necessary.

        Args:
            motion (torch.Tensor): Input motion data.

        Returns:
            torch.Tensor: Processed motion data.
        """
        if self.post_process_cfg is not None:
            if self.post_process_cfg.get("unnormalized_infer", False):
                mean = torch.from_numpy(np.load(self.post_process_cfg['mean_path'])).type_as(motion)
                std = torch.from_numpy(np.load(self.post_process_cfg['std_path'])).type_as(motion)
                motion = motion * std + mean
        return motion

    def forward_train(self,
                      h: torch.Tensor,
                      src_mask: Optional[torch.Tensor] = None,
                      emb: Optional[torch.Tensor] = None,
                      xf_out: Optional[torch.Tensor] = None,
                      motion_length: Optional[torch.Tensor] = None,
                      num_intervals: int = 1,
                      **kwargs) -> torch.Tensor:
        """
        Forward pass during training.

        Args:
            h (torch.Tensor): Input tensor of shape (B, T, D).
            src_mask (Optional[torch.Tensor]): Source mask tensor.
            emb (Optional[torch.Tensor]): Time embedding tensor.
            xf_out (Optional[torch.Tensor]): Precomputed text features.
            motion_length (Optional[torch.Tensor]): Lengths of motion sequences.
            num_intervals (int): Number of intervals for processing.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).
        """
        B, T = h.shape[0], h.shape[1]
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).repeat(1, 8, 1).to(h.device) if self.fine_mode else torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                       xf=xf_out,
                       emb=emb,
                       src_mask=src_mask,
                       cond_type=cond_type,
                       motion_length=motion_length,
                       num_intervals=num_intervals)

        output = self.out(h).view(B, T, -1).contiguous()
        return output

    def forward_test(self,
                     h: torch.Tensor,
                     src_mask: Optional[torch.Tensor] = None,
                     emb: Optional[torch.Tensor] = None,
                     xf_out: Optional[torch.Tensor] = None,
                     timesteps: Optional[torch.Tensor] = None,
                     motion_length: Optional[torch.Tensor] = None,
                     num_intervals: int = 1,
                     **kwargs) -> torch.Tensor:
        """
        Forward pass during inference.

        Args:
            h (torch.Tensor): Input tensor of shape (B, T, D).
            src_mask (Optional[torch.Tensor]): Source mask tensor.
            emb (Optional[torch.Tensor]): Time embedding tensor.
            xf_out (Optional[torch.Tensor]): Precomputed text features.
            timesteps (Optional[torch.Tensor]): Diffusion timesteps.
            motion_length (Optional[torch.Tensor]): Lengths of motion sequences.
            num_intervals (int): Number of intervals for processing.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).
        """
        B, T = h.shape[0], h.shape[1]
        text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
        none_cond_type = torch.zeros(B, 1, 1).to(h.device)

        all_cond_type = torch.cat((text_cond_type, none_cond_type), dim=0)
        h = h.repeat(2, 1, 1)
        xf_out = xf_out.repeat(2, 1, 1)
        emb = emb.repeat(2, 1)
        src_mask = src_mask.repeat(2, 1, 1)
        motion_length = motion_length.repeat(2, 1)
        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                       xf=xf_out,
                       emb=emb,
                       src_mask=src_mask,
                       cond_type=all_cond_type,
                       motion_length=motion_length,
                       num_intervals=num_intervals)
        out = self.out(h).view(2 * B, T, -1).contiguous()
        out_text = out[:B].contiguous()
        out_none = out[B:].contiguous()

        coef_cfg = self.scale_func(int(timesteps[0]))
        text_coef = coef_cfg['text_coef']
        none_coef = coef_cfg['none_coef']
        output = out_text * text_coef + out_none * none_coef
        return output



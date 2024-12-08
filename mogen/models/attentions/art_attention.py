import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock

try:
    from tutel import moe as tutel_moe
    from tutel import net
except ImportError:
    pass


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    
    Args:
        module (nn.Module): The input PyTorch module.
    
    Returns:
        nn.Module: The module with zeroed parameters.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class MOE(nn.Module):
    """
    Mixture of Experts (MoE) layer with support for time embeddings and optional framerate conditioning.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        topk (int): Number of top experts selected per input.
        input_dim (int): Dimensionality of the input features.
        ffn_dim (int): Dimensionality of the feed-forward network (FFN) used inside each expert.
        output_dim (int): Dimensionality of the output features.
        num_heads (int): Number of attention heads used in the model.
        max_seq_len (int): Maximum sequence length for the input data.
        gate_type (str): Type of gating mechanism used for MoE (e.g., "topk").
        gate_noise (float): Noise added to the gating mechanism for improved exploration.
        framerate (bool, optional): Whether to use framerate-based embedding. Defaults to False.
        embedding (bool, optional): Whether to use positional embeddings. Defaults to True.

    Attributes:
        proj (nn.Linear): Linear projection layer applied after MoE processing.
        activation (nn.GELU): Activation function used in the feed-forward layers.
        model (tutel_moe.moe_layer): The Mixture of Experts layer.
        embedding (torch.nn.Parameter): Positional or framerate-based embedding for input data.
        aux_loss (torch.Tensor): Auxiliary loss from MoE layer for load balancing across experts.
    """

    def __init__(self, num_experts: int, topk: int, input_dim: int, ffn_dim: int, output_dim: int,
                 num_heads: int, max_seq_len: int, gate_type: str, gate_noise: float, embedding: bool = True):
        super().__init__()

        # Linear projection layer to project from input_dim to output_dim
        self.proj = nn.Linear(input_dim, output_dim)
        # Activation function (GELU)
        self.activation = nn.GELU()

        # Initialize Tutel MoE layer with gating and expert setup
        try:
            data_group = net.create_groups_from_world(group_count=1).data_group
        except:
            data_group = None

        self.model = tutel_moe.moe_layer(
            gate_type={
                'type': gate_type,
                'k': topk,
                'fp32_gate': True,
                'gate_noise': gate_noise,
                'capacity_factor': 1.5  # Capacity factor to allow extra room for expert routing
            },
            experts={
                'type': 'ffn',  # Feed-forward expert type
                'count_per_node': num_experts,
                'hidden_size_per_expert': ffn_dim,
                'activation_fn': lambda x: F.gelu(x)  # Activation inside experts
            },
            model_dim=input_dim,
            batch_prioritized_routing=True,  # Prioritize routing based on batch size
            is_gshard_loss=False,  # Whether to use GShard loss for load balancing
            group=data_group
        )

        # Determine whether to use positional embedding or framerate embedding
        self.use_embedding = embedding
        if self.use_embedding:
            self.embedding = nn.Parameter(torch.randn(1, max_seq_len, num_heads, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer with optional framerate embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, D), where
                              B is the batch size,
                              T is the sequence length,
                              H is the number of attention heads,
                              D is the dimensionality of each head.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, H, output_dim), where output_dim is the projected dimensionality.
        """
        B, T, H, D = x.shape

        # Apply positional or framerate-based embedding
        if self.use_embedding:
            # Default positional embedding
            x = x + self.embedding[:, :T, :, :]

        # Flatten the input for MoE processing
        x = x.reshape(-1, D)
        
        # Pass through the Mixture of Experts layer and apply the projection
        y = self.proj(self.activation(self.model(x)))

        # Auxiliary loss for expert load balancing
        self.aux_loss = self.model.l_aux

        # Reshape the output back to (B, T, H, output_dim)
        y = y.reshape(B, T, H, -1)
        
        return y


def get_ffn(latent_dim: int, ffn_dim: int) -> nn.Sequential:
    """
    Create a feed-forward network (FFN) block.

    Args:
        latent_dim (int): Input/output dimension of the FFN.
        ffn_dim (int): Hidden dimension of the FFN.

    Returns:
        nn.Sequential: A sequential block consisting of two linear layers and a GELU activation in between.
    """
    return nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, latent_dim))


@ATTENTIONS.register_module()
class ArtAttention(nn.Module):
    """
    ArtAttention module for attending to multi-modal inputs (e.g., text, music, speech, video) 
    and generating time-dependent motion features using a Mixture of Experts (MoE) mechanism.

    Args:
        latent_dim (int): Dimensionality of the latent representation.
        num_heads (int): Number of attention heads.
        num_experts (int): Number of experts in the Mixture of Experts.
        topk (int): Number of top experts selected by the gating mechanism.
        gate_type (str): Type of gating mechanism for the MoE layer.
        gate_noise (float): Noise level for the gating mechanism.
        ffn_dim (int): Dimensionality of the feed-forward network inside the MoE.
        time_embed_dim (int): Dimensionality of the time embedding for stylization.
        max_seq_len (int): Maximum length of the motion sequence.
        dropout (float): Dropout rate applied to the output of the MoE and attention layers.
        motion_moe_dropout (float): Dropout rate applied to the motion MoE.
        has_text (bool): Whether the input includes text features.
        has_music (bool): Whether the input includes music features.
        has_speech (bool): Whether the input includes speech features.
        has_video (bool): Whether the input includes video features.
        norm (str): Type of normalization layer to use ('LayerNorm' or 'RMSNorm').

    Inputs:
        - x (torch.Tensor): Tensor of shape (B, T, D), where B is the batch size, 
          T is the sequence length, and D is the dimensionality of the input motion data.
        - emb (torch.Tensor): Time embedding for stylization, of shape (B, T, time_embed_dim).
        - src_mask (torch.Tensor): Mask for the input data, of shape (B, T).
        - motion_length (torch.Tensor): Tensor of shape (B,) representing the motion length.
        - num_intervals (int): Number of intervals for processing the motion data.
        - text_cond (torch.Tensor, optional): Conditioning mask for text data, of shape (B, 1).
        - text_word_out (torch.Tensor, optional): Word features for text, of shape (B, M, latent_dim).
        - music_cond (torch.Tensor, optional): Conditioning mask for music data, of shape (B, 1).
        - music_word_out (torch.Tensor, optional): Word features for music, of shape (B, M, latent_dim).
        - speech_cond (torch.Tensor, optional): Conditioning mask for speech data, of shape (B, 1).
        - speech_word_out (torch.Tensor, optional): Word features for speech, of shape (B, M, latent_dim).
        - video_cond (torch.Tensor, optional): Conditioning mask for video data, of shape (B, 1).
        - video_word_out (torch.Tensor, optional): Word features for video, of shape (B, M, latent_dim).
        - duration (torch.Tensor, optional): Duration of each motion sequence, of shape (B,).

    Outputs:
        - y (torch.Tensor): The final attended output, with the same shape as input x (B, T, D).
    """
    def __init__(self,
                 latent_dim,
                 num_heads,
                 num_experts,
                 topk,
                 gate_type,
                 gate_noise,
                 ffn_dim,
                 time_embed_dim,
                 max_seq_len,
                 dropout,
                 num_datasets,
                 has_text=False,
                 has_music=False,
                 has_speech=False,
                 has_video=False,
                 norm="LayerNorm"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Choose normalization type
        if norm == "LayerNorm":
            Norm = nn.LayerNorm

        # Parameters for time-related functions
        self.sigma = nn.Parameter(torch.Tensor([100]))  # Sigma for softmax-based time weighting
        self.time = torch.arange(max_seq_len)

        # Normalization for motion features
        self.norm = Norm(latent_dim * 10)

        # MoE for motion data
        self.motion_moe = MOE(num_experts, topk, latent_dim, latent_dim * 4,
                              5 * latent_dim, num_heads, max_seq_len,
                              gate_type, gate_noise)
        self.motion_moe_dropout = nn.Dropout(p=dropout)  # Dropout for motion MoE
        self.key_motion_scale = nn.Parameter(torch.Tensor([1.0]))

        # Default keys and values
        self.num_datasets = num_datasets
        self.key_dataset = nn.Parameter(torch.randn(num_datasets, 48, 10, latent_dim))
        self.key_dataset_scale = nn.Parameter(torch.Tensor([1.0]))
        self.value_dataset = nn.Parameter(torch.randn(num_datasets, 48, 10, latent_dim))
        
        self.key_rotation = nn.Parameter(torch.randn(3, 16, 10, latent_dim))
        self.value_rotation = nn.Parameter(torch.randn(3, 16, 10, latent_dim))
        self.key_rotation_scale = nn.Parameter(torch.Tensor([1.0]))

        # Conditional MoE layers for each modality (if applicable)
        self.has_text = has_text
        self.has_music = has_music
        self.has_speech = has_speech
        self.has_video = has_video
        
        if has_text or has_music or has_speech or has_video:
            self.cond_moe = MOE(num_experts, topk, latent_dim, latent_dim * 4,
                                2 * latent_dim, num_heads, max_seq_len,
                                gate_type, gate_noise, embedding=False)
        if has_text:
            self.norm_text = Norm(latent_dim * 10)
            self.key_text_scale = nn.Parameter(torch.Tensor([1.0]))
        if has_music:
            self.norm_music = Norm(latent_dim * 10)
            self.key_music_scale = nn.Parameter(torch.Tensor([1.0]))
        if has_speech:
            self.norm_speech = Norm(latent_dim * 10)
            self.key_speech_scale = nn.Parameter(torch.Tensor([1.0]))
        if has_video:
            self.norm_video = Norm(latent_dim * 10)
            self.key_video_scale = nn.Parameter(torch.Tensor([1.0]))

        # Template functions for Taylor expansion (state, velocity, acceleration, jerk)
        self.template_s = get_ffn(latent_dim, ffn_dim)
        self.template_v = get_ffn(latent_dim, ffn_dim)
        self.template_a = get_ffn(latent_dim, ffn_dim)
        self.template_j = get_ffn(latent_dim, ffn_dim)
        self.template_t = nn.Sequential(nn.Linear(latent_dim, ffn_dim),
                                        nn.GELU(), nn.Linear(ffn_dim, 1))
        self.t_sigma = nn.Parameter(torch.Tensor([1]))  # Sigma for Taylor expansion

        # Final projection with stylization block
        self.proj_out = StylizationBlock(latent_dim * num_heads,
                                         time_embed_dim, dropout)

    def forward(self,
                x: torch.Tensor,
                emb: torch.Tensor,
                src_mask: torch.Tensor,
                motion_length: torch.Tensor,
                num_intervals: int,
                text_cond: Optional[torch.Tensor] = None,
                text_word_out: Optional[torch.Tensor] = None,
                music_cond: Optional[torch.Tensor] = None,
                music_word_out: Optional[torch.Tensor] = None,
                speech_cond: Optional[torch.Tensor] = None,
                speech_word_out: Optional[torch.Tensor] = None,
                video_cond: Optional[torch.Tensor] = None,
                video_word_out: Optional[torch.Tensor] = None,
                duration: Optional[torch.Tensor] = None,
                dataset_idx: Optional[torch.Tensor] = None,
                rotation_idx: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass for the ArtAttention module, handling multi-modal inputs.

        Args:
            x (torch.Tensor): Input motion data of shape (B, T, D).
            emb (torch.Tensor): Time embedding for stylization.
            src_mask (torch.Tensor): Source mask for the input data.
            motion_length (torch.Tensor): Length of the motion data.
            num_intervals (int): Number of intervals for motion data.
            text_cond (torch.Tensor, optional): Conditioning mask for text data.
            text_word_out (torch.Tensor, optional): Text word output features.
            music_cond (torch.Tensor, optional): Conditioning mask for music data.
            music_word_out (torch.Tensor, optional): Music word output features.
            speech_cond (torch.Tensor, optional): Conditioning mask for speech data.
            speech_word_out (torch.Tensor, optional): Speech word output features.
            video_cond (torch.Tensor, optional): Conditioning mask for video data.
            video_word_out (torch.Tensor, optional): Video word output features.
            duration (torch.Tensor, optional): Duration of each motion sequence.

        Returns:
            y (torch.Tensor): The attended multi-modal motion features.
        """
        
        B, T, D = x.shape  # Batch size (B), Time steps (T), Feature dimension (D)
        H = self.num_heads
        L = self.latent_dim

        # Pass motion data through MoE
        motion_feat = self.motion_moe(self.norm(x).reshape(B, T, H, -1))
        motion_feat = self.motion_moe_dropout(motion_feat)

        # Reshape motion data for attention
        x = x.reshape(B, T, H, -1)

        # Apply source mask and compute attention over motion features
        src_mask = src_mask.view(B, T, H, 1)
        body_value = motion_feat[:, :, :, :L] * src_mask
        body_key = motion_feat[:, :, :, L: 2 * L] + (1 - src_mask) * -1000000
        body_key = F.softmax(body_key, dim=2)
        body_query = F.softmax(motion_feat[:, :, :, 2 * L: 3 * L], dim=-1)
        body_attention = torch.einsum('bnhd,bnhl->bndl', body_key, body_value)
        body_feat = torch.einsum('bndl,bnhd->bnhl', body_attention, body_query)
        body_feat = body_feat.reshape(B, T, D)

        # Key and value attention for motion
        key_motion = motion_feat[:, :, :, 3 * L: 4 * L].contiguous()
        key_motion = key_motion.view(B, T, H, -1)
        key_motion = (key_motion + (1 - src_mask) * -1000000) / self.key_motion_scale

        value_motion = motion_feat[:, :, :, 4 * L:].contiguous() * src_mask
        value_motion = value_motion.view(B, T, H, -1)

        # Process multi-modal conditioning (text, music, speech, video)
        key_dataset = self.key_dataset.index_select(0, dataset_idx) / self.key_dataset_scale
        value_dataset = self.value_dataset.index_select(0, dataset_idx)
        key_rotation = self.key_rotation.index_select(0, rotation_idx) / self.key_rotation_scale
        value_rotation = self.value_rotation.index_select(0, rotation_idx)
        key = torch.cat((key_motion, key_dataset, key_rotation), dim=1)
        value = torch.cat((value_motion, value_dataset, value_rotation), dim=1)
        N = 64
        if self.has_text and text_word_out is not None and torch.sum(text_cond) > 0:
            M = text_word_out.shape[1]
            text_feat = self.norm_text(text_word_out).reshape(B, M, H, -1)
            text_feat = self.cond_moe(text_feat)
            key_text = text_feat[:, :, :, :L].contiguous()
            key_text = key_text + (1 - text_cond.view(B, 1, 1, 1)) * -1000000
            key_text = key_text / self.key_text_scale
            key = torch.cat((key, key_text), dim=1)
            value_text = text_feat[:, :, :, L:].contiguous()
            value_text = value_text * text_cond.view(B, 1, 1, 1)
            value = torch.cat((value, value_text), dim=1)
            N += M

        if self.has_music and music_word_out is not None and torch.sum(music_cond) > 0:
            M = music_word_out.shape[1]
            music_feat = self.norm_music(music_word_out).reshape(B, M, H, -1)
            music_feat = self.cond_moe(music_feat)
            key_music = music_feat[:, :, :, :L].contiguous()
            key_music = key_music + (1 - music_cond.view(B, 1, 1, 1)) * -1000000
            key_music = key_music / self.key_music_scale
            key = torch.cat((key, key_music), dim=1)
            value_music = music_feat[:, :, :, L:].contiguous()
            value_music = value_music * music_cond.view(B, 1, 1, 1)
            value = torch.cat((value, value_music), dim=1)
            N += M

        if self.has_speech and speech_word_out is not None and torch.sum(speech_cond) > 0:
            M = speech_word_out.shape[1]
            speech_feat = self.norm_speech(speech_word_out).reshape(B, M, H, -1)
            speech_feat = self.cond_moe(speech_feat)
            key_speech = speech_feat[:, :, :, :L].contiguous()
            key_speech = key_speech + (1 - speech_cond.view(B, 1, 1, 1)) * -1000000
            key_speech = key_speech / self.key_speech_scale
            key = torch.cat((key, key_speech), dim=1)
            value_speech = speech_feat[:, :, :, L:].contiguous()
            value_speech = value_speech * speech_cond.view(B, 1, 1, 1)
            value = torch.cat((value, value_speech), dim=1)
            N += M

        if self.has_video and video_word_out is not None and torch.sum(video_cond) > 0:
            M = video_word_out.shape[1]
            video_feat = self.norm_video(video_word_out).reshape(B, M, H, -1)
            video_feat = self.cond_moe(video_feat)
            key_video = video_feat[:, :, :, :L].contiguous()
            key_video = key_video + (1 - video_cond.view(B, 1, 1, 1)) * -1000000 
            key_video = key_video + (1 - src_mask) * -1000000
            key_video = key_video / self.key_video_scale
            key = torch.cat((key, key_video), dim=1)
            value_video = video_feat[:, :, :, L:].contiguous()
            value_video = value_video * video_cond.view(B, 1, 1, 1) * src_mask
            value= torch.cat((value, value_video), dim=1)
            N += M

        key = F.softmax(key, dim=1)
        # B, H, d, l
        template = torch.einsum('bnhd,bnhl->bhdl', key, value)
        template_t_feat = self.template_t(template)
        template_t = torch.sigmoid(template_t_feat / self.t_sigma)
        template_t = template_t * motion_length.view(B, 1, 1, 1)
        template_t = template_t * duration.view(B, 1, 1, 1)
        org_t = self.time[:T].type_as(x)

        # Handle time-based calculations
        NI = num_intervals
        t = org_t.clone().view(1, 1, -1, 1, 1).repeat(B // NI, NI, 1, 1, 1)
        t = t * duration.view(B // NI, NI, 1, 1, 1)
        template_t = template_t.view(-1, NI, H, L)
        motion_length = motion_length.view(-1, NI)
        for b_ix in range(B // NI):
            sum_frames = 0
            for i in range(NI):
                t[b_ix, i] += sum_frames * float(duration[b_ix])
                template_t[b_ix, i] += sum_frames * float(duration[b_ix])
                sum_frames += motion_length[b_ix, i]
        template_t = template_t.permute(0, 2, 1, 3)
        template_t = template_t.unsqueeze(1).repeat(1, NI, 1, 1, 1)
        template_t = template_t.reshape(B, 1, H, -1)
        time_delta = t.view(B, -1, 1, 1) - template_t
        time_sqr = time_delta * time_delta
        time_coef = F.softmax(-time_sqr, dim=-1)

        template = template.view(-1, NI, H, L, L)
        template = template.permute(0, 2, 1, 3, 4).unsqueeze(1)
        template = template.repeat(1, NI, 1, 1, 1, 1)
        template = template.reshape(B, H, -1, L)

        # Taylor expansion for motion
        template_s = template + self.template_s(template)  # state
        template_v = template + self.template_v(template)  # velocity
        template_a = template + self.template_a(template)  # acceleration
        template_j = template + self.template_j(template)  # jerk
        template_t = template_t.view(B, H, -1, 1)
        template_a0 = template_s - template_v * template_t + \
            template_a * template_t * template_t - \
            template_j * template_t * template_t * template_t
        template_a1 = template_v - 2 * template_a * template_t + \
            3 * template_j * template_t * template_t
        template_a2 = template_a - 3 * template_j * template_t
        template_a3 = template_j
        a0 = torch.einsum('bnhd,bhdl->bnhl', time_coef,
                          template_a0).reshape(B, T, D)
        a1 = torch.einsum('bnhd,bhdl->bnhl', time_coef,
                          template_a1).reshape(B, T, D)
        a2 = torch.einsum('bnhd,bhdl->bnhl', time_coef,
                          template_a2).reshape(B, T, D)
        a3 = torch.einsum('bnhd,bhdl->bnhl', time_coef,
                          template_a3).reshape(B, T, D)
        t = t.view(B, -1, 1)
        y_t = a0 + a1 * t + a2 * t * t + a3 * t * t * t
        y_s = body_feat
        y = x.reshape(B, T, D) + self.proj_out(y_s + y_t, emb)

        if self.training:
            # Add auxiliary losses during training
            self.aux_loss = self.motion_moe.aux_loss
            if self.has_text or self.has_music or self.has_speech or self.has_video:
                if hasattr(self.cond_moe, 'aux_loss') and self.cond_moe.aux_loss is not None:
                    self.aux_loss += self.cond_moe.aux_loss
                    self.cond_moe.aux_loss = None
            mu = template_t_feat.squeeze(-1).mean(dim=-1)
            logvar = torch.log(template_t_feat.squeeze(-1).std(dim=-1))
            logvar[logvar > 1000000] = 0
            logvar[logvar < -1000000] = 0
            self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return y

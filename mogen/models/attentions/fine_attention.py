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


class MOE(nn.Module):
    """
    Mixture of Experts (MoE) layer implementation using the Tutel MoE library.

    Args:
        num_experts (int): Number of experts.
        topk (int): Number of top experts to route tokens to.
        input_dim (int): Input dimension of the MoE layer.
        ffn_dim (int): Feed-forward network dimension for each expert.
        output_dim (int): Output dimension of the MoE layer.
        num_heads (int): Number of attention heads.
        max_seq_len (int): Maximum sequence length.
        gate_type (str): Type of gating mechanism (e.g., 'top_k').
        gate_noise (float): Noise factor for the gating mechanism.
    """

    def __init__(self, num_experts: int, topk: int, input_dim: int, ffn_dim: int, output_dim: int, 
                 num_heads: int, max_seq_len: int, gate_type: str, gate_noise: float):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        
        try:
            data_group = net.create_groups_from_world(group_count=1).data_group
        except Exception:
            data_group = None

        self.model = tutel_moe.moe_layer(
            gate_type={
                'type': gate_type,
                'k': topk,
                'fp32_gate': True,
                'gate_noise': gate_noise,
                'capacity_factor': 1.5
            },
            experts={
                'type': 'ffn',
                'count_per_node': num_experts,
                'hidden_size_per_expert': ffn_dim,
                'activation_fn': lambda x: F.gelu(x)
            },
            model_dim=input_dim,
            batch_prioritized_routing=True,
            is_gshard_loss=False,
            group=data_group
        )
        self.embedding = nn.Parameter(torch.randn(1, max_seq_len, num_heads, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MOE layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H, D].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, H, D].
        """
        B, T, H, D = x.shape
        x = x + self.embedding[:, :T, :, :]
        x = x.reshape(-1, D)
        y = self.proj(self.activation(self.model(x)))
        self.aux_loss = self.model.l_aux
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
class SAMI(nn.Module):
    """
    SAMI: Self-Attention-based MoE Integration model for motion generation.

    Args:
        latent_dim (int): Dimension of the latent space for motion input.
        text_latent_dim (int): Dimension of the latent space for text input.
        num_heads (int): Number of motion attention heads.
        num_text_heads (int): Number of text attention heads.
        num_experts (int): Number of experts for MoE.
        topk (int): Number of top experts to route tokens to.
        gate_type (str): Type of gating mechanism.
        gate_noise (float): Noise factor for the gating mechanism.
        ffn_dim (int): Dimension of the feed-forward network.
        time_embed_dim (int): Dimension of the time embedding.
        max_seq_len (int): Maximum sequence length for motion data.
        max_text_seq_len (int): Maximum sequence length for text data.
        dropout (float): Dropout probability.
        norm (str): Type of normalization ('LayerNorm').
        att_balance (bool): Whether to balance attention weights between motion and text.
        fine_mode (bool): Whether to use fine-grained features.
        mask_cond (float): Masking condition for fine-tuning.
    """

    def __init__(self,
                 latent_dim: int,
                 text_latent_dim: int,
                 num_heads: int,
                 num_text_heads: int,
                 num_experts: int,
                 topk: int,
                 gate_type: str,
                 gate_noise: float,
                 ffn_dim: int,
                 time_embed_dim: int,
                 max_seq_len: int,
                 max_text_seq_len: int,
                 dropout: float,
                 norm: str = "LayerNorm",
                 att_balance: bool = False,
                 fine_mode: bool = False,
                 mask_cond: float = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_text_heads = num_text_heads
        self.max_seq_len = max_seq_len

        # Normalization
        Norm = nn.LayerNorm
        self.norm = Norm(latent_dim)
        self.text_norm = Norm(text_latent_dim)

        # MoE Layers for motion and text
        self.sigma = nn.Parameter(torch.Tensor([100]))
        self.time = torch.arange(max_seq_len) / max_seq_len
        self.text_moe = MOE(num_experts, topk, text_latent_dim, text_latent_dim * 4, 2 * latent_dim,
                            num_text_heads, max_text_seq_len, gate_type, gate_noise)
        self.motion_moe = MOE(num_experts, topk, latent_dim, latent_dim * 4, 3 * latent_dim,
                              num_heads, max_seq_len, gate_type, gate_noise)

        # Key-motion and attention blocks
        self.key_motion = nn.Parameter(torch.randn(max_seq_len, latent_dim))
        self.body_weight = nn.Parameter(torch.randn(num_heads, num_heads))

        # Feedforward networks for state, velocity, acceleration, and jerk
        self.template_s = get_ffn(latent_dim, ffn_dim)
        self.template_v = get_ffn(latent_dim, ffn_dim)
        self.template_a = get_ffn(latent_dim, ffn_dim)
        self.template_j = get_ffn(latent_dim, ffn_dim)

        # Time embedding block
        self.template_t = nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, 1))
        self.t_sigma = nn.Parameter(torch.Tensor([1]))

        # Output projection
        self.proj_out = StylizationBlock(latent_dim * num_heads, time_embed_dim, dropout)
        self.att_balance = att_balance
        if self.att_balance:
            self.motion_coef = nn.Parameter(torch.Tensor([0]))
            self.text_coef = nn.Parameter(torch.Tensor([0]))

        self.fine_mode = fine_mode
        self.mask_cond = mask_cond

    def forward(self, x: torch.Tensor, xf: torch.Tensor, emb: torch.Tensor, src_mask: torch.Tensor,
                cond_type: torch.Tensor, motion_length: torch.Tensor, num_intervals: int, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass of SAMI.

        Args:
            x (torch.Tensor): Motion input tensor of shape [B, T, D].
            xf (torch.Tensor): Text input tensor of shape [B, N, P].
            emb (torch.Tensor): Time embedding tensor.
            src_mask (torch.Tensor): Source mask tensor of shape [B, T].
            cond_type (torch.Tensor): Conditioning type tensor of shape [B, ?].
            motion_length (torch.Tensor): Motion length tensor.
            num_intervals (int): Number of intervals for the motion.

        Returns:
            torch.Tensor: Output tensor after motion and text MoE integration.
        """
        B, T, D = x.shape
        N = xf.shape[1] + x.shape[1]
        H = self.num_heads
        L = self.latent_dim
        
        x = x.reshape(B, T, H, -1)
        if self.fine_mode:
            text_feat = xf.reshape(B, self.num_text_heads, xf.shape[1], xf.shape[2]).permute(0, 2, 1, 3)
        else:
            text_feat = xf.reshape(B, xf.shape[1], self.num_text_heads, -1)
        
        # MoE Layers for text and motion features
        text_feat = self.text_moe(self.text_norm(text_feat))
        motion_feat = self.motion_moe(self.norm(x))

        # Weighted combination of motion features
        body_weight = F.softmax(self.body_weight, dim=1)
        body_value = motion_feat[:, :, :, :L]
        body_feat = torch.einsum('hl,bnld->bnhd', body_weight, body_value)
        body_feat = body_feat.reshape(B, T, D)

        # Apply the source mask and combine key-text and key-motion
        src_mask = src_mask.view(B, T, 1, 1)
        key_text = text_feat[:, :, :, :L].contiguous()
        
        # Handle conditional types and masks
        if self.fine_mode:
            text_cond_type = torch.cat((cond_type[:, :7, :] % 10 > self.mask_cond, cond_type[:, 7:8, :] % 10 > 0), 1).float().unsqueeze(-1)
            text_cond_type = text_cond_type.permute(0, 2, 1, 3)
            text_cond_type = text_cond_type.repeat(1, key_text.shape[1], 1, 1)
        else:
            text_cond_type = (cond_type % 10 > 0).float().unsqueeze(-1)
        
        key_text = key_text + (1 - text_cond_type) * -1000000
        if self.num_text_heads == 1:
            key_text = key_text.repeat(1, 1, H, 1)

        key_motion = motion_feat[:, :, :, L:2 * L].contiguous()
        key_motion = key_motion + (1 - src_mask) * -1000000

        # Attention balance between motion and text
        if self.att_balance:
            motion_coef = torch.sigmoid(self.motion_coef)
            text_coef = torch.sigmoid(self.text_coef)
            key_motion = F.softmax(key_motion, dim=1) * motion_coef
            key_text = F.softmax(key_text, dim=1) * text_coef
            sum_coef = motion_coef.repeat(B) + text_coef.repeat(B) * text_cond_type.view(B)
            sum_coef = sum_coef.view(B, 1, 1, 1)
            key_motion = key_motion / sum_coef
            key_text = key_text / sum_coef
            key = torch.cat((key_text, key_motion), dim=1)
        else:
            key = torch.cat((key_text, key_motion), dim=1)
            key = F.softmax(key.view(B, N, H, -1), dim=1)

        # Value combination for text and motion
        value_text = text_feat[:, :, :, L:].contiguous() * text_cond_type
        if self.num_text_heads == 1:
            value_text = value_text.repeat(1, 1, H, 1)
        value_motion = motion_feat[:, :, :, 2 * L:].contiguous() * src_mask
        value = torch.cat((value_text, value_motion), dim=1).view(B, N, H, -1)

        # Calculate the attention-weighted template
        template = torch.einsum('bnhd,bnhl->bhdl', key, value)
        template_t_feat = self.template_t(template)
        template_t = torch.sigmoid(template_t_feat / self.t_sigma)
        template_t = template_t * motion_length.view(B, 1, 1, 1)
        template_t = template_t / self.max_seq_len

        org_t = self.time[:T].type_as(x)

        # Handle time intervals for the motion
        NI = num_intervals
        t = org_t.clone().view(1, 1, -1, 1, 1).repeat(B // NI, NI, 1, 1, 1)
        template_t = template_t.view(-1, NI, H, L)
        motion_length = motion_length.view(-1, NI)
        for b_ix in range(B // NI):
            sum_frames = 0
            for i in range(NI):
                t[b_ix, i] += sum_frames / self.max_seq_len
                template_t[b_ix, i] += sum_frames / self.max_seq_len
                sum_frames += motion_length[b_ix, i]
        template_t = template_t.permute(0, 2, 1, 3).unsqueeze(1).repeat(1, NI, 1, 1, 1)
        template_t = template_t.reshape(B, 1, H, -1)

        time_delta = t.view(B, -1, 1, 1) - template_t
        time_delta = time_delta * self.max_seq_len
        time_sqr = time_delta * time_delta
        time_coef = F.softmax(-time_sqr / self.sigma, dim=-1)

        # Reshape and repeat templates for Taylor expansion
        template = template.view(-1, NI, H, L, L)
        template = template.permute(0, 2, 1, 3, 4).unsqueeze(1)
        template = template.repeat(1, NI, 1, 1, 1, 1)
        template = template.reshape(B, H, -1, L)

        # Taylor expansion for state (s), velocity (v), acceleration (a), jerk (j)
        template_s = self.template_s(template)
        template_v = self.template_v(template)
        template_a = self.template_a(template)
        template_j = self.template_j(template)

        template_t = template_t.view(B, H, -1, 1)
        template_a0 = template_s - template_v * template_t + template_a * template_t * template_t - template_j * template_t * template_t * template_t
        template_a1 = template_v - 2 * template_a * template_t + 3 * template_j * template_t * template_t
        template_a2 = template_a - 3 * template_j * template_t
        template_a3 = template_j

        # Calculate the final time-dependent output using the Taylor expansion
        a0 = torch.einsum('bnhd,bhdl->bnhl', time_coef, template_a0).reshape(B, T, D)
        a1 = torch.einsum('bnhd,bhdl->bnhl', time_coef, template_a1).reshape(B, T, D)
        a2 = torch.einsum('bnhd,bhdl->bnhl', time_coef, template_a2).reshape(B, T, D)
        a3 = torch.einsum('bnhd,bhdl->bnhl', time_coef, template_a3).reshape(B, T, D)

        t = t.view(B, -1, 1)
        y_t = a0 + a1 * t + a2 * t * t + a3 * t * t * t

        # Combine with body features and output the final result
        y_s = body_feat
        y = x.reshape(B, T, D) + self.proj_out(y_s + y_t, emb)

        if self.training:
            self.aux_loss = self.text_moe.aux_loss + self.motion_moe.aux_loss
            mu = template_t_feat.squeeze(-1).mean(dim=-1)
            logvar = torch.log(template_t_feat.squeeze(-1).std(dim=-1))
            self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return y


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock


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


@ATTENTIONS.register_module()
class SemanticsModulatedAttention(nn.Module):
    """
    Semantics-modulated attention module that integrates motion, text, and retrieval features into attention computation.

    Args:
        latent_dim (int): Dimensionality of the latent (motion) features.
        text_latent_dim (int): Dimensionality of the text features.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        time_embed_dim (int): Dimensionality of time embeddings.
    """

    def __init__(self, latent_dim: int, text_latent_dim: int, num_heads: int, dropout: float, time_embed_dim: int):
        super().__init__()
        self.num_heads = num_heads

        # Layer Normalization for motion and text features
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)

        # Linear projections for queries, keys, and values
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_text = nn.Linear(text_latent_dim, latent_dim)
        self.value_text = nn.Linear(text_latent_dim, latent_dim)
        self.key_motion = nn.Linear(latent_dim, latent_dim)
        self.value_motion = nn.Linear(latent_dim, latent_dim)

        # Retrieval feature processing (motion and text)
        self.retr_norm1 = nn.LayerNorm(2 * latent_dim)
        self.retr_norm2 = nn.LayerNorm(latent_dim)
        self.key_retr = nn.Linear(2 * latent_dim, latent_dim)
        self.value_retr = zero_module(nn.Linear(latent_dim, latent_dim))

        # Dropout and output projection
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x: torch.Tensor, xf: torch.Tensor, emb: torch.Tensor, src_mask: torch.Tensor,
                cond_type: torch.Tensor, re_dict: dict) -> torch.Tensor:
        """
        Forward pass of SemanticsModulatedAttention.

        Args:
            x (torch.Tensor): Motion features of shape (B, T, D).
            xf (torch.Tensor): Text features of shape (B, N, L).
            emb (torch.Tensor): Time embedding.
            src_mask (torch.Tensor): Source mask for the input motion features.
            cond_type (torch.Tensor): Condition type tensor.
            re_dict (dict): Dictionary containing retrieval motion, text, and mask data.

        Returns:
            torch.Tensor: Output tensor after attention modulation, shape (B, T, D).
        """
        B, T, D = x.shape
        re_motion = re_dict['re_motion']
        re_text = re_dict['re_text']
        re_mask = re_dict['re_mask'].reshape(B, -1, 1)
        N = xf.shape[1] + x.shape[1] + re_motion.shape[1] * re_motion.shape[2]  # Total number of attention keys

        H = self.num_heads
        query = self.query(self.norm(x))  # Query from motion features

        # Key and Value from text and retrieval features
        text_cond_type = (cond_type % 10 > 0).float()
        retr_cond_type = (cond_type // 10 > 0).float()
        re_text = re_text.repeat(1, 1, re_motion.shape[2], 1)
        re_feat_key = torch.cat((re_motion, re_text), dim=-1).reshape(B, -1, 2 * D)

        # Calculate keys for text, retrieval, and motion
        key_text = self.key_text(self.text_norm(xf)) + (1 - text_cond_type) * -1000000
        key_retr = self.key_retr(self.retr_norm1(re_feat_key)) + (1 - retr_cond_type) * -1000000 + (1 - re_mask) * -1000000
        key_motion = self.key_motion(self.norm(x)) + (1 - src_mask) * -1000000

        key = torch.cat((key_text, key_retr, key_motion), dim=1)  # Concatenate all keys

        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)

        # Value computation from text, retrieval, and motion features
        re_feat_value = re_motion.reshape(B, -1, D)
        value_text = self.value_text(self.text_norm(xf)) * text_cond_type
        value_retr = self.value_retr(self.retr_norm2(re_feat_value)) * retr_cond_type * re_mask
        value_motion = self.value_motion(self.norm(x)) * src_mask
        value = torch.cat((value_text, value_retr, value_motion), dim=1).view(B, N, H, -1)

        # Attention computation and output projection
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y



@ATTENTIONS.register_module()
class DualSemanticsModulatedAttention(nn.Module):
    """
    Dual semantics-modulated attention module that handles two streams of motion features and integrates
    them with text and retrieval features.

    Args:
        latent_dim (int): Dimensionality of the latent (motion) features.
        text_latent_dim (int): Dimensionality of the text features.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        time_embed_dim (int): Dimensionality of time embeddings.
    """

    def __init__(self, latent_dim: int, text_latent_dim: int, num_heads: int, dropout: float, time_embed_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        # Layer Normalization for motion and text features
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)

        # Linear projections for queries, keys, and values
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_text = nn.Linear(text_latent_dim, latent_dim)
        self.value_text = nn.Linear(text_latent_dim, latent_dim)
        self.key_motion = nn.Linear(latent_dim, latent_dim)
        self.value_motion = nn.Linear(latent_dim, latent_dim)
        self.key_inter = nn.Linear(latent_dim, latent_dim)
        self.value_inter = nn.Linear(latent_dim, latent_dim)

        # Retrieval feature processing (motion and text)
        self.retr_norm1 = nn.LayerNorm(2 * latent_dim)
        self.retr_norm2 = nn.LayerNorm(latent_dim)
        self.key_retr = nn.Linear(2 * latent_dim, latent_dim)
        self.value_retr = zero_module(nn.Linear(latent_dim, latent_dim))

        # Dropout and output projection
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x: torch.Tensor, xf: torch.Tensor, emb: torch.Tensor, src_mask: torch.Tensor,
                cond_type: torch.Tensor, re_dict: dict) -> torch.Tensor:
        """
        Forward pass of DualSemanticsModulatedAttention.

        Args:
            x (torch.Tensor): Motion features of shape (B, T, 2*D).
            xf (torch.Tensor): Text features of shape (B, N, L).
            emb (torch.Tensor): Time embedding.
            src_mask (torch.Tensor): Source mask for the input motion features.
            cond_type (torch.Tensor): Condition type tensor.
            re_dict (dict): Dictionary containing retrieval motion, text, and mask data.

        Returns:
            torch.Tensor: Output tensor after dual attention modulation, shape (B, T, 2*D).
        """
        x1 = x[:, :, :self.latent_dim].contiguous()
        x2 = x[:, :, self.latent_dim:].contiguous()
        B, T, D = x1.shape
        re_motion = re_dict['re_motion']
        re_text = re_dict['re_text']
        re_mask = re_dict['re_mask'].reshape(B, -1, 1)
        N = xf.shape[1] + x.shape[1] * 2 + re_motion.shape[1] * re_motion.shape[2]

        H = self.num_heads

        # Query computation for both streams
        query1 = self.query(self.norm(x1))
        query2 = self.query(self.norm(x2))

        # Retrieval key/value feature preparation
        text_cond_type = (cond_type % 10 > 0).float()
        retr_cond_type = (cond_type // 10 > 0).float()
        re_text = re_text.repeat(1, 1, re_motion.shape[2], 1)
        re_feat_key = torch.cat((re_motion, re_text), dim=-1)
        re_feat_key = re_feat_key.reshape(B, -1, 2 * D)

        # Keys for text, retrieval, and motion
        key_text = self.key_text(self.text_norm(xf)) + (1 - text_cond_type) * -1000000
        key_retr = self.key_retr(self.retr_norm1(re_feat_key)) + (1 - retr_cond_type) * -1000000 + (1 - re_mask) * -1000000
        key_motion1 = self.key_motion(self.norm(x1)) + (1 - src_mask) * -1000000
        key_motion2 = self.key_motion(self.norm(x2)) + (1 - src_mask) * -1000000

        # Cross-attention keys for inter-stream communication
        key_inter1 = self.key_inter(self.norm(x2)) + (1 - src_mask) * -1000000
        key_inter2 = self.key_inter(self.norm(x1)) + (1 - src_mask) * -1000000

        # Concatenate all keys for the two streams
        key1 = torch.cat((key_text, key_retr, key_motion1, key_inter1), dim=1)
        key2 = torch.cat((key_text, key_retr, key_motion2, key_inter2), dim=1)

        # Softmax over queries and keys
        query1 = F.softmax(query1.view(B, T, H, -1), dim=-1)
        query2 = F.softmax(query2.view(B, T, H, -1), dim=-1)
        key1 = F.softmax(key1.view(B, N, H, -1), dim=1)
        key2 = F.softmax(key2.view(B, N, H, -1), dim=1)

        # Value computation for text, retrieval, and motion
        re_feat_value = re_motion.reshape(B, -1, D)
        value_text = self.value_text(self.text_norm(xf)) * text_cond_type
        value_retr = self.value_retr(self.retr_norm2(re_feat_value)) * retr_cond_type * re_mask
        value_motion1 = self.value_motion(self.norm(x1)) * src_mask
        value_motion2 = self.value_motion(self.norm(x2)) * src_mask

        # Inter-stream value exchange
        value_inter1 = self.value_inter(self.norm(x2)) * src_mask
        value_inter2 = self.value_inter(self.norm(x1)) * src_mask

        # Concatenate values for both streams
        value1 = torch.cat((value_text, value_retr, value_motion1, value_inter1), dim=1).view(B, N, H, -1)
        value2 = torch.cat((value_text, value_retr, value_motion2, value_inter2), dim=1).view(B, N, H, -1)

        # Compute attention outputs for both streams
        attention1 = torch.einsum('bnhd,bnhl->bhdl', key1, value1)
        attention2 = torch.einsum('bnhd,bnhl->bhdl', key2, value2)

        # Apply attention to queries and compute final output
        y1 = torch.einsum('bnhd,bhdl->bnhl', query1, attention1).reshape(B, T, D)
        y2 = torch.einsum('bnhd,bhdl->bnhl', query2, attention2).reshape(B, T, D)

        # Combine both streams and apply output projection
        y1 = x1 + self.proj_out(y1, emb)
        y2 = x2 + self.proj_out(y2, emb)
        y = torch.cat((y1, y2), dim=-1)

        return y

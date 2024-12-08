import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock


@ATTENTIONS.register_module()
class EfficientSelfAttention(nn.Module):
    """
    Efficient Self-Attention mechanism for motion generation tasks.

    Args:
        latent_dim (int): Dimension of the latent space.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        time_embed_dim (Optional[int]): Dimension of the time embedding (optional).
    """

    def __init__(self,
                 latent_dim: int,
                 num_heads: int,
                 dropout: float,
                 time_embed_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self,
                x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                emb: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass of Efficient Self-Attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D].
            src_mask (Optional[torch.Tensor]): Source mask of shape [B, T] (optional).
            emb (Optional[torch.Tensor]): Time embedding tensor of shape [B, D] (optional).

        Returns:
            torch.Tensor: Output of the self-attention module.
        """
        B, T, D = x.shape
        H = self.num_heads

        query = self.query(self.norm(x))

        if src_mask is None:
            key = self.key(self.norm(x))
        else:
            key = self.key(self.norm(x)) + (1 - src_mask) * -1000000

        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)

        if src_mask is None:
            value = self.value(self.norm(x)).view(B, T, H, -1)
        else:
            value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)

        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)

        if self.time_embed_dim is None:
            y = x + y
        else:
            y = x + self.proj_out(y, emb)

        return y


@ATTENTIONS.register_module()
class EfficientCrossAttention(nn.Module):
    """
    Efficient Cross-Attention mechanism, attending to text and motion inputs.

    Args:
        latent_dim (int): Dimension of the latent space for motion input.
        text_latent_dim (int): Dimension of the latent space for text input.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        time_embed_dim (int): Dimension of the time embedding.
    """

    def __init__(self,
                 latent_dim: int,
                 text_latent_dim: int,
                 num_heads: int,
                 dropout: float,
                 time_embed_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        if time_embed_dim is not None:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
        else:
            self.proj_out = None

    def forward(self,
                x: torch.Tensor,
                xf: torch.Tensor,
                emb: Optional[torch.Tensor] = None,
                cond_type: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass of Efficient Cross-Attention.

        Args:
            x (torch.Tensor): Input motion tensor of shape [B, T, D].
            xf (torch.Tensor): Input text tensor of shape [B, N, L].
            emb (torch.Tensor): Time embedding tensor of shape [B, D].
            cond_type (Optional[torch.Tensor]): Conditioning type tensor (optional).

        Returns:
            torch.Tensor: Output of the cross-attention module.
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_heads

        query = self.query(self.norm(x))

        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)

        if cond_type is None:
            key = F.softmax(key.view(B, N, H, -1), dim=1)
            value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        else:
            text_cond_type = ((cond_type % 10) > 0).float().view(B, 1, 1)
            text_cond_type = text_cond_type.repeat(1, xf.shape[1], 1)
            key = key + (1 - text_cond_type) * -1000000
            key = F.softmax(key.view(B, N, H, -1), dim=1)
            value = self.value(self.text_norm(xf) * text_cond_type).view(B, N, H, -1)

        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        if self.proj_out is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return y


@ATTENTIONS.register_module()
class EfficientMixedAttention(nn.Module):
    """
    Efficient Mixed Attention, combining text and motion attention.

    Args:
        latent_dim (int): Dimension of the latent space for motion input.
        text_latent_dim (int): Dimension of the latent space for text input.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        time_embed_dim (int): Dimension of the time embedding.
    """

    def __init__(self,
                 latent_dim: int,
                 text_latent_dim: int,
                 num_heads: int,
                 dropout: float,
                 time_embed_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)

        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_text = nn.Linear(text_latent_dim, latent_dim)
        self.value_text = nn.Linear(text_latent_dim, latent_dim)
        self.key_motion = nn.Linear(latent_dim, latent_dim)
        self.value_motion = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        if time_embed_dim is not None:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
        else:
            self.proj_out = None

    def forward(self,
                x: torch.Tensor,
                xf: torch.Tensor,
                src_mask: torch.Tensor,
                emb: Optional[torch.Tensor] = None,
                cond_type: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass of Efficient Mixed Attention.

        Args:
            x (torch.Tensor): Input motion tensor of shape [B, T, D].
            xf (torch.Tensor): Input text tensor of shape [B, N, L].
            emb (torch.Tensor): Time embedding tensor of shape [B, D].
            src_mask (torch.Tensor): Source mask tensor of shape [B, T].
            cond_type (torch.Tensor): Conditioning type tensor.

        Returns:
            torch.Tensor: Output of the mixed attention module.
        """
        B, T, D = x.shape
        N = xf.shape[1] + x.shape[1]
        H = self.num_heads

        query = self.query(self.norm(x)).view(B, T, H, -1)

        text_cond_type = (cond_type % 10 > 0).float()
        src_mask = src_mask.view(B, T, 1)

        key_text = self.key_text(self.text_norm(xf))
        key_text = key_text + (1 - text_cond_type) * -1000000
        key_motion = self.key_motion(self.norm(x)) + (1 - src_mask) * -1000000
        key = torch.cat((key_text, key_motion), dim=1)

        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = self.dropout(F.softmax(key.view(B, N, H, -1), dim=1))

        value = torch.cat(
            (self.value_text(self.text_norm(xf)) * text_cond_type, self.value_motion(self.norm(x)) * src_mask),
            dim=1
        ).view(B, N, H, -1)

        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)

        if self.proj_out is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return y

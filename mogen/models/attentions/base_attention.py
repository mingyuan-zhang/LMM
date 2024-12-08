import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock


@ATTENTIONS.register_module()
class BaseMixedAttention(nn.Module):
    """
    Base class for Mixed Attention, combining text and motion attention.

    Args:
        latent_dim (int): Dimension of the latent space for motion input.
        text_latent_dim (int): Dimension of the latent space for text input.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        time_embed_dim (int): Dimension of the time embedding.
    """

    def __init__(self, latent_dim: int, text_latent_dim: int, num_heads: int, dropout: float, time_embed_dim: int):
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
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x: torch.Tensor, xf: torch.Tensor, emb: torch.Tensor, src_mask: torch.Tensor,
                cond_type: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass of Mixed Attention.

        Args:
            x (torch.Tensor): Input motion tensor of shape [B, T, D].
            xf (torch.Tensor): Input text tensor of shape [B, N, L].
            emb (torch.Tensor): Time embedding tensor of shape [B, D].
            src_mask (torch.Tensor): Source mask tensor of shape [B, T].
            cond_type (torch.Tensor): Conditioning type tensor of shape [B].

        Returns:
            torch.Tensor: Output of the mixed attention module.
        """
        B, T, D = x.shape
        N = xf.shape[1] + x.shape[1]
        H = self.num_heads

        query = self.query(self.norm(x)).view(B, T, H, -1)

        # Text conditioning type
        text_cond_type = ((cond_type % 10) > 0).float().view(B, 1, 1)
        text_cond_type = text_cond_type.repeat(1, xf.shape[1], 1)

        key = torch.cat(
            (self.key_text(self.text_norm(xf)), self.key_motion(self.norm(x))),
            dim=1
        ).view(B, N, H, -1)

        attention = torch.einsum('bnhl,bmhl->bnmh', query, key)

        motion_mask = src_mask.view(B, 1, T, 1)
        text_mask = text_cond_type.view(B, 1, -1, 1)
        mask = torch.cat((text_mask, motion_mask), dim=2)
        attention = attention + (1 - mask) * -1000000  # Masking for softmax
        attention = F.softmax(attention, dim=2)

        value = torch.cat(
            (self.value_text(self.text_norm(xf)) * text_cond_type, self.value_motion(self.norm(x)) * src_mask),
            dim=1
        ).view(B, N, H, -1)

        y = torch.einsum('bnmh,bmhl->bnhl', attention, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)

        return y


@ATTENTIONS.register_module()
class BaseSelfAttention(nn.Module):
    """
    Base class for Self-Attention mechanism.

    Args:
        latent_dim (int): Dimension of the latent space.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        time_embed_dim (Optional[int]): Dimension of the time embedding (optional).
    """

    def __init__(self, latent_dim: int, num_heads: int, dropout: float, time_embed_dim: Optional[int] = None):
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

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None, emb: Optional[torch.Tensor] = None, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass of Self-Attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D].
            emb (torch.Tensor): Time embedding tensor of shape [B, D].
            src_mask (torch.Tensor): Source mask tensor of shape [B, T].

        Returns:
            torch.Tensor: Output of the self-attention module.
        """
        B, T, D = x.shape
        H = self.num_heads

        query = self.query(self.norm(x)).view(B, T, H, -1)
        key = self.key(self.norm(x)).view(B, T, H, -1)

        attention = torch.einsum('bnhl,bmhl->bnmh', query, key)
        if src_mask is not None:
            mask = src_mask.view(B, 1, T, 1)
            attention = attention + (1 - mask) * -1000000  # Masking for softmax
        attention = F.softmax(attention, dim=2)

        if src_mask is not None:
            value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        else:
            value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhl->bnhl', attention, value).reshape(B, T, D)
        if self.time_embed_dim is None:
            y = x + y
        else:
            y = x + self.proj_out(y, emb)
        return y


@ATTENTIONS.register_module()
class BaseCrossAttention(nn.Module):
    """
    Base class for Cross-Attention mechanism, attending over text and motion inputs.

    Args:
        latent_dim (int): Dimension of the latent space for motion input.
        text_latent_dim (int): Dimension of the latent space for text input.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        time_embed_dim (int): Dimension of the time embedding.
    """

    def __init__(self, latent_dim: int, text_latent_dim: int, num_heads: int, dropout: float, time_embed_dim: int):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)

        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x: torch.Tensor, xf: torch.Tensor, emb: torch.Tensor, src_mask: torch.Tensor,
                cond_type: Optional[torch.Tensor] = None, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass of Cross-Attention.

        Args:
            x (torch.Tensor): Input motion tensor of shape [B, T, D].
            xf (torch.Tensor): Input text tensor of shape [B, N, L].
            emb (torch.Tensor): Time embedding tensor of shape [B, D].
            src_mask (torch.Tensor): Source mask tensor of shape [B, T].
            cond_type (Optional[torch.Tensor]): Conditioning type tensor of shape [B]. Defaults to None.

        Returns:
            torch.Tensor: Output of the cross-attention module.
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_heads

        query = self.query(self.norm(x)).view(B, T, H, -1)

        if cond_type is None:
            text_cond_type = 1
            mask = 1
        else:
            text_cond_type = ((cond_type % 10) > 0).float().view(B, 1, 1)
            text_cond_type = text_cond_type.repeat(1, xf.shape[1], 1)
            mask = text_cond_type.view(B, 1, -1, 1)

        key = self.key(self.text_norm(xf)).view(B, N, H, -1)

        attention = torch.einsum('bnhl,bmhl->bnmh', query, key)
        attention = attention + (1 - mask) * -1000000  # Masking for softmax
        attention = F.softmax(attention, dim=2)

        value = (self.value(self.text_norm(xf)) * text_cond_type).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhl->bnhl', attention, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)

        return y

import torch
from torch import nn
from typing import Optional

from mogen.models.utils.misc import zero_module
from mogen.models.utils.position_encoding import timestep_embedding
from mogen.models.utils.stylization_block import StylizationBlock

from ..builder import SUBMODULES, build_attention
from .remodiffuse import ReMoDiffuseTransformer


class FFN(nn.Module):
    """
    A feed-forward network (FFN) with optional stylization block.
    
    Args:
        latent_dim (int): The dimension of the input and output latent space.
        ffn_dim (int): The dimension of the hidden feed-forward network.
        dropout (float): The dropout rate to apply after activation.
        time_embed_dim (int): The dimension of the time embedding.
    """
    def __init__(self, latent_dim: int, ffn_dim: int, dropout: float, time_embed_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the FFN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, latent_dim*2).
            emb (torch.Tensor): Time embedding tensor.
        
        Returns:
            torch.Tensor: Output tensor after FFN and stylization block.
        """
        x1 = x[:, :, :self.latent_dim].contiguous()
        x2 = x[:, :, self.latent_dim:].contiguous()
        y1 = self.linear2(self.dropout(self.activation(self.linear1(x1))))
        y1 = x1 + self.proj_out(y1, emb)
        y2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        y2 = x2 + self.proj_out(y2, emb)
        y = torch.cat((y1, y2), dim=-1)
        return y


class DecoderLayer(nn.Module):
    """
    A single decoder layer consisting of a cross-attention block and a feed-forward network (FFN).
    
    Args:
        ca_block_cfg (Optional[dict]): Configuration for the cross-attention block.
        ffn_cfg (Optional[dict]): Configuration for the feed-forward network.
    """
    def __init__(self, ca_block_cfg: Optional[dict] = None, ffn_cfg: Optional[dict] = None):
        super().__init__()
        self.ca_block = build_attention(ca_block_cfg)
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass of the decoder layer.
        
        Args:
            **kwargs: Arguments passed to the cross-attention and FFN layers.
        
        Returns:
            torch.Tensor: Output tensor after passing through the layer.
        """
        if self.ca_block is not None:
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


@SUBMODULES.register_module()
class MoMatMoGenTransformer(ReMoDiffuseTransformer):
    """
    MoMatMoGenTransformer is a motion generation transformer model, which uses ReMoDiffuse as a base.
    
    Args:
        ReMoDiffuseTransformer: Base transformer class.
    """
    def build_temporal_blocks(self, sa_block_cfg: Optional[dict], ca_block_cfg: Optional[dict], ffn_cfg: Optional[dict]):
        """
        Build temporal decoder blocks using the provided configurations.
        
        Args:
            sa_block_cfg (Optional[dict]): Self-attention block configuration.
            ca_block_cfg (Optional[dict]): Cross-attention block configuration.
            ffn_cfg (Optional[dict]): Feed-forward network configuration.
        """
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                DecoderLayer(ca_block_cfg=ca_block_cfg, ffn_cfg=ffn_cfg))

    def forward(self,
                motion: torch.Tensor,
                timesteps: torch.Tensor,
                motion_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass for motion generation.
        
        Args:
            motion (torch.Tensor): Input motion tensor of shape (B, T, D).
            timesteps (torch.Tensor): Timestep embeddings.
            motion_mask (Optional[torch.Tensor]): Motion mask, if any.
        
        Returns:
            torch.Tensor: Output tensor after processing the motion data.
        """
        T = motion.shape[1]
        conditions = self.get_precompute_condition(device=motion.device,
                                                   **kwargs)
        if len(motion_mask.shape) == 2:
            src_mask = motion_mask.clone().unsqueeze(-1)
        else:
            src_mask = motion_mask.clone()

        if self.time_embedding_type == 'sinusoidal':
            emb = self.time_embed(
                timestep_embedding(timesteps, self.latent_dim))
        else:
            emb = self.time_embed(self.time_tokens(timesteps))

        if self.use_text_proj:
            emb = emb + conditions['xf_proj']

        motion1 = motion[:, :, :self.input_feats].contiguous()
        motion2 = motion[:, :, self.input_feats:].contiguous()
        h1 = self.joint_embed(motion1)
        h2 = self.joint_embed(motion2)
        if self.use_pos_embedding:
            h1 = h1 + self.sequence_embedding.unsqueeze(0)[:, :T, :]
            h2 = h2 + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        h = torch.cat((h1, h2), dim=-1)

        if self.training:
            output = self.forward_train(h=h,
                                        src_mask=src_mask,
                                        emb=emb,
                                        timesteps=timesteps,
                                        **conditions)
        else:
            output = self.forward_test(h=h,
                                       src_mask=src_mask,
                                       emb=emb,
                                       timesteps=timesteps,
                                       **conditions)
        if self.use_residual_connection:
            output = motion + output
        return output

    def forward_train(self,
                      h: Optional[torch.Tensor] = None,
                      src_mask: Optional[torch.Tensor] = None,
                      emb: Optional[torch.Tensor] = None,
                      xf_out: Optional[torch.Tensor] = None,
                      re_dict: Optional[dict] = None,
                      **kwargs) -> torch.Tensor:
        """
        Training forward pass for the motion generation transformer.
        
        Args:
            h (Optional[torch.Tensor]): Input tensor.
            src_mask (Optional[torch.Tensor]): Source mask.
            emb (Optional[torch.Tensor]): Embedding tensor.
            xf_out (Optional[torch.Tensor]): Output of the cross-attention block.
            re_dict (Optional[dict]): Dictionary for recurrent features.
        
        Returns:
            torch.Tensor: Output tensor after processing.
        """
        B, T = h.shape[0], h.shape[1]
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                       xf=xf_out,
                       emb=emb,
                       src_mask=src_mask,
                       cond_type=cond_type,
                       re_dict=re_dict)

        out1 = self.out(h[:, :, :self.latent_dim].contiguous())
        out1 = out1.view(B, T, -1).contiguous()
        out2 = self.out(h[:, :, self.latent_dim:].contiguous())
        out2 = out2.view(B, T, -1).contiguous()
        output = torch.cat((out1, out2), dim=-1)
        return output

    def forward_test(self,
                     h: Optional[torch.Tensor] = None,
                     src_mask: Optional[torch.Tensor] = None,
                     emb: Optional[torch.Tensor] = None,
                     xf_out: Optional[torch.Tensor] = None,
                     re_dict: Optional[dict] = None,
                     timesteps: Optional[torch.Tensor] = None,
                     **kwargs) -> torch.Tensor:
        """
        Testing forward pass for the motion generation transformer.
        
        Args:
            h (Optional[torch.Tensor]): Input tensor.
            src_mask (Optional[torch.Tensor]): Source mask.
            emb (Optional[torch.Tensor]): Embedding tensor.
            xf_out (Optional[torch.Tensor]): Output of the cross-attention block.
            re_dict (Optional[dict]): Dictionary for recurrent features.
            timesteps (Optional[torch.Tensor]): Timestep embeddings.
        
        Returns:
            torch.Tensor: Output tensor after processing.
        """
        B, T = h.shape[0], h.shape[1]
        both_cond_type = torch.zeros(B, 1, 1).to(h.device) + 99
        text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
        retr_cond_type = torch.zeros(B, 1, 1).to(h.device) + 10
        none_cond_type = torch.zeros(B, 1, 1).to(h.device)

        all_cond_type = torch.cat(
            (both_cond_type, text_cond_type, retr_cond_type, none_cond_type),
            dim=0)
        h = h.repeat(4, 1, 1)
        xf_out = xf_out.repeat(4, 1, 1)
        emb = emb.repeat(4, 1)
        src_mask = src_mask.repeat(4, 1, 1)
        if re_dict['re_motion'].shape[0] != h.shape[0]:
            re_dict['re_motion'] = re_dict['re_motion'].repeat(4, 1, 1, 1)
            re_dict['re_text'] = re_dict['re_text'].repeat(4, 1, 1, 1)
            re_dict['re_mask'] = re_dict['re_mask'].repeat(4, 1, 1)

        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                       xf=xf_out,
                       emb=emb,
                       src_mask=src_mask,
                       cond_type=all_cond_type,
                       re_dict=re_dict)

        out1 = self.out(h[:, :, :self.latent_dim].contiguous())
        out1 = out1.view(4 * B, T, -1).contiguous()
        out2 = self.out(h[:, :, self.latent_dim:].contiguous())
        out2 = out2.view(4 * B, T, -1).contiguous()
        out = torch.cat((out1, out2), dim=-1)
        out_both = out[:B].contiguous()
        out_text = out[B:2 * B].contiguous()
        out_retr = out[2 * B:3 * B].contiguous()
        out_none = out[3 * B:].contiguous()

        coef_cfg = self.scale_func(int(timesteps[0]))
        both_coef = coef_cfg['both_coef']
        text_coef = coef_cfg['text_coef']
        retr_coef = coef_cfg['retr_coef']
        none_coef = coef_cfg['none_coef']
        output = out_both * both_coef
        output += out_text * text_coef
        output += out_retr * retr_coef
        output += out_none * none_coef
        return output

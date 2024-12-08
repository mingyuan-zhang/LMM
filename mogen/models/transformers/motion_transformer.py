from abc import ABCMeta, abstractmethod
import clip
import torch
from torch import nn
from mmcv.runner import BaseModule

from ..builder import build_attention
from mogen.models.utils.position_encoding import (
    timestep_embedding
)
from mogen.models.utils.stylization_block import StylizationBlock
from mogen.models.utils.misc import set_requires_grad, zero_module


class CLIPWrapper:

    def __init__(self, clip_model):
        self.clip_model = clip_model
        self.device = "cpu"

    def __call__(self, **kwargs):
        return self.clip_model(**kwargs)

    def encode_text(self, text):
        if text.is_cuda and self.device == "cpu":
            self.clip_model = self.clip_model.cuda()
            self.device = "cuda"
        if not text.is_cuda and self.device == "cuda":
            self.clip_model = self.clip_model.cpu()
            self.device = "cpu"
        return self.clip_model.encode_text(text)

    def to(self, device):
        self.clip_model = self.clip_model.to(device)


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim=None):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if time_embed_dim is not None:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
        else:
            self.proj_out = None

    def forward(self, x, emb=None, **kwargs):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if self.proj_out is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return y


class DecoderLayer(nn.Module):

    def __init__(self,
                 sa_block_cfg=None,
                 ca_block_cfg=None,
                 ffn_cfg=None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ca_block = build_attention(ca_block_cfg)
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({'x': x})
        if self.ca_block is not None:
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


class MotionTransformer(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 input_feats,
                 max_seq_len=240,
                 latent_dim=512,
                 time_embed_dim=2048,
                 num_layers=8,
                 sa_block_cfg=None,
                 ca_block_cfg=None,
                 ffn_cfg=None,
                 text_encoder=None,
                 use_pos_embedding=True,
                 use_residual_connection=False,
                 time_embedding_type='sinusoidal',
                 post_process_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.input_feats = input_feats
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.use_pos_embedding = use_pos_embedding
        if self.use_pos_embedding:
            self.sequence_embedding = nn.Parameter(torch.randn(max_seq_len, latent_dim))
        self.build_text_encoder(text_encoder)

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.time_embedding_type = time_embedding_type
        if time_embedding_type != 'none':
            if time_embedding_type == 'learnable':
                self.time_tokens = nn.Embedding(1000, self.latent_dim)
            self.time_embed = nn.Sequential(
                nn.Linear(self.latent_dim, self.time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
            )
        self.build_temporal_blocks(sa_block_cfg, ca_block_cfg, ffn_cfg)

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))
        self.use_residual_connection = use_residual_connection
        self.post_process_cfg = post_process_cfg

    def build_temporal_blocks(self, sa_block_cfg, ca_block_cfg, ffn_cfg):
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                DecoderLayer(
                    sa_block_cfg=sa_block_cfg,
                    ca_block_cfg=ca_block_cfg,
                    ffn_cfg=ffn_cfg
                )
            )

    def build_text_encoder(self, text_encoder):
        if text_encoder is None:
            self.use_text_proj = False
            return
        text_latent_dim = text_encoder['latent_dim']
        num_text_layers = text_encoder.get('num_layers', 0)
        text_ff_size = text_encoder.get('ff_size', 2048)
        pretrained_model = text_encoder['pretrained_model']
        text_num_heads = text_encoder.get('num_heads', 4)
        dropout = text_encoder.get('dropout', 0)
        activation = text_encoder.get('activation', 'gelu')
        self.use_text_proj = text_encoder.get('use_text_proj', False)

        if pretrained_model == 'clip':
            clip_model, _ = clip.load('ViT-B/32', "cpu")
            set_requires_grad(clip_model, False)
            self.clip = CLIPWrapper(clip_model)
            if text_latent_dim != 512:
                self.text_pre_proj = nn.Linear(512, text_latent_dim)
            else:
                self.text_pre_proj = nn.Identity()
        else:
            raise NotImplementedError()

        if num_text_layers > 0:
            self.use_text_finetune = True
            textTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=text_latent_dim,
                nhead=text_num_heads,
                dim_feedforward=text_ff_size,
                dropout=dropout,
                activation=activation)
            self.textTransEncoder = nn.TransformerEncoder(
                textTransEncoderLayer,
                num_layers=num_text_layers)
        else:
            self.use_text_finetune = False
        self.text_ln = nn.LayerNorm(text_latent_dim)
        if self.use_text_proj:
            self.text_proj = nn.Sequential(
                nn.Linear(text_latent_dim, self.time_embed_dim)
            )

    def encode_text(self, text, clip_feat, device):
        B = len(text)
        if type(text[0]) is dict:
            knames = ["head", "stem", "left_arm", "right_arm", "left_leg", "right_leg", "pelvis", "all"]
            new_text = []
            for item in text:
                for kname in knames:
                    new_text.append(item[kname])
            text = new_text
        text = clip.tokenize(text, truncate=True).to(device)
        if clip_feat is None:
            with torch.no_grad():
                if isinstance(self.clip, CLIPWrapper):
                    self.clip.to(device)
                    dtype = self.clip.clip_model.dtype
                    # [batch_size, n_ctx, d_model]
                    x = self.clip.clip_model.token_embedding(text).type(dtype)
                    x = x + self.clip.clip_model.positional_embedding.type(dtype)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.clip.clip_model.transformer(x)
                    x = self.clip.clip_model.ln_final(x).type(dtype)
                else:
                    dtype = self.clip.dtype
                    # [batch_size, n_ctx, d_model]
                    x = self.clip.token_embedding(text).type(dtype)
                    x = x + self.clip.positional_embedding.type(dtype)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.clip.transformer(x)
                    x = self.clip.ln_final(x).type(dtype)
        else:
            x = clip_feat.float().to(device)
            if len(x.shape) == 4:
                x = x.permute(1, 0, 2, 3)
                x = x.reshape([x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
            else:
                x = x.permute(1, 0, 2)

        # T, B, D
        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        if self.use_text_proj:
            xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
            # B, T, D
            xf_out = xf_out.permute(1, 0, 2)
            return xf_proj, xf_out
        else:
            xf_out = xf_out.permute(1, 0, 2)
            return xf_out

    @abstractmethod
    def get_precompute_condition(self, **kwargs):
        pass

    @abstractmethod
    def forward_train(self, h, src_mask, emb, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, h, src_mask, emb, **kwargs):
        pass

    def forward(self,
                motion,
                timesteps=None,
                motion_mask=None,
                motion_length=None,
                num_intervals=1,
                **kwargs):
        """
        motion: B, T, D
        """
        B, T = motion.shape[0], motion.shape[1]
        conditions = self.get_precompute_condition(device=motion.device,
                                                   motion_length=motion_length,
                                                   **kwargs)
        if len(motion_mask.shape) == 2:
            src_mask = motion_mask.clone().unsqueeze(-1)
        else:
            src_mask = motion_mask.clone()

        if self.time_embedding_type != 'none':
            if self.time_embedding_type == 'sinusoidal':
                emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
            else:
                emb = self.time_embed(self.time_tokens(timesteps))

            if self.use_text_proj:
                emb = emb + conditions['xf_proj']
        else:
            emb = None
        # B, T, latent_dim
        h = self.joint_embed(motion)
        if self.use_pos_embedding:
            h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        if self.training:
            output = self.forward_train(
                h=h,
                src_mask=src_mask,
                emb=emb,
                timesteps=timesteps,
                motion_length=motion_length,
                num_intervals=num_intervals,
                motion=motion,
                **conditions)
        else:
            output = self.forward_test(
                h=h,
                src_mask=src_mask,
                emb=emb,
                timesteps=timesteps,
                motion_length=motion_length,
                num_intervals=num_intervals,
                **conditions)
        if self.use_residual_connection:
            output = motion + output
        return output

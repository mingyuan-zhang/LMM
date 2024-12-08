import clip
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mogen.models.utils.misc import set_requires_grad

from ..builder import SUBMODULES

loss_ce = nn.CrossEntropyLoss()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2).float() * \
            (-np.log(10000.0) / d_model)
        div_term = torch.exp(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)#.transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class MotionEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim, ff_size, num_layers, num_heads,
                 dropout, activation):
        super().__init__()

        self.input_feats = input_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats * 2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim,
                                                       self.dropout,
                                                       max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer,
                                                 num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    def forward(self, motion, motion_mask):
        x, mask = motion, motion_mask
        B, T = x.shape[:2]

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        idx = torch.zeros(B, dtype=torch.long, device=x.device)
        emb = torch.cat([self.query_token[idx][:, None], x_emb], dim=1)

        seq_mask = (mask > 0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)

        h = h.permute(1, 0, 2)
        h = self.transformer(h, src_key_padding_mask=~valid_mask).permute(
            1, 0, 2)
        h = self.out_ln(h)
        motion_emb = self.out(h[:, 0])

        return motion_emb


@SUBMODULES.register_module()
class InterCLIP(BaseModule):

    def __init__(self,
                 input_dim=258,
                 latent_dim=1024,
                 ff_size=2048,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 init_cfg=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.motion_encoder = MotionEncoder(input_dim=input_dim,
                                            latent_dim=latent_dim,
                                            ff_size=ff_size,
                                            num_layers=num_layers,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            activation=activation)

        self.latent_dim = self.latent_dim

        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        self.latent_scale = nn.Parameter(torch.Tensor([1]))

        set_requires_grad(self.token_embedding, False)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=ff_size,
            dropout=0.1,
            activation="gelu")
        self.textTransEncoder = nn.TransformerEncoder(textTransEncoderLayer,
                                                      num_layers=8)
        self.text_ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 512)

        self.clip_training = "text_"
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')
        assert init_cfg['type'] == 'Pretrained'
        self.load_pretrained(init_cfg['checkpoint'])

    def compute_loss(self, batch):
        losses = {}
        losses["total"] = 0

        # compute clip losses
        batch = self.encode_text(batch)
        batch = self.encode_motion(batch)

        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        losses.update(clip_losses)
        losses["total"] += mixed_clip_loss

        return losses["total"], losses

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def encode_motion(self,
                      motion,
                      motion_length=None,
                      motion_mask=None,
                      **kwargs):
        motion_emb = self.motion_encoder(motion, motion_mask)
        motion_emb = motion_emb / motion_emb.norm(dim=-1, keepdim=True)
        motion_emb = motion_emb * self.latent_scale
        return motion_emb

    def encode_text(self, text, device=None, **kwargs):
        raw_text = text
        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)
            pe_tokens = x + self.positional_embedding.type(self.dtype)

        pe_tokens = pe_tokens.permute(1, 0, 2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)

        out = self.text_ln(out)

        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out)

        text_emb = out
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb * self.latent_scale

        return text_emb

    def load_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if "model" in k:
                state_dict[k.replace("model.", "")] = state_dict.pop(k)
        self.load_state_dict(state_dict, strict=True)

import numpy as np
import torch
from torch import nn
import random
from typing import Optional, List, Dict

from mogen.models.utils.misc import zero_module
from ..builder import SUBMODULES, build_attention
from ..utils.stylization_block import StylizationBlock
from .motion_transformer import MotionTransformer
from mogen.models.utils.position_encoding import timestep_embedding
from scipy.ndimage import gaussian_filter


def get_tomato_slice(idx: int) -> List[int]:
    """Return specific slices for the tomato dataset."""
    if idx == 0:
        result = [0, 1, 2, 3, 463, 464, 465]
    else:
        result = [
            4 + (idx - 1) * 3,
            4 + (idx - 1) * 3 + 1,
            4 + (idx - 1) * 3 + 2,
            157 + (idx - 1) * 6,
            157 + (idx - 1) * 6 + 1,
            157 + (idx - 1) * 6 + 2,
            157 + (idx - 1) * 6 + 3,
            157 + (idx - 1) * 6 + 4,
            157 + (idx - 1) * 6 + 5,
            463 + idx * 3,
            463 + idx * 3 + 1,
            463 + idx * 3 + 2,
        ]
    return result


def get_part_slice(idx_list: List[int], func) -> List[int]:
    """Return a list of slices by applying the provided function."""
    result = []
    for idx in idx_list:
        result.extend(func(idx))
    return result


class SinglePoseEncoder(nn.Module):
    """Encoder module for individual pose, separating different body parts."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        func = get_tomato_slice
        self.root_slice = get_part_slice([0], func)
        self.head_slice = get_part_slice([12, 15], func)
        self.stem_slice = get_part_slice([3, 6, 9], func)
        self.larm_slice = get_part_slice([14, 17, 19, 21], func)
        self.rarm_slice = get_part_slice([13, 16, 18, 20], func)
        self.lleg_slice = get_part_slice([2, 5, 8, 11], func)
        self.rleg_slice = get_part_slice([1, 4, 7, 10], func)
        self.lhnd_slice = get_part_slice(range(22, 37), func)
        self.rhnd_slice = get_part_slice(range(37, 52), func)
        self.face_slice = range(619, 669)

        # Initialize linear layers for each body part embedding
        self.root_embed = nn.Linear(len(self.root_slice), latent_dim)
        self.head_embed = nn.Linear(len(self.head_slice), latent_dim)
        self.stem_embed = nn.Linear(len(self.stem_slice), latent_dim)
        self.larm_embed = nn.Linear(len(self.larm_slice), latent_dim)
        self.rarm_embed = nn.Linear(len(self.rarm_slice), latent_dim)
        self.lleg_embed = nn.Linear(len(self.lleg_slice), latent_dim)
        self.rleg_embed = nn.Linear(len(self.rleg_slice), latent_dim)
        self.lhnd_embed = nn.Linear(len(self.lhnd_slice), latent_dim)
        self.rhnd_embed = nn.Linear(len(self.rhnd_slice), latent_dim)
        self.face_embed = nn.Linear(len(self.face_slice), latent_dim)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """Forward pass to embed different parts of the motion tensor."""
        root_feat = self.root_embed(motion[:, :, self.root_slice].contiguous())
        head_feat = self.head_embed(motion[:, :, self.head_slice].contiguous())
        stem_feat = self.stem_embed(motion[:, :, self.stem_slice].contiguous())
        larm_feat = self.larm_embed(motion[:, :, self.larm_slice].contiguous())
        rarm_feat = self.rarm_embed(motion[:, :, self.rarm_slice].contiguous())
        lleg_feat = self.lleg_embed(motion[:, :, self.lleg_slice].contiguous())
        rleg_feat = self.rleg_embed(motion[:, :, self.rleg_slice].contiguous())
        lhnd_feat = self.lhnd_embed(motion[:, :, self.lhnd_slice].contiguous())
        rhnd_feat = self.rhnd_embed(motion[:, :, self.rhnd_slice].contiguous())
        face_feat = self.face_embed(motion[:, :, self.face_slice].contiguous())

        # Concatenate all embeddings
        feat = torch.cat((root_feat, head_feat, stem_feat,
                          larm_feat, rarm_feat, lleg_feat, rleg_feat,
                          lhnd_feat, rhnd_feat, face_feat), dim=-1)
        return feat


class PoseEncoder(nn.Module):
    """Encoder for multi-dataset scenarios, handling different datasets."""

    def __init__(self, latent_dim: int, num_datasets: int):
        super().__init__()
        self.models = nn.ModuleList()
        self.num_datasets = num_datasets
        self.latent_dim = latent_dim

        # Initialize single pose encoders for each dataset
        for _ in range(num_datasets):
            self.models.append(SinglePoseEncoder(latent_dim=latent_dim))

    def forward(self, motion: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-dataset encoding."""
        B, T = motion.shape[:2]
        output = torch.zeros(B, T, 10 * self.latent_dim).type_as(motion)
        num_finish = 0

        # Process each dataset's motion separately
        for i in range(self.num_datasets):
            batch_motion = motion[dataset_idx == i]
            if len(batch_motion) == 0:
                continue
            num_finish += len(batch_motion)
            batch_motion = self.models[i](batch_motion)
            output[dataset_idx == i] = batch_motion
        assert num_finish == B
        return output


class SinglePoseDecoder(nn.Module):
    """Decoder module for individual pose, reconstructing body parts."""

    def __init__(self, latent_dim: int = 64, output_dim: int = 669):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        func = get_tomato_slice
        self.root_slice = get_part_slice([0], func)
        self.head_slice = get_part_slice([12, 15], func)
        self.stem_slice = get_part_slice([3, 6, 9], func)
        self.larm_slice = get_part_slice([14, 17, 19, 21], func)
        self.rarm_slice = get_part_slice([13, 16, 18, 20], func)
        self.lleg_slice = get_part_slice([2, 5, 8, 11], func)
        self.rleg_slice = get_part_slice([1, 4, 7, 10], func)
        self.lhnd_slice = get_part_slice(range(22, 37), func)
        self.rhnd_slice = get_part_slice(range(37, 52), func)
        self.face_slice = range(619, 669)

        # Initialize linear layers for each body part output
        self.root_out = nn.Linear(latent_dim, len(self.root_slice))
        self.head_out = nn.Linear(latent_dim, len(self.head_slice))
        self.stem_out = nn.Linear(latent_dim, len(self.stem_slice))
        self.larm_out = nn.Linear(latent_dim, len(self.larm_slice))
        self.rarm_out = nn.Linear(latent_dim, len(self.rarm_slice))
        self.lleg_out = nn.Linear(latent_dim, len(self.lleg_slice))
        self.rleg_out = nn.Linear(latent_dim, len(self.rleg_slice))
        self.lhnd_out = nn.Linear(latent_dim, len(self.lhnd_slice))
        self.rhnd_out = nn.Linear(latent_dim, len(self.rhnd_slice))
        self.face_out = nn.Linear(latent_dim, len(self.face_slice))
        

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """Forward pass to decode body parts from latent representation."""
        B, T = motion.shape[:2]
        D = self.latent_dim

        # Decode each part using corresponding linear layer
        root_feat = self.root_out(motion[:, :, :D].contiguous())
        head_feat = self.head_out(motion[:, :, D: 2 * D].contiguous())
        stem_feat = self.stem_out(motion[:, :, 2 * D: 3 * D].contiguous())
        larm_feat = self.larm_out(motion[:, :, 3 * D: 4 * D].contiguous())
        rarm_feat = self.rarm_out(motion[:, :, 4 * D: 5 * D].contiguous())
        lleg_feat = self.lleg_out(motion[:, :, 5 * D: 6 * D].contiguous())
        rleg_feat = self.rleg_out(motion[:, :, 6 * D: 7 * D].contiguous())
        lhnd_feat = self.lhnd_out(motion[:, :, 7 * D: 8 * D].contiguous())
        rhnd_feat = self.rhnd_out(motion[:, :, 8 * D: 9 * D].contiguous())
        face_feat = self.face_out(motion[:, :, 9 * D:].contiguous())

        # Combine outputs into final tensor
        output = torch.zeros(B, T, self.output_dim).type_as(motion)
        output[:, :, self.root_slice] = root_feat
        output[:, :, self.head_slice] = head_feat
        output[:, :, self.stem_slice] = stem_feat
        output[:, :, self.larm_slice] = larm_feat
        output[:, :, self.rarm_slice] = rarm_feat
        output[:, :, self.lleg_slice] = lleg_feat
        output[:, :, self.rleg_slice] = rleg_feat
        output[:, :, self.lhnd_slice] = lhnd_feat
        output[:, :, self.rhnd_slice] = rhnd_feat
        output[:, :, self.face_slice] = face_feat

        return output


class PoseDecoder(nn.Module):
    """Decoder for multi-dataset scenarios, handling different datasets."""

    def __init__(self, latent_dim: int, output_dim: int, num_datasets: int):
        super().__init__()
        self.models = nn.ModuleList()
        self.num_datasets = num_datasets
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Initialize single pose decoders for each dataset
        for _ in range(num_datasets):
            self.models.append(
                SinglePoseDecoder(latent_dim=latent_dim, output_dim=output_dim)
            )

    def forward(self, motion: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-dataset decoding."""
        B, T = motion.shape[:2]
        output = torch.zeros(B, T, self.output_dim).type_as(motion)
        num_finish = 0

        # Process each dataset's motion separately
        for i in range(self.num_datasets):
            batch_motion = motion[dataset_idx == i]
            if len(batch_motion) == 0:
                continue
            num_finish += len(batch_motion)
            batch_motion = self.models[i](batch_motion)
            output[dataset_idx == i] = batch_motion
        assert num_finish == B
        return output


class SFFN(nn.Module):
    """SFFN module with multiple linear layers, acting on different parts of the input."""

    def __init__(self,
                 latent_dim: int,
                 ffn_dim: int,
                 dropout: float,
                 time_embed_dim: int,
                 activation: str = "GELU"):
        super().__init__()
        self.linear1_list = nn.ModuleList()
        self.linear2_list = nn.ModuleList()

        if activation == "GELU":
            self.activation = nn.GELU()
        self.linear1 = nn.Linear(latent_dim * 10, ffn_dim * 10)
        self.linear2 = nn.Linear(ffn_dim * 10, latent_dim * 10)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim * 10, time_embed_dim, dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for SFFN, applying stylization block."""
        B, T, D = x.shape
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x.reshape(B, T, D) + self.proj_out(y, emb)

        return y


class FFN(nn.Module):
    """Feed-forward network with GELU activation and dropout."""

    def __init__(self, latent_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with normalization and residual connection."""
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + y
        return y


class DecoderLayer(nn.Module):
    """Decoder layer consisting of conditional attention block and SFFN."""

    def __init__(self, ca_block_cfg: Optional[Dict] = None, ffn_cfg: Optional[Dict] = None):
        super().__init__()
        self.ca_block = build_attention(ca_block_cfg) if ca_block_cfg else None
        self.ffn = SFFN(**ffn_cfg) if ffn_cfg else None

    def forward(self, **kwargs) -> torch.Tensor:
        """Forward pass for the decoder layer."""
        if self.ca_block is not None:
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


class EncoderLayer(nn.Module):
    """Encoder layer consisting of self-attention block and FFN."""

    def __init__(self, sa_block_cfg: Optional[Dict] = None, ffn_cfg: Optional[Dict] = None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg) if sa_block_cfg else None
        self.ffn = FFN(**ffn_cfg) if ffn_cfg else None

    def forward(self, **kwargs) -> torch.Tensor:
        """Forward pass for the encoder layer."""
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x

class Transformer(nn.Module):
    """Transformer model with self-attention and feed-forward network layers."""

    def __init__(self,
                 input_dim: int = 1024,
                 latent_dim: int = 1024,
                 num_heads: int = 10,
                 num_layers: int = 4,
                 max_seq_len: int = 300,
                 stride: int = 1,
                 dropout: float = 0):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.proj_in = nn.Linear(input_dim, latent_dim)
        self.embedding = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))
        self.latent_dim = latent_dim
        self.stride = stride
        self.num_heads = num_heads
        self.dropout = dropout
        
        sa_block_cfg = dict(
            type='EfficientSelfAttention',
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        ffn_cfg = dict(
            latent_dim=latent_dim,
            ffn_dim=latent_dim * 4,
            dropout=dropout
        )
        for _ in range(num_layers):
            self.blocks.append(
                EncoderLayer(sa_block_cfg=sa_block_cfg, ffn_cfg=ffn_cfg)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer layers."""
        x = x[:, ::self.stride, :]
        x = self.proj_in(x)
        T = x.shape[1]
        x = x + self.embedding[:, :T, :]
        # Apply each encoder layer
        for block in self.blocks:
            x = block(x=x)

        return x


@SUBMODULES.register_module()
class LargeMotionModel(MotionTransformer):
    """Large motion model with optional multi-modal conditioning (text, music, video, etc.)."""

    def __init__(self,
                 num_parts: int = 10,
                 latent_part_dim: int = 64,
                 num_cond_layers: int = 2,
                 num_datasets: int = 27,
                 guidance_cfg: Optional[Dict] = None,
                 moe_route_loss_weight: float = 1.0,
                 template_kl_loss_weight: float = 0.0001,
                 dataset_names: Optional[List[str]] = None,
                 text_input_dim: Optional[int] = None,
                 music_input_dim: Optional[int] = None,
                 speech_input_dim: Optional[int] = None,
                 video_input_dim: Optional[int] = None,
                 music_input_stride: Optional[int] = 3,
                 speech_input_stride: Optional[int] = 3,
                 cond_drop_rate: float = 0,
                 random_mask: float = 0,
                 dropout: float = 0,
                 **kwargs):
        kwargs['latent_dim'] = latent_part_dim * num_parts
        self.num_parts = num_parts
        self.latent_part_dim = latent_part_dim
        self.num_datasets = num_datasets
        self.dropout = dropout
        
        super().__init__(**kwargs)
        self.guidance_cfg = guidance_cfg

        self.joint_embed = PoseEncoder(
            latent_dim=self.latent_part_dim,
            num_datasets=self.num_datasets)
        self.out = zero_module(PoseDecoder(
            latent_dim=self.latent_part_dim,
            output_dim=self.input_feats,
            num_datasets=self.num_datasets))

        self.dataset_proj = {name: i for i, name in enumerate(dataset_names or [])}
        self.rotation_proj = {'h3d_rot': 0, 'smpl_rot': 1, 'bvh_rot': 2}

        self.moe_route_loss_weight = moe_route_loss_weight
        self.template_kl_loss_weight = template_kl_loss_weight
        self.cond_drop_rate = cond_drop_rate

        # Conditional transformers for multi-modal inputs
        self.text_cond = text_input_dim is not None
        self.music_cond = music_input_dim is not None
        self.speech_cond = speech_input_dim is not None
        self.video_cond = video_input_dim is not None

        if self.text_cond:
            self.text_transformer = Transformer(
                input_dim=text_input_dim,
                latent_dim=self.latent_dim,
                num_heads=self.num_parts,
                num_layers=num_cond_layers,
                dropout=self.dropout)
        if self.music_cond:
            self.music_transformer = Transformer(
                input_dim=music_input_dim,
                latent_dim=self.latent_dim,
                num_heads=self.num_parts,
                num_layers=num_cond_layers,
                dropout=self.dropout,
                stride=music_input_stride)
        if self.speech_cond:
            self.speech_transformer = Transformer(
                input_dim=speech_input_dim,
                latent_dim=self.latent_dim,
                num_heads=self.num_parts,
                num_layers=num_cond_layers,
                dropout=self.dropout,
                stride=speech_input_stride)
        if self.video_cond:
            self.video_transformer = Transformer(
                input_dim=video_input_dim,
                latent_dim=self.latent_dim,
                num_heads=self.num_parts,
                num_layers=num_cond_layers,
                dropout=self.dropout)

        self.mask_token = nn.Parameter(torch.randn(self.num_parts, self.latent_part_dim))
        self.clean_token = nn.Parameter(torch.randn(self.num_parts, self.latent_part_dim))
        self.random_mask = random_mask

    def build_temporal_blocks(self,
                              sa_block_cfg: Optional[Dict] = None, 
                              ca_block_cfg: Optional[Dict] = None,
                              ffn_cfg: Optional[Dict] = None):
        """Build temporal decoder blocks with attention and feed-forward networks."""
        self.temporal_decoder_blocks = nn.ModuleList()
        ca_block_cfg['latent_dim'] = self.latent_part_dim
        ca_block_cfg['num_heads'] = self.num_parts
        ca_block_cfg['ffn_dim'] = self.latent_part_dim * 4
        ca_block_cfg['time_embed_dim'] = self.time_embed_dim
        ca_block_cfg['max_seq_len'] = self.max_seq_len
        ca_block_cfg['dropout'] = self.dropout
        for _ in range(self.num_layers):
            ffn_cfg_block = dict(
                latent_dim=self.latent_part_dim,
                ffn_dim=self.latent_part_dim * 4,
                dropout=self.dropout,
                time_embed_dim=self.time_embed_dim
            )
            self.temporal_decoder_blocks.append(
                DecoderLayer(ca_block_cfg=ca_block_cfg, ffn_cfg=ffn_cfg_block)
            )

    def scale_func(self, timestep: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """Scale function for diffusion, adjusting weights based on timestep."""
        guidance_cfg = self.guidance_cfg[dataset_name]
        if guidance_cfg['type'] == 'constant':
            w = torch.ones_like(timestep).float() * guidance_cfg['scale']
        elif guidance_cfg['type'] == 'linear':
            scale = guidance_cfg['scale']
            w = (1 - (1000 - timestep) / 1000) * scale + 1
        else:
            raise NotImplementedError()
        return w

    def aux_loss(self) -> Dict[str, torch.Tensor]:
        """Compute auxiliary and KL losses for multi-modal routing."""
        aux_loss = 0
        kl_loss = 0
        for module in self.temporal_decoder_blocks:
            if hasattr(module.ca_block, 'aux_loss'):
                aux_loss += module.ca_block.aux_loss
            if hasattr(module.ca_block, 'kl_loss'):
                kl_loss += module.ca_block.kl_loss
        losses = {}
        if aux_loss > 0:
            losses['moe_route_loss'] = aux_loss * self.moe_route_loss_weight
        if kl_loss > 0:
            losses['template_kl_loss'] = kl_loss * self.template_kl_loss_weight
        return losses

    def get_precompute_condition(self,
                                 text_word_feat: Optional[torch.Tensor] = None,
                                 text_word_out: Optional[torch.Tensor] = None,
                                 text_cond: Optional[torch.Tensor] = None,
                                 music_word_feat: Optional[torch.Tensor] = None,
                                 music_word_out: Optional[torch.Tensor] = None,
                                 music_cond: Optional[torch.Tensor] = None,
                                 speech_word_feat: Optional[torch.Tensor] = None,
                                 speech_word_out: Optional[torch.Tensor] = None,
                                 speech_cond: Optional[torch.Tensor] = None,
                                 video_word_feat: Optional[torch.Tensor] = None,
                                 video_word_out: Optional[torch.Tensor] = None,
                                 video_cond: Optional[torch.Tensor] = None,
                                 **kwargs) -> Dict[str, torch.Tensor]:
        """Precompute conditions for various modalities (text, music, speech, video)."""
        output = {}
        if self.text_cond and text_word_feat is not None:
            text_word_feat = text_word_feat.float()
            if text_word_out is None:
                if text_cond is None or torch.sum(text_cond) == 0:
                    latent_dim = self.text_transformer.latent_dim
                    B, N = text_word_feat.shape[:2]
                    text_word_out = torch.zeros(B, N, latent_dim).type_as(text_word_feat)
                else:
                    text_word_out = self.text_transformer(text_word_feat)
            output['text_word_out'] = text_word_out
        if self.music_cond and music_word_feat is not None:
            music_word_feat = music_word_feat.float()
            if music_word_out is None:
                if music_cond is None or torch.sum(music_cond) == 0:
                    latent_dim = self.music_transformer.latent_dim
                    B, N = music_word_feat.shape[:2]
                    music_word_out = torch.zeros(B, N, latent_dim).type_as(music_word_feat)
                else:
                    music_word_out = self.music_transformer(music_word_feat)
            output['music_word_out'] = music_word_out
        if self.speech_cond and speech_word_feat is not None:
            speech_word_feat = speech_word_feat.float()
            if speech_word_out is None:
                if speech_cond is None or torch.sum(speech_cond) == 0:
                    latent_dim = self.speech_transformer.latent_dim
                    B, N = speech_word_feat.shape[:2]
                    speech_word_out = torch.zeros(B, N, latent_dim).type_as(speech_word_feat)
                else:
                    speech_word_out = self.speech_transformer(speech_word_feat)
            output['speech_word_out'] = speech_word_out
        if self.video_cond and video_word_feat is not None:
            video_word_feat = video_word_feat.float()
            if video_word_out is None:
                if video_cond is None or torch.sum(video_cond) == 0:
                    latent_dim = self.video_transformer.latent_dim
                    B, N = video_word_feat.shape[:2]
                    video_word_out = torch.zeros(B, N, latent_dim).type_as(video_word_feat)
                else:
                    video_word_out = self.video_transformer(video_word_feat)
            output['video_word_out'] = video_word_out
        return output

    def post_process(self, motion: torch.Tensor) -> torch.Tensor:
        """Post-process motion data (e.g., unnormalization)."""
        if self.post_process_cfg is not None and self.post_process_cfg.get("unnormalized_infer", False):
            mean = torch.from_numpy(np.load(self.post_process_cfg['mean_path'])).type_as(motion)
            std = torch.from_numpy(np.load(self.post_process_cfg['std_path'])).type_as(motion)
            motion = motion * std + mean
        return motion

    def forward_train(self,
                      h: torch.Tensor,
                      src_mask: torch.Tensor,
                      emb: torch.Tensor,
                      timesteps: torch.Tensor,
                      motion_length: Optional[torch.Tensor] = None,
                      text_word_out: Optional[torch.Tensor] = None,
                      text_cond: Optional[torch.Tensor] = None,
                      music_word_out: Optional[torch.Tensor] = None,
                      music_cond: Optional[torch.Tensor] = None,
                      speech_word_out: Optional[torch.Tensor] = None,
                      speech_cond: Optional[torch.Tensor] = None,
                      video_word_out: Optional[torch.Tensor] = None,
                      video_cond: Optional[torch.Tensor] = None,
                      num_intervals: int = 1,
                      duration: Optional[torch.Tensor] = None,
                      dataset_idx: Optional[torch.Tensor] = None,
                      rotation_idx: Optional[torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        """Forward pass for training, applying multi-modal conditions."""
        B, T = h.shape[:2]
        # Apply conditional masking if needed
        if self.text_cond and text_cond is not None:
            text_cond_mask = torch.rand(B).type_as(h)
            text_cond[text_cond_mask < self.cond_drop_rate] = 0
        if self.music_cond and music_cond is not None:
            music_cond_mask = torch.rand(B).type_as(h)
            music_cond[music_cond_mask < self.cond_drop_rate] = 0
        if self.speech_cond and speech_cond is not None:
            speech_cond_mask = torch.rand(B).type_as(h)
            speech_cond[speech_cond_mask < self.cond_drop_rate] = 0
        if self.video_cond and video_cond is not None:
            video_cond_mask = torch.rand(B).type_as(h)
            video_cond[video_cond_mask < self.cond_drop_rate] = 0

        # Apply each temporal decoder block
        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                       emb=emb,
                       src_mask=src_mask,
                       motion_length=motion_length,
                       text_cond=text_cond,
                       text_word_out=text_word_out,
                       music_cond=music_cond,
                       music_word_out=music_word_out,
                       speech_cond=speech_cond,
                       speech_word_out=speech_word_out,
                       video_cond=video_cond,
                       video_word_out=video_word_out,
                       num_intervals=num_intervals,
                       duration=duration,
                       dataset_idx=dataset_idx,
                       rotation_idx=rotation_idx)

        # Output layer
        output = self.out(h, dataset_idx).view(B, T, -1).contiguous()
        return output

    def forward_test(self,
                     h: torch.Tensor,
                     src_mask: torch.Tensor,
                     emb: torch.Tensor,
                     timesteps: torch.Tensor,
                     motion_length: torch.Tensor,
                     text_word_out: Optional[torch.Tensor] = None,
                     text_cond: Optional[torch.Tensor] = None,
                     music_word_out: Optional[torch.Tensor] = None,
                     music_cond: Optional[torch.Tensor] = None,
                     speech_word_out: Optional[torch.Tensor] = None,
                     speech_cond: Optional[torch.Tensor] = None,
                     video_word_out: Optional[torch.Tensor] = None,
                     video_cond: Optional[torch.Tensor] = None,
                     num_intervals: int = 1,
                     duration: Optional[torch.Tensor] = None,
                     dataset_idx: Optional[torch.Tensor] = None,
                     rotation_idx: Optional[torch.Tensor] = None,
                     dataset_name: Optional[str] = 'humanml3d_t2m',
                     **kwargs) -> torch.Tensor:
        """Forward pass for testing, including scaling and conditional fusion."""
        B, T = h.shape[:2]
        # Duplicate tensors for conditional and non-conditional cases
        h = h.repeat(2, 1, 1)
        emb = emb.repeat(2, 1)
        src_mask = src_mask.repeat(2, 1, 1, 1)
        motion_length = motion_length.repeat(2, 1)
        duration = duration.repeat(2)
        
        # dataset_idx_att = [self.dataset_proj['all'] for i in range(B)]
        # dataset_idx_att = torch.tensor(dataset_idx_att, dtype=torch.long).to(h.device)
        # dataset_idx_att = torch.cat((dataset_idx, dataset_idx_att))
        dataset_idx = dataset_idx.repeat(2)
        rotation_idx = rotation_idx.repeat(2)
        
        if self.text_cond and text_cond is not None and text_word_out is not None:
            text_cond = text_cond.repeat(2, 1)
            text_cond[B:] = 0
            text_word_out = text_word_out.repeat(2, 1, 1)
        if self.music_cond and music_cond is not None and music_word_out is not None:
            music_cond = music_cond.repeat(2, 1)
            music_cond[B:] = 0
            music_word_out = music_word_out.repeat(2, 1, 1)
        if self.speech_cond and speech_cond is not None and speech_word_out is not None:
            speech_cond = speech_cond.repeat(2, 1)
            speech_cond[B:] = 0
            speech_word_out = speech_word_out.repeat(2, 1, 1)
        if self.video_cond and video_cond is not None and video_word_out is not None:
            video_cond = video_cond.repeat(2, 1)
            video_cond[B:] = 0
            video_word_out = video_word_out.repeat(2, 1, 1)

        # Apply each temporal decoder block
        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                       emb=emb,
                       src_mask=src_mask,
                       motion_length=motion_length,
                       text_cond=text_cond,
                       text_word_out=text_word_out,
                       music_cond=music_cond,
                       music_word_out=music_word_out,
                       speech_cond=speech_cond,
                       speech_word_out=speech_word_out,
                       video_cond=video_cond,
                       video_word_out=video_word_out,
                       num_intervals=num_intervals,
                       duration=duration,
                       dataset_idx=dataset_idx,
                       rotation_idx=rotation_idx)

        # Process the output from conditional and non-conditional branches
        output = self.out(h, dataset_idx).view(2 * B, T, -1).contiguous()
        scale = self.scale_func(timesteps, dataset_name).view(-1, 1, 1)
        output_cond = output[:B].contiguous()
        output_none = output[B:].contiguous()

        # Fuse conditional and non-conditional outputs
        output = output_cond * scale + output_none * (1 - scale)
        return output

    def create_mask_from_length(self, T: int, motion_length: torch.Tensor) -> torch.Tensor:
        """Create a binary mask based on motion length."""
        B = motion_length.shape[0]
        src_mask = torch.zeros(B, T)
        for bix in range(B):
            src_mask[bix, :int(motion_length[bix])] = 1
        return src_mask

    def forward(self,
                motion: torch.Tensor,
                timesteps: torch.Tensor,
                motion_mask: Optional[torch.Tensor] = None,
                motion_length: Optional[torch.Tensor] = None,
                num_intervals: int = 1,
                motion_metas: Optional[List[Dict]] = None,
                text_seq_feat: Optional[torch.Tensor] = None,
                text_word_feat: Optional[torch.Tensor] = None,
                text_cond: Optional[torch.Tensor] = None,
                music_seq_feat: Optional[torch.Tensor] = None,
                music_word_feat: Optional[torch.Tensor] = None,
                music_cond: Optional[torch.Tensor] = None,
                speech_seq_feat: Optional[torch.Tensor] = None,
                speech_word_feat: Optional[torch.Tensor] = None,
                speech_cond: Optional[torch.Tensor] = None,
                video_seq_feat: Optional[torch.Tensor] = None,
                video_word_feat: Optional[torch.Tensor] = None,
                video_cond: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Unified forward pass for both training and testing."""
        B, T = motion.shape[:2]
        # Precompute conditioning features
        conditions = self.get_precompute_condition(
            motion_length=motion_length,
            text_seq_feat=text_seq_feat,
            text_word_feat=text_word_feat,
            text_cond=text_cond,
            music_seq_feat=music_seq_feat,
            music_word_feat=music_word_feat,
            music_cond=music_cond,
            speech_seq_feat=speech_seq_feat,
            speech_word_feat=speech_word_feat,
            speech_cond=speech_cond,
            video_seq_feat=video_seq_feat,
            video_word_feat=video_word_feat,
            video_cond=video_cond,
            device=motion.device,
            **kwargs
        )
        if self.training:
            new_motion_mask = motion_mask.clone()
            rand_mask = torch.rand_like(motion_mask)
            threshold = torch.rand(B).type_as(rand_mask)
            threshold = threshold.view(B, 1, 1).repeat(1, T, self.num_parts)
            new_motion_mask[rand_mask < threshold] = 0
            motion_mask = new_motion_mask
        else:
            t = int(timesteps[0])

        motion_mask = motion_mask.view(B, T, 10, 1)
        
        # Temporal embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        
        # Prepare duration and framerate embeddings
        duration = []
        for meta in motion_metas:
            framerate = meta['meta_data']['framerate']
            duration.append(1.0 / framerate)

        duration = torch.tensor(duration, dtype=motion.dtype).to(motion.device)

        # Dataset index embedding
        dataset_idx = []
        for i in range(B):
            dataset_name = motion_metas[i]['meta_data']['dataset_name']
            if torch.rand(1).item() < 0.1 and self.training:
                dataset_name = 'all'
            idx = self.dataset_proj[dataset_name]
            dataset_idx.append(idx)
        dataset_idx = torch.tensor(dataset_idx, dtype=torch.long).to(motion.device)
        self.dataset_idx = dataset_idx.clone().detach()
        
        # Rotation index embedding
        rotation_idx = [self.rotation_proj[meta['meta_data']['rotation_type']] for meta in motion_metas]
        rotation_idx = torch.tensor(rotation_idx, dtype=torch.long).to(motion.device)

        # Embed motion with pose encoder
        h = self.joint_embed(motion, dataset_idx)
        h = h.view(B, T, 10, -1) * motion_mask + (1 - motion_mask) * self.mask_token
        h = h.view(B, T, -1)

        # Source mask based on motion length
        src_mask = self.create_mask_from_length(T, motion_length).to(motion.device)
        src_mask = src_mask.view(B, T, 1, 1).repeat(1, 1, 10, 1)

        # Training or testing forward
        if self.training:
            output = self.forward_train(
                h=h,
                emb=emb,
                src_mask=src_mask,
                timesteps=timesteps,
                motion_length=motion_length,
                text_cond=text_cond,
                music_cond=music_cond,
                speech_cond=speech_cond,
                video_cond=video_cond,
                num_intervals=num_intervals,
                duration=duration,
                dataset_idx=dataset_idx,
                rotation_idx=rotation_idx,
                **conditions
            )
        else:
            output = self.forward_test(
                h=h,
                emb=emb,
                src_mask=src_mask,
                timesteps=timesteps,
                motion_length=motion_length,
                text_cond=text_cond,
                music_cond=music_cond,
                speech_cond=speech_cond,
                video_cond=video_cond,
                num_intervals=num_intervals,
                duration=duration,
                dataset_idx=dataset_idx,
                rotation_idx=rotation_idx,
                dataset_name=dataset_name,
                **conditions
            )

        return output

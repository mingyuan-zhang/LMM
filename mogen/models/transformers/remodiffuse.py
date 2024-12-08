import random
import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from typing import List, Dict, Optional, Union

from mogen.models.utils.misc import zero_module

from ..builder import SUBMODULES, build_attention
from .motion_transformer import MotionTransformer


class FFN(nn.Module):
    """
    Feed-forward network (FFN) used in the transformer layers. 
    It consists of two linear layers with a GELU activation in between.

    Args:
        latent_dim (int): Input dimension of the FFN.
        ffn_dim (int): Hidden dimension of the FFN.
        dropout (float): Dropout rate applied after activation.
    """

    def __init__(self, latent_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward pass for the FFN.

        Args:
            x (Tensor): Input tensor of shape (B, T, D).

        Returns:
            Tensor: Output tensor after the FFN, of shape (B, T, D).
        """
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + y
        return y


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting of self-attention and feed-forward network.

    Args:
        sa_block_cfg (Optional[dict]): Configuration for the self-attention block.
        ca_block_cfg (Optional[dict]): Configuration for the cross-attention block (if applicable).
        ffn_cfg (dict): Configuration for the feed-forward network.
    """

    def __init__(self, sa_block_cfg: Optional[dict] = None, ca_block_cfg: Optional[dict] = None, ffn_cfg: dict = None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs) -> Tensor:
        """
        Forward pass for the encoder layer.

        Args:
            kwargs: Dictionary containing the input tensor (x) and other related parameters.

        Returns:
            Tensor: Output tensor after the encoder layer.
        """
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


class RetrievalDatabase(nn.Module):
    """
    Retrieval database for retrieving motions and text features based on given captions.

    Args:
        num_retrieval (int): Number of retrievals for each caption.
        topk (int): Number of top results to consider.
        retrieval_file (str): Path to the retrieval file containing text, motion, and length data.
        latent_dim (Optional[int]): Dimension of the latent space.
        output_dim (Optional[int]): Output dimension of the retrieved features.
        num_layers (Optional[int]): Number of layers in the text encoder.
        num_motion_layers (Optional[int]): Number of layers in the motion encoder.
        kinematic_coef (Optional[float]): Coefficient for scaling kinematic similarity.
        max_seq_len (Optional[int]): Maximum sequence length.
        num_heads (Optional[int]): Number of attention heads.
        ff_size (Optional[int]): Feed-forward size for the transformer layers.
        stride (Optional[int]): Stride for downsampling motion data.
        sa_block_cfg (Optional[dict]): Configuration for the self-attention block.
        ffn_cfg (Optional[dict]): Configuration for the feed-forward network.
        dropout (Optional[float]): Dropout rate.
    """

    def __init__(self,
                 num_retrieval: int,
                 topk: int,
                 retrieval_file: str,
                 latent_dim: Optional[int] = 512,
                 output_dim: Optional[int] = 512,
                 num_layers: Optional[int] = 2,
                 num_motion_layers: Optional[int] = 4,
                 kinematic_coef: Optional[float] = 0.1,
                 max_seq_len: Optional[int] = 196,
                 num_heads: Optional[int] = 8,
                 ff_size: Optional[int] = 1024,
                 stride: Optional[int] = 4,
                 sa_block_cfg: Optional[dict] = None,
                 ffn_cfg: Optional[dict] = None,
                 dropout: Optional[float] = 0):
        super().__init__()
        self.num_retrieval = num_retrieval
        self.topk = topk
        self.latent_dim = latent_dim
        self.stride = stride
        self.kinematic_coef = kinematic_coef
        self.num_layers = num_layers
        self.num_motion_layers = num_motion_layers
        self.max_seq_len = max_seq_len

        # Load data from the retrieval file
        data = np.load(retrieval_file)
        self.text_features = torch.Tensor(data['text_features'])
        self.captions = data['captions']
        self.motions = data['motions']
        self.m_lengths = data['m_lengths']
        self.clip_seq_features = data['clip_seq_features']
        self.train_indexes = data.get('train_indexes', None)
        self.test_indexes = data.get('test_indexes', None)

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.motion_proj = nn.Linear(self.motions.shape[-1], self.latent_dim)
        self.motion_pos_embedding = nn.Parameter(
            torch.randn(max_seq_len, self.latent_dim))
        self.motion_encoder_blocks = nn.ModuleList()

        # Build motion encoder blocks
        for i in range(num_motion_layers):
            self.motion_encoder_blocks.append(
                EncoderLayer(sa_block_cfg=sa_block_cfg, ffn_cfg=ffn_cfg))

        # Transformer for encoding text
        TransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                       nhead=num_heads,
                                                       dim_feedforward=ff_size,
                                                       dropout=dropout,
                                                       activation="gelu")
        self.text_encoder = nn.TransformerEncoder(TransEncoderLayer,
                                                  num_layers=num_layers)
        self.results = {}

    def extract_text_feature(self, text: str, clip_model: nn.Module, device: torch.device) -> Tensor:
        """
        Extract text features from CLIP model.

        Args:
            text (str): Input text caption.
            clip_model (nn.Module): CLIP model for encoding the text.
            device (torch.device): Device for computation.

        Returns:
            Tensor: Extracted text features of shape (1, 512).
        """
        text = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        return text_features

    def encode_text(self, text: List[str], device: torch.device) -> Tensor:
        """
        Encode text using the CLIP model's text encoder.

        Args:
            text (List[str]): List of input text captions.
            device (torch.device): Device for computation.

        Returns:
            Tensor: Encoded text features of shape (B, T, D).
        """
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        # B, T, D
        xf_out = x.permute(1, 0, 2)
        return xf_out

    def retrieve(self, caption: str, length: int, clip_model: nn.Module, device: torch.device, idx: Optional[int] = None) -> List[int]:
        """
        Retrieve motions and text features based on a given caption.

        Args:
            caption (str): Input text caption.
            length (int): Length of the corresponding motion sequence.
            clip_model (nn.Module): CLIP model for encoding the text.
            device (torch.device): Device for computation.
            idx (Optional[int]): Index for retrieval (if provided).

        Returns:
            List[int]: List of indexes for the retrieved motions.
        """
        value = hash(caption)
        if value in self.results:
            return self.results[value]
        text_feature = self.extract_text_feature(caption, clip_model, device)

        rel_length = torch.LongTensor(self.m_lengths).to(device)
        rel_length = torch.abs(rel_length - length)
        rel_length = rel_length / torch.clamp(rel_length, min=length)
        semantic_score = F.cosine_similarity(self.text_features.to(device),
                                             text_feature)
        kinematic_score = torch.exp(-rel_length * self.kinematic_coef)
        score = semantic_score * kinematic_score
        indexes = torch.argsort(score, descending=True)
        data = []
        cnt = 0
        for idx in indexes:
            caption, m_length = self.captions[idx], self.m_lengths[idx]
            if not self.training or m_length != length:
                cnt += 1
                data.append(idx.item())
                if cnt == self.num_retrieval:
                    self.results[value] = data
                    return data
        assert False

    def generate_src_mask(self, T: int, length: List[int]) -> Tensor:
        """
        Generate source mask for the motion sequences based on the motion lengths.

        Args:
            T (int): Maximum sequence length.
            length (List[int]): List of motion lengths for each sample.

        Returns:
            Tensor: A binary mask tensor of shape (B, T), where `B` is the batch size, 
            and `T` is the maximum sequence length. Mask values are 1 for valid positions 
            and 0 for padded positions.
        """
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, captions: List[str], lengths: List[int], clip_model: nn.Module, device: torch.device, idx: Optional[List[int]] = None) -> Dict[str, Tensor]:
        """
        Forward pass for retrieving motion sequences and text features.

        Args:
            captions (List[str]): List of input text captions.
            lengths (List[int]): List of corresponding motion lengths.
            clip_model (nn.Module): CLIP model for encoding the text.
            device (torch.device): Device for computation.
            idx (Optional[List[int]]): Optional list of indices for retrieval.

        Returns:
            Dict[str, Tensor]: Dictionary containing retrieved text and motion features.
            - re_text: Retrieved text features of shape (B, num_retrieval, T, D).
            - re_motion: Retrieved motion features of shape (B, num_retrieval, T, D).
            - re_mask: Source mask for the retrieved motion of shape (B, num_retrieval, T).
            - raw_motion: Raw motion features of shape (B, T, motion_dim).
            - raw_motion_length: Motion sequence lengths (before any stride).
            - raw_motion_mask: Raw binary mask for valid motion positions of shape (B, T).
        """
        B = len(captions)
        all_indexes = []
        for b_ix in range(B):
            length = int(lengths[b_ix])
            if idx is None:
                batch_indexes = self.retrieve(captions[b_ix], length, clip_model, device)
            else:
                batch_indexes = self.retrieve(captions[b_ix], length, clip_model, device, idx[b_ix])
            all_indexes.extend(batch_indexes)

        all_indexes = np.array(all_indexes)
        all_motions = torch.Tensor(self.motions[all_indexes]).to(device)
        all_m_lengths = torch.Tensor(self.m_lengths[all_indexes]).long()

        # Generate masks and positional encodings
        T = all_motions.shape[1]
        src_mask = self.generate_src_mask(T, all_m_lengths).to(device)
        raw_src_mask = src_mask.clone()
        re_motion = self.motion_proj(all_motions) + self.motion_pos_embedding.unsqueeze(0)

        for module in self.motion_encoder_blocks:
            re_motion = module(x=re_motion, src_mask=src_mask.unsqueeze(-1))

        re_motion = re_motion.view(B, self.num_retrieval, T, -1).contiguous()
        re_motion = re_motion[:, :, ::self.stride, :].contiguous()  # Apply stride
        src_mask = src_mask[:, ::self.stride].contiguous()
        src_mask = src_mask.view(B, self.num_retrieval, -1).contiguous()

        # Process text sequences
        T = 77  # CLIP's max token length
        all_text_seq_features = torch.Tensor(self.clip_seq_features[all_indexes]).to(device)
        all_text_seq_features = all_text_seq_features.permute(1, 0, 2)
        re_text = self.text_encoder(all_text_seq_features)
        re_text = re_text.permute(1, 0, 2)
        re_text = re_text.view(B, self.num_retrieval, T, -1).contiguous()
        re_text = re_text[:, :, -1:, :].contiguous()  # Use the last token only for each sequence

        re_dict = {
            're_text': re_text,
            're_motion': re_motion,
            're_mask': src_mask,
            'raw_motion': all_motions,
            'raw_motion_length': all_m_lengths,
            'raw_motion_mask': raw_src_mask
        }
        return re_dict


@SUBMODULES.register_module()
class ReMoDiffuseTransformer(MotionTransformer):
    """
    Transformer model for motion retrieval and diffusion.

    Args:
        retrieval_cfg (dict): Configuration for the retrieval database.
        scale_func_cfg (dict): Configuration for scaling functions.
        kwargs: Additional arguments for the base DiffusionTransformer.
    """

    def __init__(self, retrieval_cfg: dict, scale_func_cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.database = RetrievalDatabase(**retrieval_cfg)
        self.scale_func_cfg = scale_func_cfg

    def scale_func(self, timestep: int) -> Dict[str, float]:
        """
        Scale function for adjusting the guidance between text and retrieval.

        Args:
            timestep (int): Current diffusion timestep.

        Returns:
            Dict[str, float]: Scaling coefficients for different guidance types.
            - both_coef: Coefficient for both text and retrieval guidance.
            - text_coef: Coefficient for text-only guidance.
            - retr_coef: Coefficient for retrieval-only guidance.
            - none_coef: Coefficient for no guidance.
        """
        coarse_scale = self.scale_func_cfg['coarse_scale']
        w = (1 - (1000 - timestep) / 1000) * coarse_scale + 1
        if timestep > 100:
            if random.randint(0, 1) == 0:
                output = {
                    'both_coef': w,
                    'text_coef': 0,
                    'retr_coef': 1 - w,
                    'none_coef': 0
                }
            else:
                output = {
                    'both_coef': 0,
                    'text_coef': w,
                    'retr_coef': 0,
                    'none_coef': 1 - w
                }
        else:
            both_coef = self.scale_func_cfg['both_coef']
            text_coef = self.scale_func_cfg['text_coef']
            retr_coef = self.scale_func_cfg['retr_coef']
            none_coef = 1 - both_coef - text_coef - retr_coef
            output = {
                'both_coef': both_coef,
                'text_coef': text_coef,
                'retr_coef': retr_coef,
                'none_coef': none_coef
            }
        return output

    def get_precompute_condition(self, 
                                 text: Optional[str] = None,
                                 motion_length: Optional[Tensor] = None,
                                 xf_out: Optional[Tensor] = None,
                                 re_dict: Optional[Dict] = None,
                                 device: Optional[torch.device] = None,
                                 sample_idx: Optional[Tensor] = None,
                                 clip_feat: Optional[Tensor] = None,
                                 **kwargs) -> Dict[str, Union[Tensor, Dict]]:
        """
        Precompute conditions for both text and retrieval-guided diffusion.

        Args:
            text (Optional[str]): Input text string for guidance.
            motion_length (Optional[Tensor]): Lengths of the motion sequences.
            xf_out (Optional[Tensor]): Encoded text feature (if precomputed).
            re_dict (Optional[Dict]): Dictionary of retrieval results (if precomputed).
            device (Optional[torch.device]): Device to perform computation on.
            sample_idx (Optional[Tensor]): Sample indices for retrieval.
            clip_feat (Optional[Tensor]): Clip features (if used).

        Returns:
            Dict[str, Union[Tensor, Dict]]: Dictionary containing encoded features and retrieval results.
        """
        if xf_out is None:
            xf_out = self.encode_text(text, clip_feat, device)
        output = {'xf_out': xf_out}
        if re_dict is None:
            re_dict = self.database(text, motion_length, self.clip, device, idx=sample_idx)
        output['re_dict'] = re_dict
        return output

    def post_process(self, motion: Tensor) -> Tensor:
        """
        Post-process the generated motion by normalizing or un-normalizing it.

        Args:
            motion (Tensor): Generated motion data.

        Returns:
            Tensor: Post-processed motion data.
        """
        if self.post_process_cfg is not None:
            if self.post_process_cfg.get("unnormalized_infer", False):
                mean = torch.from_numpy(np.load(self.post_process_cfg['mean_path'])).type_as(motion)
                std = torch.from_numpy(np.load(self.post_process_cfg['std_path'])).type_as(motion)
                motion = motion * std + mean
        return motion

    def forward_train(self,
                      h: Tensor,
                      src_mask: Tensor,
                      emb: Tensor,
                      xf_out: Optional[Tensor] = None,
                      re_dict: Optional[Dict] = None,
                      **kwargs) -> Tensor:
        """
        Forward training pass for motion retrieval and diffusion model.

        Args:
            h (Tensor): Input motion features of shape (B, T, D).
            src_mask (Tensor): Mask for the motion data of shape (B, T, 1).
            emb (Tensor): Embedding tensor for timesteps.
            xf_out (Optional[Tensor]): Precomputed text features.
            re_dict (Optional[Dict]): Dictionary of retrieval features.

        Returns:
            Tensor: Output motion data of shape (B, T, D).
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

        output = self.out(h).view(B, T, -1).contiguous()
        return output

    def forward_test(self,
                     h: Tensor,
                     src_mask: Tensor,
                     emb: Tensor,
                     xf_out: Optional[Tensor] = None,
                     re_dict: Optional[Dict] = None,
                     timesteps: Optional[Tensor] = None,
                     **kwargs) -> Tensor:
        """
        Forward testing pass for motion retrieval and diffusion model. This method handles
        multiple conditional types such as both text and retrieval-based guidance.

        Args:
            h (Tensor): Input motion features of shape (B, T, D).
            src_mask (Tensor): Mask for the motion data of shape (B, T, 1).
            emb (Tensor): Embedding tensor for timesteps.
            xf_out (Optional[Tensor]): Precomputed text features.
            re_dict (Optional[Dict]): Dictionary of retrieval features.
            timesteps (Optional[Tensor]): Tensor containing current timesteps in the diffusion process.

        Returns:
            Tensor: Output motion data after applying multiple conditional types, of shape (B, T, D).
        """
        B, T = h.shape[0], h.shape[1]
        
        # Define condition types for different guidance types
        both_cond_type = torch.zeros(B, 1, 1).to(h.device) + 99
        text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
        retr_cond_type = torch.zeros(B, 1, 1).to(h.device) + 10
        none_cond_type = torch.zeros(B, 1, 1).to(h.device)

        # Concatenate all conditional types and repeat inputs for different guidance modes
        all_cond_type = torch.cat((both_cond_type, text_cond_type, retr_cond_type, none_cond_type), dim=0)
        h = h.repeat(4, 1, 1)
        xf_out = xf_out.repeat(4, 1, 1)
        emb = emb.repeat(4, 1)
        src_mask = src_mask.repeat(4, 1, 1)

        # Repeat retrieval features if necessary
        if re_dict['re_motion'].shape[0] != h.shape[0]:
            re_dict['re_motion'] = re_dict['re_motion'].repeat(4, 1, 1, 1)
            re_dict['re_text'] = re_dict['re_text'].repeat(4, 1, 1, 1)
            re_dict['re_mask'] = re_dict['re_mask'].repeat(4, 1, 1)

        # Pass through the temporal decoder blocks
        for module in self.temporal_decoder_blocks:
            h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=all_cond_type, re_dict=re_dict)

        # Retrieve output features and handle different guidance coefficients
        out = self.out(h).view(4 * B, T, -1).contiguous()
        out_both = out[:B].contiguous()
        out_text = out[B:2 * B].contiguous()
        out_retr = out[2 * B:3 * B].contiguous()
        out_none = out[3 * B:].contiguous()

        # Apply scaling coefficients based on the timestep
        coef_cfg = self.scale_func(int(timesteps[0]))
        both_coef = coef_cfg['both_coef']
        text_coef = coef_cfg['text_coef']
        retr_coef = coef_cfg['retr_coef']
        none_coef = coef_cfg['none_coef']

        # Compute the final output by blending the different guidance outputs
        output = out_both * both_coef
        output += out_text * text_coef
        output += out_retr * retr_coef
        output += out_none * none_coef

        return output

